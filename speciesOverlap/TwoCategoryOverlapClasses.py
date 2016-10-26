import gc
import sys
import os

from time import time
from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy as sp
import scipy.sparse as sprs

import cPickle as pickle

from speciesOverlap.OverlapClasses import OverlapCalculator
from speciesOverlap.utilities import update_progress

from itertools import izip

class TwoCategoryOverlapCalculator(OverlapCalculator):

    def __init__(self,pond_species_matrix,glade_species_matrix,weighted=False,int_to_pond=None,int_to_glade=None,pond_to_int=None,glade_to_int=None,verbose=False,delete_original_matrix=False):

        self.verbose = verbose

        new_pond_species_matrix = sprs.vstack([pond_species_matrix,glade_species_matrix])

        self.N_ponds = pond_species_matrix.shape[0]
        self.N_glades = glade_species_matrix.shape[0]

        OverlapCalculator.__init__(self,
                                   new_pond_species_matrix,
                                   weighted=weighted,
                                   verbose = verbose,
                                   delete_original_matrix = True,
                                   )

        self.pond_to_int = pond_to_int
        self.int_to_pond = int_to_pond

        self.glade_to_int = glade_to_int
        self.int_to_glade = int_to_glade

        self.is_twocategory = True


    def get_overlap_matrix_single(self):

        # get dot product of each chunk submatrix with existence matrix
        if self.verbose:
            print "calculating first overlap matrix..."

        W = self.new_pond_species_matrix[:self.N_ponds,:].dot(self.existence_matrix[self.N_ponds:,:].T)

        # finalize calculation
        if self.weighted:
            if self.verbose:
                print "calculating second overlap matrix..."

            Wg = self.new_pond_species_matrix[self.N_ponds:,:].dot( self.existence_matrix[:self.N_ponds,:].T )

            del self.new_pond_species_matrix
            del self.existence_matrix
            gc.collect()

            self.overlap_matrix = W.multiply( Wg.T ) 

            del W
            del Wg
            gc.collect()
        else:
            del self.new_pond_species_matrix
            del self.existence_matrix
            gc.collect()

            self.overlap_matrix = W

        return self.overlap_matrix

    def save_ovcalc(self,filename):

        row, col = self.overlap_matrix.nonzero()
        data = self.overlap_matrix.data
        pickle.dump({
                        'row': row, 
                        'col': col,
                        'data': data,
                        'pond_to_int': self.pond_to_int,
                        'glade_to_int': self.glade_to_int,
                        'int_to_pond': self.int_to_pond,
                        'int_to_glade': self.int_to_glade,
                        'N_ponds': self.N_ponds,
                        'N_glades': self.N_glades,
                        'k_pond': self.k_pond,
                    },
                    open(filename,'wb')
                    )

class NumpyDictTwoCategoryOverlapCalculator(TwoCategoryOverlapCalculator):

    def __init__(self,data_dict_ponds,data_dict_glades,verbose=False,delete_original_data=False):
        """ requires data_list to be a list like 
            [ (pond_identifier, species_identifier, weight), (..., ), ... ]
        """

        for key,val in data_dict_ponds.iteritems():
            if type(val) in [tuple,list]:
                weighted = True
            elif type(val) == np.ndarray:
                weighted = False
            else:
                raise ValueError("Unexpected type:", type(val),'. Expected 2 or 3.')
            break

        data_dicts = [ data_dict_ponds, data_dict_glades ]
        self.species_to_int = {}
        self.int_to_species = {}

        ponds_to_ints = []
        ints_to_ponds = []

        matrix_data = []

        len_ponds = len(data_dict_ponds)
        len_glades = len(data_dict_glades)
        len_all = len_ponds + len_glades

        species_counter = np.iinfo(np.uint64).max

        if verbose:
            times = []
            entry_count = 1
            update_progress(0,len_all,times,status="converting data to sparse matrix")

        p_cols = []
        p_rows = []
        p_data = []
        


        for i_dl,data_dict in enumerate(data_dicts): 
            row = []
            col = []
            data = []

            pond_counter = 0

            # convert pond names to integers and vice versa
            int_to_pond = { i:k for i,k in izip(xrange(len(data_dict)), data_dict.iterkeys()) }
            pond_to_int = { k:i for i,k in izip(xrange(len(data_dict)), data_dict.iterkeys()) }

            if verbose:
                start = time()

            for pond,entry in data_dict.iteritems():

                if weighted:
                    species_array = entry[0]
                    data_array = entry[1]
                else:
                    species_array = entry
                    data_array = np.ones_like(species_array)

                current_pond_int = pond_to_int[pond]

                row.append(current_pond_int*np.ones_like(data_array,dtype=np.uint32))
                col.append(species_array)
                data.append(data_array)

                if verbose:
                    end = time()
                    times.append(end-start)
                    update_progress(entry_count,len_all,times,status="preparing data for sparse conversion")
                    start = time()
                    entry_count += 1

            row = np.concatenate(row)
            col = np.concatenate(col)
            data = np.concatenate(data)

            p_cols.append(col)
            p_rows.append(row)
            p_data.append(data)

            # sort_ndcs = np.argsort(row)

            # row = row[sort_ndcs]
            # col = col[sort_ndcs]
            # data = data[sort_ndcs]
            # matrix_data.append( ( (data,(row,col)), len(int_to_pond)) ) 
            ints_to_ponds.append(int_to_pond)
            ponds_to_ints.append(pond_to_int)

        if verbose:
            start = time()
            print "find unique species"

        all_species = np.unique( np.concatenate( p_cols ) )
        species_counter = len(all_species)


        if verbose:
            end = time()
            print "took %d seconds" % (end-start)
            print "get species dict" 

        species_to_int = { all_species[i]: i for i in xrange(species_counter) }

        if verbose:
            end = time()
            print "took %d seconds" % (end-start)
            print "convert column vectors"
            start = time()

        new_p_cols = []
        for c in p_cols:
            new_col = np.array([ species_to_int[sp] for sp in c ])
            new_p_cols.append(new_col)
        
        if verbose:
            end = time()
            print "took %d seconds" % (end-start)
            print "pond matrix conversion to csr sparse"
            start = time()

        pond_species_matrix = sprs.csr_matrix((p_data[0],(p_rows[0],new_p_cols[0])),shape=(len_ponds,species_counter))

        if verbose:
            end = time()
            print "took %d seconds" % (end-start)
            print "glade matrix conversion to csr sparse"
            start = time()

        glade_species_matrix = sprs.csr_matrix((p_data[1],(p_rows[1],new_p_cols[1])),shape=(len_glades,species_counter))

        if verbose:
            end = time()
            print "took %d seconds" % (end-start)

        TwoCategoryOverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   glade_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=ints_to_ponds[0],
                                   int_to_glade=ints_to_ponds[1],
                                   pond_to_int=ponds_to_ints[0],
                                   glade_to_int=ponds_to_ints[1],
                                   verbose = verbose,
                                   delete_original_matrix = True
                                   )

        del matrix_data

    def old_init_slow(self,data_dict_ponds,data_dict_glades,verbose=False,delete_original_data=False,maxint=np.iinfo(np.uint64).max):
        """ requires data_list to be a list like 
            [ (pond_identifier, species_identifier, weight), (..., ), ... ]
        """

        for key,val in data_dict_ponds.iteritems():
            if type(val) in [tuple,list]:
                weighted = True
            elif type(val) == np.ndarray:
                weighted = False
            else:
                raise ValueError("Unexpected type:", type(val),'. Expected 2 or 3.')
            break

        data_dicts = [ data_dict_ponds, data_dict_glades ]
        self.species_to_int = {}
        self.int_to_species = {}
        species_counter = 0

        ponds_to_ints = []
        ints_to_ponds = []

        matrix_data = []

        len_ponds = len(data_dict_ponds)
        len_glades = len(data_dict_glades)
        len_all = len_ponds + len_glades

        if verbose:
            times = []
            entry_count = 1
            update_progress(0,len_all,times,status="converting data to sparse matrix")


        for i_dl,data_dict in enumerate(data_dicts): 
            row = []
            col = []
            data = []

            pond_counter = 0

            # convert pond names to integers and vice versa
            int_to_pond = { i:k for i,k in izip(xrange(len(data_dict)), data_dict.iterkeys()) }
            pond_to_int = { k:i for i,k in izip(xrange(len(data_dict)), data_dict.iterkeys()) }

            if verbose:
                start = time()

            for pond,entry in data_dict.iteritems():

                if weighted:
                    species_array = entry[0]
                    data_array = entry[1]
                else:
                    species_array = entry
                    data_array = np.ones_like(species_array)

                current_pond_int = pond_to_int[pond]

                for species,dat in izip(species_array,data_array):

                    if species not in self.species_to_int:
                        self.species_to_int[species] = species_counter
                        self.int_to_species[species_counter] = species
                        current_species_int = species_counter
                        species_counter += 1
                    else:
                        current_species_int = self.species_to_int[species]

                    row.append(current_pond_int)
                    col.append(current_species_int)
                    data.append(dat)

                if verbose:
                    end = time()
                    times.append(end-start)
                    update_progress(entry_count,len_all,times,status="converting data to sparse matrix")
                    start = time()
                    entry_count += 1

            row = np.array(row,dtype=np.int32)
            col = np.array(col,dtype=np.int32)
            data = np.array(data,dtype=np.float32)

            # sort_ndcs = np.argsort(row)

            # row = row[sort_ndcs]
            # col = col[sort_ndcs]
            # data = data[sort_ndcs]
            matrix_data.append( ( (data,(row,col)), len(int_to_pond)) ) 
            ints_to_ponds.append(int_to_pond)
            ponds_to_ints.append(pond_to_int)

        pond_species_matrix = sprs.csr_matrix(matrix_data[0][0],shape=(matrix_data[0][1],species_counter))
        glade_species_matrix = sprs.csr_matrix(matrix_data[1][0],shape=(matrix_data[1][1],species_counter))

        TwoCategoryOverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   glade_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=ints_to_ponds[0],
                                   int_to_glade=ints_to_ponds[1],
                                   pond_to_int=ponds_to_ints[0],
                                   glade_to_int=ponds_to_ints[1],
                                   verbose = verbose,
                                   delete_original_matrix = True
                                   )

        del matrix_data


class TupleListTwoCategoryOverlapCalculator(TwoCategoryOverlapCalculator):

    def __init__(self,data_list_ponds,data_list_glades,verbose=False,delete_original_data=False):
        """ requires data_list to be a list like 
            [ (pond_identifier, species_identifier, weight), (..., ), ... ]
        """

        if len(data_list_ponds[0]) == 2:
            weighted = False
        elif len(data_list_ponds[0]) == 3:
            weighted = True
        else:
            raise ValueError("Unexpected value size of data_list:", len(data_list[0]),'. Expected 2 or 3.')

        data_lists = [ data_list_ponds, data_list_glades ]
        self.species_to_int = {}
        self.int_to_species = {}
        species_counter = 0

        ints_to_ponds = []
        ponds_to_ints = []

        matrix_data = []

        len_ponds = len(data_list_ponds)
        len_glades = len(data_list_glades)
        len_all = len_ponds + len_glades

        if verbose and len_all>20000:
            times = []
            entry_count = 1
            chunk_size = 5000
            n_chunks = int(np.ceil(len_all/float(chunk_size)))


        for i_dl,data_list in enumerate(data_lists): 
            row = []
            col = []
            data = []

            pond_counter = 0

            pond_to_int = {}
            int_to_pond = {}

            if verbose:
                start = time()
                times = []

            for entry in data_list:
                pond = entry[0]
                species = entry[1]

                if weighted:
                    dat = entry[2]
                else:
                    dat = 1

                if pond not in pond_to_int:
                    pond_to_int[pond] = pond_counter
                    int_to_pond[pond_counter] = pond
                    current_pond_int = pond_counter
                    pond_counter += 1
                else:
                    current_pond_int = pond_to_int[pond]

                if species not in self.species_to_int:
                    self.species_to_int[species] = species_counter
                    self.int_to_species[species_counter] = species
                    current_species_int = species_counter
                    species_counter += 1
                else:
                    current_species_int = self.species_to_int[species]

                row.append(current_pond_int)
                col.append(current_species_int)
                data.append(dat)

                if verbose and len_all>20000 and (entry_count % chunk_size == 0 or entry_count == len_all):
                    end = time()
                    times.append(end-start)
                    update_progress(entry_count/chunk_size,n_chunks,times,status="converting data to sparse matrix")
                    start = time()

                if verbose and len_all>20000:
                    entry_count += 1

            row = np.array(row,dtype=np.int32)
            col = np.array(col,dtype=np.int32)
            data = np.array(data,dtype=np.float32)

            # sort_ndcs = np.argsort(row)

            # row = row[sort_ndcs]
            # col = col[sort_ndcs]
            # data = data[sort_ndcs]
            matrix_data.append( ( (data,(row,col)), pond_counter) ) 
            ints_to_ponds.append(int_to_pond)
            ponds_to_ints.append(pond_to_int)

        pond_species_matrix = sprs.csr_matrix(matrix_data[0][0],shape=(matrix_data[0][1],species_counter))
        glade_species_matrix = sprs.csr_matrix(matrix_data[1][0],shape=(matrix_data[1][1],species_counter))

        if delete_original_data:
            del data_lists[0]
            del data_lists[0]
            gc.collect()

        TwoCategoryOverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   glade_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=ints_to_ponds[0],
                                   int_to_glade=ints_to_ponds[1],
                                   pond_to_int=ponds_to_ints[0],
                                   glade_to_int=ponds_to_ints[1],
                                   verbose = verbose,
                                   delete_original_matrix = True
                                   )

        del matrix_data




if __name__=="__main__":


    from simind import abundance_based_similarity as AbSim
    from simind import incidence_based_similarity as InSim
    
    A = np.array( [ 
                    [ 2, 3, 0, 0, 0 ],
                    [ 0, 1, 2, 3, 0 ],
                    [ 1, 0, 2, 3, 3 ],
                    #[ 3, 2, 0, 0 ],
                ])

    print A

    print "TEST"

    pond_0 = A[0,:].nonzero()[0]
    pond_1 = A[1,:].nonzero()[0]
    pond_2 = A[2,:].nonzero()[0]
    w_0 = { sp: A[0,sp] for sp in pond_0 }
    w_1 = { sp: A[1,sp] for sp in pond_1 }
    w_2 = { sp: A[2,sp] for sp in pond_2 }

    print (0,0), AbSim(pond_0, pond_0, w_0, w_0).S_AB
    print (0,1), AbSim(pond_0, pond_1, w_0, w_1).S_AB
    print (0,2), AbSim(pond_0, pond_2, w_0, w_2).S_AB
    print (1,0), AbSim(pond_1, pond_0, w_1, w_0).S_AB
    print (1,1), AbSim(pond_1, pond_1, w_1, w_1).S_AB
    print (1,2), AbSim(pond_1, pond_2, w_1, w_2).S_AB

    print "TUPLE LIST TEST"
    data = [ 
            ( 'a', '0', 2 ),
            ( 'a', '1', 3 ),
            ( 'b', '1', 1 ),
            ( 'b', '2', 2 ),
            ( 'b', '3', 3 ),
            ( 'c', '0', 1 ),
            ( 'c', '2', 2 ),
            ( 'c', '3', 3 ),
            ( 'c', '4', 3 ),
           ]
    OvCalc = TupleListTwoCategoryOverlapCalculator(data[:5],data[5:],verbose=True)
    OvCalc.get_overlap_matrix_single()

    print OvCalc.overlap_matrix.toarray()

    ponds = { 
              'a': np.array([0,1]),
              'b': np.array([1,2,3])
            }

    glades = { 
              'c': np.array([0,2,3,4])
             }

    OvCalc = NumpyDictTwoCategoryOverlapCalculator(ponds,glades,verbose=True)
    OvCalc.get_overlap_matrix_single()

    print OvCalc.overlap_matrix.toarray()

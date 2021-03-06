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

from speciesOverlap.utilities import _get_sizeof_string
from speciesOverlap.utilities import update_progress

def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]    

def _get_dot_for_indices(indices,OvCalc):

    start = time()
    # get submatrix for current pond indices
    curr_matrix = OvCalc.new_pond_species_matrix[indices,:]
    curr_row,curr_col = curr_matrix.nonzero()
    curr_data = curr_matrix.data

    # shift obtained indices to actually wanted pond indices and construct shaped matrix
    curr_row = np.array([ indices[r] for r in curr_row],dtype=np.int32)
    curr_matrix = sprs.csr_matrix((curr_data,(curr_row,curr_col)), shape=OvCalc.shape)

    # calculate dot product as demanded
    dot_result = curr_matrix.dot(OvCalc.existence_matrix.T)

    del curr_matrix

    gc.collect()

    #print indices[0],"-",indices[-1],":  ",sys.getsizeof(dot_result)/1e6,"MB"
    end = time()
    print indices[0],"-",indices[-1],", time needed:", end-start, "s"

    return dot_result

class OverlapCalculator():

    def __init__(self,pond_species_matrix,weighted=False,int_to_pond=None,pond_to_int=None,verbose=False,delete_original_matrix=False):

        self.is_twocategory = False
        self.verbose = verbose

        self.int_to_pond = int_to_pond

        self.pond_to_int = pond_to_int

        # get shape and nonzero coordinates
        self.Np, self.Ns = pond_species_matrix.shape
        self.shape = pond_species_matrix.shape
        self.row, self.col = pond_species_matrix.nonzero()
        self.weighted = weighted

        #get cumulated occurences of species per pond
        self.k_pond = np.array(pond_species_matrix.sum(axis=1)).flatten()

        if self.weighted:

            # norm species occurences to species probability per pond
            data = pond_species_matrix.data / self.k_pond[self.row].astype(float)
            self.new_pond_species_matrix = sprs.csr_matrix((data,(self.row,self.col)),shape=(self.Np,self.Ns))

            # get existence matrix (has a one in every nonzero entry)
            data = np.ones_like(self.row)
            self.existence_matrix = sprs.csr_matrix((data,(self.row,self.col)),shape=(self.Np,self.Ns))

            if delete_original_matrix:
                del pond_species_matrix
                gc.collect()

        else:
            self.existence_matrix = pond_species_matrix
            self.new_pond_species_matrix = pond_species_matrix

        if self.verbose:
            matrix_size = self.new_pond_species_matrix.data.nbytes + self.new_pond_species_matrix.indptr.nbytes + self.new_pond_species_matrix.indices.nbytes
            print ("Found %d ponds and %d species. Matrix size in memory is "+_get_sizeof_string(matrix_size)) % (self.Np,self.Ns)

    def get_overlap_matrix_single(self):

        # get dot product of each chunk submatrix with existence matrix
        if self.verbose:
            print "calculating overlap matrix..."
        W = self.new_pond_species_matrix.dot(self.existence_matrix.T)

        del self.new_pond_species_matrix
        del self.existence_matrix
        gc.collect()

        # finalize calculation
        if self.weighted:
            if self.verbose:
                print "calculating weighted overlap..." 
            self.overlap_matrix = W.multiply( W.T ) 
            del W
            gc.collect()
        else:
            self.overlap_matrix = W

        return self.overlap_matrix

    def get_overlap_matrix(self,nprocs=1,chunk_size=10000):

        # This is a bad workaround to get subprocesses that actually work on several CPUs
        # see last post in https://github.com/ipython/ipython/issues/840
        # or http://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy/15641148#15641148
        os.system("taskset -p 0xff %d" % os.getpid())

        # get chunks of pond indices
        indices = [ np.array(chunk, dtype=np.int32) for chunk in _chunks(np.arange(self.Np,dtype=np.int32),chunk_size) ]

        # initialize worker pool
        pool = Pool(nprocs)

        # make function suitable for a constant argument (self)
        partial_get_dot_for_indices = partial(_get_dot_for_indices, OvCalc=self)

        # get dot product of each chunk submatrix with existence matrix
        dot_result = pool.map(partial_get_dot_for_indices,indices)

        # safely close pool
        pool.close()
        pool.join()

        # collect results as sum over all indices chunks
        W = sum(dot_result)

        del dot_result
        gc.collect()

        # finalize calculation
        if self.weighted:
            self.overlap_matrix = W.multiply( W.T ) 
            del W
            gc.collect()
        else:
            self.overlap_matrix = W


        return self.overlap_matrix

    def save_sparse(self,filename):

        row, col = self.overlap_matrix.nonzero()
        data = self.overlap_matrix.data
        np.savez(open(filename,'wb'),row=row,col=col,data=data)

    def save_ovcalc(self,filename):

        row, col = self.overlap_matrix.nonzero()
        data = self.overlap_matrix.data
        pickle.dump({
                        'row': row, 
                        'col': col,
                        'data': data,
                        'pond_to_int': self.pond_to_int,
                        'int_to_pond': self.int_to_pond,
                        'N_ponds': self.Np,
                        'k_pond': self.k_pond,
                    },
                    open(filename,'wb')
                    )

class FinishedOvCalc():

    def __init__(self,
                 filename=None,
                 props=None):

        if filename is not None:
            props = pickle.load(open(filename,'rb'))
            row, col, data = props["row"], props["col"], props["data"]
        elif props is not None:
            self.overlap_matrix = props['overlap_matrix']
        else:
            raise ValueError("Unexpected input arguments filename =",filename,"; props =", props)

        if "N_glades" in props:
            self.is_twocategory = True
            self.glade_to_int = props['glade_to_int']
            self.int_to_glade = props['int_to_glade']
            self.N_glades = props["N_glades"]

        else:
            self.is_twocategory = False

        if "N_ponds" in props:
            self.N_ponds = props["N_ponds"]

        self.pond_to_int = props['pond_to_int']
        self.int_to_pond = props['int_to_pond']
        self.k_pond = props["k_pond"]


        if filename is not None:
            if self.is_twocategory:
                self.overlap_matrix = sprs.csr_matrix((data,(row,col)),shape=(self.N_ponds,self.N_glades))
            else:
                self.overlap_matrix = sprs.csr_matrix((data,(row,col)),shape=(self.N_ponds,self.N_ponds))

    

    def save_ovcalc(self,filename):

        row, col = self.overlap_matrix.nonzero()
        data = self.overlap_matrix.data

        values = {
                    'row': row, 
                    'col': col,
                    'data': data,
                    'pond_to_int': self.pond_to_int,
                    'int_to_pond': self.int_to_pond,
                    'N_ponds': self.N_ponds,
                    'k_pond': self.k_pond,
                 },
        
        if self.is_twocategory:
            values['N_glades'] = self.N_glades
            values['glade_to_int'] = self.glade_to_int
            values['int_to_glade'] = self.int_to_glade

        pickle.dump(values,
                    open(filename,'wb')
                    )

class ColumnListOverlapCalculator(OverlapCalculator):

    def __init__(self,column_list,verbose=False,delete_original_data=True):

        """ requires data_list to be a list like 
            [ (pond_identifier, species_identifier, weight), (..., ), ... ]
        """


        if len(column_list) == 2:
            weighted = False
        elif len(column_list) == 3:
            weighted = True
        else:
            raise ValueError("Unexpected size of column_list:", len(column_list),'. Expected 2 or 3.')

        if len(column_list[0]) != len(column_list[1]) or \
            (weighted and ( len(column_list[2]) != len(column_list[1]) or\
                            len(column_list[0]) != len(column_list[2])\
                          )\
                          ):
            raise ValueError("Columns don't have same length")

        pond_set = set(list(column_list[0]))
        species_set = set(list(column_list[1]))

        self.pond_to_int = {}
        self.species_to_int = {}
        self.int_to_pond = {}
        self.int_to_species = {}

        pond_counter = 0
        for pond in pond_set:
            self.pond_to_int[pond] = pond_counter
            self.int_to_pond[pond_counter] = pond
            pond_counter += 1

        species_counter = 0
        for species in species_set:
            self.species_to_int[species] = species_counter
            self.int_to_species[species_counter] = species
            species_counter += 1

        row = np.array([ self.pond_to_int[pond] for pond in column_list[0] ], dtype=np.int32)
        col = np.array([ self.species_to_int[species] for species in column_list[1] ], dtype=np.int32)

        if weighted:
            data = np.array([ weight for weight in column_list[2] ],dtype=np.float32)
        else:
            data = np.ones_like(row)

        if delete_original_data:
            del column_list
            gc.collect()

        pond_species_matrix = sprs.csr_matrix((data,(row,col)),shape=(pond_counter,species_counter))


        OverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=self.int_to_pond,
                                   pond_to_int=self.pond_to_int,
                                   verbose = verbose,
                                   delete_original_matrix = True,
                                   )



class TupleListOverlapCalculator(OverlapCalculator):

    def __init__(self,data_list,verbose=False,delete_original_data=False):
        """ requires data_list to be a list like 
            [ (pond_identifier, species_identifier, weight), (..., ), ... ]
        """

        if len(data_list[0]) == 2:
            weighted = False
        elif len(data_list[0]) == 3:
            weighted = True
        else:
            raise ValueError("Unexpected value size of data_list:", len(data_list[0]),'. Expected 2 or 3.')

        row = []
        col = []
        data = []

        pond_counter = 0
        species_counter = 0

        self.pond_to_int = {}
        self.species_to_int = {}
        self.int_to_pond = {}
        self.int_to_species = {}

        len_all = len(data_list)

        if verbose and len_all>20000:
            times = []
            entry_count = 1
            chunk_size = 5000
            n_chunks = int(np.ceil(len_all/float(chunk_size)))
            start = time()

        for entry in data_list:
            pond = entry[0]
            species = entry[1]

            if weighted:
                dat = entry[2]
            else:
                dat = 1

            if pond not in self.pond_to_int:
                self.pond_to_int[pond] = pond_counter
                self.int_to_pond[pond_counter] = pond
                current_pond_int = pond_counter
                pond_counter += 1
            else:
                current_pond_int = self.pond_to_int[pond]

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

        if delete_original_data:
            del data_list
            gc.collect()

        pond_species_matrix = sprs.csr_matrix((data,(row,col)),shape=(pond_counter,species_counter))

        OverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=self.int_to_pond,
                                   pond_to_int=self.pond_to_int,
                                   verbose = verbose,
                                   delete_original_matrix = True
                                   )




if __name__=="__main__":


    from simind import abundance_based_similarity as AbSim
    from simind import incidence_based_similarity as InSim
    
    A = np.array( [ 
                    [ 2, 3, 0, 0 ],
                    [ 0, 1, 2, 3 ],
                    [ 0, 0, 2, 3 ],
                    #[ 3, 2, 0, 0 ],
                ])

    print "RESULT"
    #print get_weighted_overlap_matrix(sprs.csr_matrix(A))

    OvCalc = OverlapCalculator(sprs.csr_matrix(A),weighted=True)
    print OvCalc.get_overlap_matrix(nprocs=2,chunk_size=2)

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



    print " SINGLE OCCURENCE "
    row,col = A.nonzero()
    data = np.ones_like(row)
    A2 = sprs.csr_matrix((data,(row,col)),shape=A.shape)

    OvCalc = OverlapCalculator(sprs.csr_matrix(A2),weighted=False)
    print OvCalc.get_overlap_matrix(nprocs=2,chunk_size=2)


    print "TEST"

    A2 = A2.toarray()

    pond_0 = A2[0,:].nonzero()[0]
    pond_1 = A2[1,:].nonzero()[0]
    pond_2 = A2[2,:].nonzero()[0]

    print (0,0), InSim(pond_0, pond_0).S_AB
    print (0,1), InSim(pond_0, pond_1).S_AB
    print (0,2), InSim(pond_0, pond_2).S_AB
    print (1,0), InSim(pond_1, pond_0).S_AB
    print (1,1), InSim(pond_1, pond_1).S_AB
    print (1,2), InSim(pond_1, pond_2).S_AB


    print "TUPLE LIST TEST"
    data = [ 
            ( 'a', '0', 2 ),
            ( 'a', '1', 3 ),
            ( 'b', '1', 1 ),
            ( 'b', '2', 2 ),
            ( 'b', '3', 3 ),
            ( 'c', '2', 2 ),
            ( 'c', '3', 3 ),
           ]
    OvCalc = TupleListOverlapCalculator(data,verbose=True)
    OvCalc.get_overlap_matrix(2,2)

    print OvCalc.overlap_matrix

    print "COLUMN LIST TEST"
    data2 = []
    data2.append([ a[0] for a in data ]) 
    data2.append([ a[1] for a in data ]) 
    data2.append([ a[2] for a in data ]) 
    OvCalc = ColumnListOverlapCalculator(data2)
    OvCalc.get_overlap_matrix(2,2)

    print OvCalc.overlap_matrix

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

class TwoCategoryOverlapCalculator(OverlapCalculator):

    def __init__(self,pond_species_matrix,glade_species_matrix,weighted=False,int_to_pond=None,int_to_glade=None,int_to_species=None,pond_to_int=None,glade_to_int=None,species_to_int=None,verbose=False,delete_original_matrix=False):

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

        self.int_to_pond = int_to_pond
        self.int_to_species = int_to_species

        self.pond_to_int = pond_to_int
        self.species_to_int = species_to_int

        self.glade_to_int = glade_to_int
        self.int_to_glade = int_to_glade


    def get_overlap_matrix_single(self):

        # get dot product of each chunk submatrix with existence matrix
        W = self.new_pond_species_matrix[:self.N_ponds,:].dot(self.existence_matrix[self.N_ponds:,:].T)

        # finalize calculation
        if self.weighted:
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
                    },
                    open(filename,'wb')
                    )



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


        for i_dl,data_list in enumerate(data_lists): 
            row = []
            col = []
            data = []

            pond_counter = 0

            pond_to_int = {}
            int_to_pond = {}

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
                                   int_to_species=self.int_to_species,
                                   pond_to_int=ponds_to_ints[0],
                                   glade_to_int=ponds_to_ints[1],
                                   species_to_int=self.species_to_int,
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


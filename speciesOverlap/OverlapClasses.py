import gc

from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy as sp
import scipy.sparse as sprs

import cPickle as pickle

def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]    

def _get_dot_for_indices(indices,OvCalc):

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

    print indices[0],"-",indices[-1],":  ",sys.getsizeof(dot_result)/1e6,"MB"

    return dot_result

class OverlapCalculator():

    def __init__(self,pond_species_matrix,weighted=False,int_to_pond=None,int_to_species=None,pond_to_int=None,species_to_int=None):

        self.int_to_pond = int_to_pond
        self.int_to_species = int_to_species

        self.pond_to_int = pond_to_int
        self.species_to_int = species_to_int

        # get shape and nonzero coordinates
        self.Np, self.Ns = pond_species_matrix.shape
        self.shape = pond_species_matrix.shape
        self.row, self.col = pond_species_matrix.nonzero()
        self.weighted = weighted

        if self.weighted:

            #get cumulated occurences of species per pond
            self.k_pond = np.array(pond_species_matrix.sum(axis=1)).flatten()

            # norm species occurences to species probability per pond
            data = pond_species_matrix.data / self.k_pond[self.row].astype(float)
            self.new_pond_species_matrix = sprs.csr_matrix((data,(self.row,self.col)),shape=(self.Np,self.Ns))

            # get existence matrix (has a one in every nonzero entry)
            data = np.ones_like(self.row)
            self.existence_matrix = sprs.csr_matrix((data,(self.row,self.col)),shape=(self.Np,self.Ns))

        else:
            self.existence_matrix = pond_species_matrix
            self.new_pond_species_matrix = pond_species_matrix

    def get_overlap_matrix(self,nprocs=1,chunk_size=10000):

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

        # finalize calculation
        if self.weighted:
            self.overlap_matrix = W.multiply( W.T ) 
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
                        'species_to_int': self.species_to_int,
                        'int_to_pond': self.int_to_pond,
                        'int_to_species': self.int_to_species,
                        'N_pond': self.Np,
                        'N_species': self.Ns
                    },
                    open(filename,'wb')
                    )

    def load_ovcalc(self,filename):

        props = pickle.load(open(filename,'rb'))
        self.Ns, self.Np = props["N_species"], props["N_pond"]
        row, col, data = props["row"], props["col"], props["data"]
        self.overlap_matrix = sprs.csr_matrix((data,(row,col)),shape=(self.Np,self.Ns))

        self.pond_to_int = props['pond_to_int']
        self.species_to_int = props['species_to_int']
        self.int_to_pond = props['int_to_pond']
        self.int_to_species = props['int_to_species']

class ColumnListOverlapCalculator(OverlapCalculator):

    def __init__(self,column_list):

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

        pond_species_matrix = sprs.csr_matrix((data,(row,col)),shape=(pond_counter,species_counter))


        OverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=self.int_to_pond,
                                   int_to_species=self.int_to_species,
                                   pond_to_int=self.pond_to_int,
                                   species_to_int=self.species_to_int,
                                   )



class TupleListOverlapCalculator(OverlapCalculator):

    def __init__(self,data_list):
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

        row = np.array(row,dtype=np.int32)
        col = np.array(col,dtype=np.int32)
        data = np.array(data,dtype=np.float32)

        # sort_ndcs = np.argsort(row)

        # row = row[sort_ndcs]
        # col = col[sort_ndcs]
        # data = data[sort_ndcs]

        pond_species_matrix = sprs.csr_matrix((data,(row,col)),shape=(pond_counter,species_counter))

        OverlapCalculator.__init__(self,
                                   pond_species_matrix,
                                   weighted=weighted,
                                   int_to_pond=self.int_to_pond,
                                   int_to_species=self.int_to_species,
                                   pond_to_int=self.pond_to_int,
                                   species_to_int=self.species_to_int,
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
    OvCalc = TupleListOverlapCalculator(data)
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

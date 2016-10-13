import numpy as np
import scipy as sp
import scipy.sparse as sprs
import simind

def get_weighted_overlap_matrix(pond_species_matrix):

    return get_overlap_matrix(pond_species_matrix,True)

def get_overlap_matrix(pond_species_matrix,weighted=False):

    # get shape and nonzero coordinates
    Np, Ns = pond_species_matrix.shape
    row, col = pond_species_matrix.nonzero()

    if weighted:

        #get cumulated occurences of species per pond
        k_pond = np.array(pond_species_matrix.sum(axis=1)).flatten()

        # norm species occurences to species probability per pond
        data = pond_species_matrix.data / k_pond[row].astype(float)
        new_pond_species_matrix = sprs.csr_matrix((data,(row,col)),shape=(Np,Ns))

        # get existence matrix (has a one in every nonzero entry)
        data = np.ones_like(row)
        existence_matrix = sprs.csr_matrix((data,(row,col)),shape=(Np,Ns)) 

    else:
        existence_matrix = pond_species_matrix
        new_pond_species_matrix = pond_species_matrix

    # get overlap matrix (cumulated probabilities of species occuring in pond a that 
    # also occur in pond b
    W = new_pond_species_matrix.dot( existence_matrix.T )

    if weighted:
        # multiply probability of either overlap species
        result = W.multiply( W.T )
    else:
        result = W

    return result




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
    print get_weighted_overlap_matrix(sprs.csr_matrix(A))

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

    print get_overlap_matrix(A2)

    print "TEST"

    A2 = A2.toarray()

    pond_0 = A2[0,:].nonzero()[0]
    pond_1 = A2[1,:].nonzero()[0]
    pond_2 = A2[2,:].nonzero()[0]
    print pond_0

    print (0,0), InSim(pond_0, pond_0).S_AB
    print (0,1), InSim(pond_0, pond_1).S_AB
    print (0,2), InSim(pond_0, pond_2).S_AB
    print (1,0), InSim(pond_1, pond_0).S_AB
    print (1,1), InSim(pond_1, pond_1).S_AB
    print (1,2), InSim(pond_1, pond_2).S_AB


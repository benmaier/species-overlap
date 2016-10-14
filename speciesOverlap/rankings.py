import gc

from time import time

import numpy as np
import scipy.sparse as sprs
from scipy.stats.mstats import zscore

from itertools import izip

import bottleneck as bn

from speciesOverlap import FinishedOvCalc
from speciesOverlap import TupleListOverlapCalculator as TLOvCalc
from speciesOverlap import TupleListTwoCategoryOverlapCalculator as TLTCOvCalc
from speciesOverlap.utilities import update_progress

class Ranking():
    
    def __init__(self,ovcalc,function_to_apply_to_matrix=None,znorm_target=False,standard_mean=None,ranklength=None,verbose=False):

        if isinstance(ovcalc,basestring):
            self.ovcalc = FinishedOvCalc(filename)
        else:
            self.ovcalc = ovcalc

        self.prefunc = function_to_apply_to_matrix
        self.znorm_target = znorm_target
        self.standard_mean = standard_mean

        self.ranklength = ranklength

        self.N_source, self.N_target = self.ovcalc.overlap_matrix.shape

        self.verbose = verbose

    def compute_single(self,use_transposed=False, is_single_category=True,save_scores=False,ranks=[]):


        if use_transposed:

            # this is tricky. Transposing does not change the indices-, indptr-arrays
            # so we have to convert the transposed matrix to csr in order to 
            # make sure that sorting is done in rows.
            # however, for znorming we need it in csc.
            # so, we keep it csc if znormed is required
            if self.znorm_target:
                new_matrix = self.ovcalc.overlap_matrix.transpose(copy=True)
            else:
                new_matrix = self.ovcalc.overlap_matrix.T.tocsr()

            # swap meaning of labels in transpose mode
            if not is_single_category:
                int_to_pond = self.ovcalc.int_to_glade
                int_to_glade = self.ovcalc.int_to_pond
        else:
            # here, no copy is needed, since znorm will make a copy.
            # if theres no data manipulation then no copy is needed.
            new_matrix = self.ovcalc.overlap_matrix
            if not is_single_category:
                int_to_pond = self.ovcalc.int_to_pond
                int_to_glade = self.ovcalc.int_to_glade

        if is_single_category:
            int_to_pond = self.ovcalc.int_to_pond
            int_to_glade = self.ovcalc.int_to_pond


        if self.prefunc is not None:
            new_matrix = self.prefunc(new_matrix)

        if self.znorm_target:

            # we only have to convert it to csc if it's not transposed
            if not use_transposed:
                new_matrix = new_matrix.tocsc()

            if self.standard_mean is not None:
                manipulate = lambda x: ( zscore(x) + self.standard_mean ) / self.standard_mean
            else:
                manipulate = lambda x: zscore(x)

            times = []
            start = time()
            for col in xrange(new_matrix.shape[1]):

                #print int_to_glade[col], new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]]
                new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] = manipulate( new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] )

                if self.verbose:
                    end = time()
                    times.append(end-start)
                    update_progress(col+1,new_matrix.shape[1],times,status="calculating zscore")
                    start = time()

            new_matrix = new_matrix.tocsr()            

        new_matrix = new_matrix * (-1.)

        times = []
        start = time()

        for row in xrange(new_matrix.shape[0]):

            data_length = len( new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ])

            if self.ranklength is None or data_length <= self.ranklength:
                greatest_indices = np.argsort( new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ])
            else:
                # get n smallest values (or greatest, respectively, since we multiplied with -1)
                # note that bottleneck does not return them in order, just the n smallest
                greatest_indices = bn.argpartsort( new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ], self.ranklength)[:self.ranklength]
                # sort n smallest values to have them in order
                greatest_indices = greatest_indices[ np.argsort( (new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ])[greatest_indices] ) ]


            col_indices = (new_matrix.indices[new_matrix.indptr[row]:new_matrix.indptr[row+1]])[greatest_indices]

            if save_scores:
                vals = - (new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ])[greatest_indices]
                for col,val in izip(col_indices,vals):
                    ranks.append([ int_to_pond[row], int_to_glade[col], val ])
            else:
                for col in col_indices:
                    ranks.append([ int_to_pond[row], int_to_glade[col] ])

            if self.verbose:
                end = time()
                times.append(end-start)
                update_progress(row+1,new_matrix.shape[0],times,status="ranking and saving")
                start = time()


        return ranks

    def compute(self,save_scores=False,rank_object=None):

        if self.ovcalc.is_twocategory:
            if rank_object is None:
                rank_object = [[],[]]

            self.compute_single(is_single_category=False,save_scores=save_scores,ranks=rank_object[0])
            self.compute_single(use_transposed=True,is_single_category=False,save_scores=save_scores,ranks=rank_object[1])

            ranks = rank_object

        else:
            if rank_object is None:
                rank_object = []
            ranks = self.compute_single(save_scores=save_scores,ranks=rank_object)

        return ranks


class NormedOverlapRanking(Ranking):

    def __init__(self,ovcalc,standard_mean=2.,ranklength=None,verbose=False):

        Ranking.__init__(self,ovcalc,znorm_target=True,standard_mean=standard_mean,ranklength=ranklength,verbose=verbose)


if __name__=="__main__":

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
            ( 'd', '2', 2 ),
            ( 'd', '3', 3 ),
            ( 'd', '4', 3 ),
           ]

    ovcalc1 = TLOvCalc(data,verbose=True)
    ovcalc1.get_overlap_matrix_single()

    ranking = NormedOverlapRanking(ovcalc1)
    print ranking.compute()

    ovcalc2 = TLTCOvCalc(data[:5],data[5:],verbose=True)
    ovcalc2.get_overlap_matrix_single()

    ranking = NormedOverlapRanking(ovcalc2,verbose=True)
    print ranking.compute()

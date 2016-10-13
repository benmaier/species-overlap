import gc

import numpy as np
import scipy.sparse as sprs
from scipy.stats.mstats import zscore

import bottleneck as bn

from speciesOverlap import FinishedOvCalc
from speciesOverlap import TupleListOverlapCalculator as TLOvCalc
from speciesOverlap import TupleListTwoCategoryOverlapCalculator as TLTCOvCalc

class Ranking():
    
    def __init__(self,ovcalc,function_to_apply_to_matrix=None,znorm_target=False,standard_mean=None,ranklength=None):

        if isinstance(ovcalc,basestring):
            self.ovcalc = FinishedOvCalc(filename)
        else:
            self.ovcalc = ovcalc

        self.prefunc = function_to_apply_to_matrix
        self.znorm_target = znorm_target
        self.standard_mean = standard_mean

        if self.ranklength is None:
            self.ranklength = max(self.ovcalc.overlap_matrix.shape)
        else:
            self.ranklength = ranklength

        self.N_source, self.N_target = self.ovcalc.overlap_matrix.shape

    def compute_single(self,use_transposed=False, is_single_category=True):

        if use_transposed:
            new_matrix = self.ovcalc.overlap_matrix.T
            # swap meaning of labels in transpose mode
            if not is_single_category:
                int_to_pond = self.ovcalc.int_to_glade
                int_to_glade = self.ovcalc.int_to_pond
        else:
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
            new_matrix = new_matrix.tocsc()

            if self.standard_mean is not None:
                for col in xrange(len(new_matrix.indptr)-1):
                    new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] = \
                            ( -zscore( new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] ) + self.standard_mean ) / self.standard_mean
            else:
                for col in xrange(len(new_matrix.indptr)-1):
                    new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] = -zscore( new_matrix.data[new_matrix.indptr[col]:new_matrix.indptr[col+1]] )

            new_matrix = new_matrix.tocsr()

        ranks = []

        for row in xrange(new_matrix.shape[0]):
            greatest_indices = bn.argpartsort( new_matrix.data[ new_matrix.indptr[row]:new_matrix.indptr[row+1] ], self.ranklength)
            col_indices = (new_matrix.indices[new_matrix.indptr[row]:new_matrix.indptr[row+1]])[greatest_indices]
            for col in col_indices:
                ranks.append([ int_to_pond[row], int_to_glade[col] ])

        return ranks

    def compute(self):

        if hasattr(ovCalc,'N_glades'):

            ranks = self.compute_single(is_single_category=False)
            ranks += self.compute_single(use_transposed=True,is_single_category=False)

        else:
            ranks = self.compute_single()

        return ranks


class NormedOverlapRanking(Ranking):

    def __init__(self,ovcalc,standard_mean=2.,ranklength=None):

        Ranking.__init__(self,ovcalc,znorm_target=True,standard_mean=standard_mean,ranklength=ranklength)


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

    Ranking = NormedOverlapRanking(ovcalc1)
    print Ranking.compute()

    ovcalc2 = TLTCOvCalc(data[:5],data[5:],verbose=True)
    ovcalc2.get_overlap_matrix_single()

    Ranking = NormedOverlapRanking(ovcalc2)
    print Ranking.compute()

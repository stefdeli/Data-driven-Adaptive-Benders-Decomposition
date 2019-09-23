# -*- coding: utf-8 -*-
"""
For GNU GLP v3:

{{Data-driven Adaptive Benders Decomposition for the Stochastic Unit Commitment Problem}}
Copyright (C) {{ 2019 }}  {{ ETH Zurich }}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

import random



def kMedoids(D, k, tmax=100):

    # determine dimensions of distance matrix D

    m, n = D.shape



    if k > n:

        raise Exception('too many medoids')



    # find a set of valid initial cluster medoid indices since we

    # can't seed different clusters with two points at the same location

    valid_medoid_inds = set(range(n))

    invalid_medoid_inds = set([])

    rs,cs = np.where(D==0)

    # the rows, cols must be shuffled because we will keep the first duplicate below

    index_shuf = list(range(len(rs)))

    np.random.shuffle(index_shuf)

    rs = rs[index_shuf]

    cs = cs[index_shuf]

    for r,c in zip(rs,cs):

        # if there are two points with a distance of 0...

        # keep the first one for cluster init

        if r < c and r not in invalid_medoid_inds:

            invalid_medoid_inds.add(c)

    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)



    if k > len(valid_medoid_inds):

        raise Exception('too many medoids (after removing {} duplicate points)'.format(

            len(invalid_medoid_inds)))



    # randomly initialize an array of k medoid indices

    M = np.array(valid_medoid_inds)

    np.random.shuffle(M)

    M = np.sort(M[:k])



    # create a copy of the array of medoid indices

    Mnew = np.copy(M)



    # initialize a dictionary to represent clusters

    C = {}

    for t in range(tmax):

        # determine clusters, i. e. arrays of data indices

        J = np.argmin(D[:,M], axis=1)

        for kappa in range(k):

            C[kappa] = np.where(J==kappa)[0]

        # update cluster medoids

        for kappa in range(k):

            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)

            j = np.argmin(J)

            Mnew[kappa] = C[kappa][j]

        np.sort(Mnew)

        # check for convergence

        if np.array_equal(M, Mnew):

            break

        M = np.copy(Mnew)

    else:

        # final update of cluster memberships

        J = np.argmin(D[:,M], axis=1)

        for kappa in range(k):

            C[kappa] = np.where(J==kappa)[0]



    # return results

    return M, C

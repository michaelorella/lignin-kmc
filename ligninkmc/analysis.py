# !/usr/bin/env python
# coding=utf-8

"""
The above code is the traditional use case for running the kmc code, assuming that there are rates for the individual
events. Below is the use case for analyzing the results obtained from a single simulation of lignification.

adj = res['adjacency_matrix']
mons = res['monomers']
t = res['time']
analysis = kmc.analyze( adjacency = adj , nodes = mons )

{'Chain Lengths': ______ ,'Bonds': _______ ,'RCF Yields': ________ ,'RCF Bonds': _______}

"""

import scipy.sparse as sp
import numpy as np
from collections import Counter
from ligninkmc.kmc_common import (AO4, B1, B1_ALT, B5, BB, BO4, C5C5, C5O4)


################################################################################
# ANALYSIS CODE
################################################################################


def find_fragments(adj=None):
    """
    Implementation of a modified depth first search on the adjacency matrix provided to identify isolated graphs within
    the superstructure. This allows us to easily track the number of isolated fragments and the size of each of these
    fragments. This implementation does not care about the specific values within the adjacency matrix, but effectively
    treats the adjacency matrix as boolean.

    a = sp.dok_matrix((2,2))
    findFragments(a)

    {{0},{1}}

    a.resize((5,5))
    a[0,1] = 1; a[1,0] = 1; a[0,2] = 1; a[2,0] = 1; a[3,4] = 1; a[4,3] = 1
    findFragments(a)

    {{0,1,2},{3,4}}

    a = sp.dok_matrix((5,5))
    a[0,4] = 1; a[4,0] = 1
    findFragments(a)

    {{0,4},{1},{2},{3}}

    :param adj: DOK_MATRIX  -- NxN sparse matrix in dictionary of keys format that contains all of the connectivity
        information for the current lignification state
    :return: A set of sets of the unique integer identifiers for the monomers contained within each fragment.
    """
    remaining_nodes = list(range(adj.get_shape()[0]))
    current_node = 0
    connected_fragments = [set()]
    connection_stack = []

    csr_adj = adj.tocsr(copy=True)

    while current_node is not None:
        # Indicate that we are currently visiting this node by removing it
        remaining_nodes.remove(current_node)

        # Add to the current_fragment
        current_fragment = connected_fragments[-1]

        # Look for what's connected to this row
        connections = {node for node in csr_adj[current_node].indices}

        # Add these connections to our current fragment
        current_fragment.update({current_node})

        # Visit any nodes that the current node is connected to that still need to be visited
        connection_stack.extend(
            [node for node in connections if (node in remaining_nodes and node not in connection_stack)])

        # Get the next node that should be visited
        if len(connection_stack) != 0:
            current_node = connection_stack.pop()
        elif len(remaining_nodes) != 0:
            current_node = remaining_nodes[0]
            connected_fragments.append(set())
        else:
            current_node = None
    return connected_fragments


def fragment_size(frags=None):
    """
    A rigorous way to analyze the size of fragments that have been identified using the findFragments(adj) tool. Makes
    a dictionary of monomer identities mapped to the length of the fragment that contains them.

    Inputs:
        frags   -- set of sets -- The set of monomer identifier sets that were output from the findFragments code, or
        the sets of monomers that are connected to each other

    Outputs:
        Dictionary mapping the identity of each monomer [0,N-1] to the length of the fragment that it is found in


    frags = {{0},{1}}
    fragmentSize(frags)

    { 0:1 , 1:1 }

    frags = {{0,4,2},{1,3}}
    fragmentSize(frags)

    { 0:3 , 1:2 , 2:3 , 3:2 , 4:3 }

    frags = {{0,1,2,3,4}}
    fragmentSize(frags)

    { 0:5 , 1:5 , 2:5 , 3:5 , 4:5 }
    """
    sizes = {}
    for fragment in frags:
        length = len(fragment)
        for node in fragment:
            sizes[node] = length


def break_bond(adj=None, bond_type=None):
    """
    Function for removing all of a certain type of bond from the adjacency matrix. This is primarily used for the
    analysis at the end of the simulations when in silico RCF should occur. The update happens via conditional removal
    of the matching values in the adjacency matrix.


    a = sp.dok_matrix((5,5))
    a[1,0] = 4; a[0,1] = 8; a[2,3] = 8; a[3,2] = 8;
    breakBond(a,BO4).to_dense()

    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,8,0],
     [0,0,8,0,0],
     [0,0,0,0,0]]

   a = sp.dok_matrix( [[0,4,0,0,0],
                      [8,0,1,0,0],
                      [0,8,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
   breakBond(a, B1_ALT)

    [[0,0,0,0,0],
     [0,0,1,0,0],
     [0,8,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]]

    :param adj: dock_matrix, the adjacency matrix for the lignin polymer that has been simulated, and needs
        certain bonds removed
    :param bond_type: str, the string containing the bond type that should be broken. These are the standard
        nomenclature, except for b1alt, which removes the previous bond between the beta position and another monomer
        on the monomer that is bound through 1
    :return:
    """
    new_adj = adj.todok(1)  # Copy the matrix into a new matrix

    breakage = {B1: (lambda break_row, col: (adj[(break_row, col)] == 1 and adj[(col, break_row)] == 8) or
                                            (adj[(break_row, col)] == 8 and adj[(col, break_row)] == 1)),
                B1_ALT: (lambda break_row, col: (adj[(row, col)] == 1 and adj[(col, row)] == 8) or
                                          (adj[(row, col)] == 8 and adj[(col, row)] == 1)),
                B5: (lambda row, col: (adj[(row, col)] == 5 and adj[(col, row)] == 8) or (adj[(row, col)] == 8 and
                                                                                          adj[(col, row)] == 5)),
                BO4: (lambda row, col: (adj[(row, col)] == 4 and adj[(col, row)] == 8) or (adj[(row, col)] == 8 and
                                                                                           adj[(col, row)] == 4)),
                AO4: (lambda row, col: (adj[(row, col)] == 4 and adj[(col, row)] == 7) or (adj[(row, col)] == 7 and
                                                                                           adj[(col, row)] == 4)),
                C5O4: (lambda row, col: (adj[(row, col)] == 4 and adj[(col, row)] == 5) or (adj[(row, col)] == 5 and
                                                                                            adj[(col, row)] == 4)),
                BB: (lambda row, col: (adj[(row, col)] == 8 and adj[(col, row)] == 8)),
                C5C5: (lambda row, col: (adj[(row, col)] == 5 and adj[(col, row)] == 5))}

    for el in adj.keys():
        row = el[0]
        col = el[1]

        if breakage[bond_type](row, col) and bond_type != B1_ALT:
            new_adj[(row, col)] = 0
            new_adj[(col, row)] = 0
        elif breakage[bond_type](row, col):
            if adj[(row, col)] == 1:  # The other 8 is in this row
                break_bond_in_matrix(adj, new_adj, row)
            else:  # The other 8 is in the other row
                break_bond_in_matrix(adj, new_adj, col)
    return new_adj


def break_bond_in_matrix(adj, new_adj, row):
    data = adj.tocoo().getrow(row).data
    cols = adj.tocoo().getrow(row).indices
    for i, idx in enumerate(cols):
        if data[i] == 8:
            break
        new_adj[(row, idx)] = 0
        new_adj[(idx, row)] = 0


def count_bonds(adj=None):
    """
    Counter for the different bonds that are present in the adjacency matrix. Primarily used for getting easy analysis
    of the properties of a simulated lignin from the resulting adjacency matrix.

    a = sp.dok_matrix( [[0,8,0,0,0],
                            [4,0,8,0,0],
                            [0,5,0,8,0],
                            [0,0,8,0,4],
                            [0,0,0,8,0]] )
    countBonds(a)

    { BO4:2 , B1 : 0 , BB : 1 , B5 : 1 , C5C5 : 0 , AO4: 0 , C5O4 : 0 }
    :param adj: DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: dictionary mapping bond strings to the frequency of that specific bond
    """
    count = {BO4: 0, B1: 0, BB: 0, B5: 0, C5C5: 0, AO4: 0, C5O4: 0}
    bonds = {(4, 8): BO4, (8, 4): BO4, (8, 1): B1, (1, 8): B1, (8, 8): BB, (5, 5): C5C5, (8, 5): B5,
             (5, 8): B5, (7, 4): AO4,
             (4, 7): AO4, (5, 4): C5O4, (4, 5): C5O4}

    for el in sp.triu(adj).todok().keys():  # Don't double count by looking only at the upper triangular keys
        row = el[0]
        col = el[1]

        bond = (adj[(row, col)], adj[(col, row)])
        count[bonds[bond]] += 1

    return count


def count_yields(adj=None):
    """
    Use the depth first search implemented in findFragments(adj) to locate individual fragments, and then determine the
    sizes of these individual fragments to obtain estimates of simulated oligomeric yields.

    a = sp.dok_matrix( [ [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    countYields(a)

    { 1 : 5 }

    a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    countYields(a)

    { 2 : 1 , 1 : 3 }

    a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,8,0],
                             [0,0,5,0,0],
                             [0,0,0,0,0] ] )
    countYields(a)

    { 2 : 2 , 1 : 1 }

    a = sp.dok_matrix( [ [0,4,8,0,0],
                             [8,0,0,0,0],
                             [5,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    countYields(a)

    { 3 : 1 , 1 : 2 }

    :param adj: DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: A Counter dictionary mapping the length of fragments to the number of occurrences of that length
    """

    # Figure out what is still connected by using the determineCycles function, and look at the length of each
    # connected piece
    oligomers = find_fragments(adj=adj)
    counts = Counter(map(len, oligomers))
    return counts


def analyze(adjacency=None, nodes=None):
    """
    Performs the analysis for a single simulation to extract the relevant macroscopic properties, such as both the
    simulated frequency of different oligomer sizes and the number of each different type of bond before and after in
    silico RCF. The specific code to handle each of these properties is written in the countBonds(adj) and
    countYields(adj) specifically.

     a = sp.dok_matrix( [ [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    analyze(a)

    { 'Chain Lengths' : output from count_yields(a) , 'Bonds' : output from count_bonds(a) , 'RCF Yields' : output from
    count_yields(a') where a' has bonds broken , 'RCF Bonds' : output from count_bonds(a') }

    :param adjacency: DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
    :param nodes: list, list of monomer objects that have identities matching the indices of the adjacency matrix
    :return: A dictionary of keywords to the desired result - e.g. Chain Lengths, RCF Yields, Bonds, and RCF Bonds
    """

    # Remove any excess b1 bonds from the matrix, e.g. bonds that should be
    # broken during synthesis
    adjacency = break_bond(adj=adjacency, bond_type=B1_ALT)

    # Examine the initial polymers before any bonds are broken
    yields = count_yields(adj=adjacency)
    bond_distributions = count_bonds(adj=adjacency)

    # Simulate the RCF process at complete conversion by breaking all of the
    # alkyl C-O bonds that were formed during the reaction
    rcf_adj = break_bond(adj=break_bond(adj=break_bond(adj=adjacency, bond_type=BO4), bond_type=AO4),
                         bond_type=C5O4)

    # Now count the bonds and yields remaining
    rcf_yields = count_yields(adj=rcf_adj)
    rcf_bonds = count_bonds(adj=rcf_adj)

    return {'Chain Lengths': yields, 'Bonds': bond_distributions,
            'RCF Yields': rcf_yields, 'RCF Bonds': rcf_bonds}


def degree(adj):
    """
    Determines the degree for each monomer within the polymer chain. The "degree" concept in graph theory
    is the number of edges connected to a node. In the context of lignin, that is simply the number of
    connected residues to a specific residue, and can be used to determine derived properties like the
    branching coefficient.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated

    Outputs:
        The degree for each monomer as a numpy array.
    """
    return np.bincount(adj.nonzero()[0])


def branching_coefficient(adj):
    """
    Based on the definition in Dellon et al. (10.1021/acs.energyfuels.7b01150), this is the number of
    monomers with degree 3 or more divided by the total number of monomers.

    :param adj: DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: The branching coefficient that corresponds to the adjacency matrix
    """

    degrees = degree(adj)
    return np.sum(degrees >= 3) / len(degrees)

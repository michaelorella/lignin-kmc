# !/usr/bin/env python
# coding=utf-8

"""
Code base for simulating the in planta polymerization of monolignols through Gillepsie algorithm adaptations.
Added the visualization tools here.
"""

import re
import copy
import numpy as np
from collections import (defaultdict, OrderedDict)
from scipy import triu
from scipy.sparse import dok_matrix
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from common_wrangler.common import (InvalidDataError, create_out_fname, warning, round_sig_figs)
from ligninkmc.kmc_common import (Event, Monomer, AO4, B1, B5, BB, BO4, C5C5, C5O4, OX, Q, GROW, TIME, OLIGOMER,
                                  MONOMER, AFFECTED, ADJ_MATRIX, MONO_LIST, MAX_NUM_DECIMAL, ATOMS, BONDS,
                                  G, S, H, C, S4, G4, G7, B1_ALT, CHAIN_LEN, CHAIN_MONOS, CHAIN_BRANCHES,
                                  CHAIN_BRANCH_COEFF, RCF_BONDS, RCF_YIELDS, RCF_MONOS, RCF_BRANCHES, RCF_BRANCH_COEFF,
                                  DEF_TCL_FNAME, DEF_CHAIN_ID, DEF_PSF_FNAME, DEF_TOPPAR, INT_TO_TYPE_DICT)

DrawingOptions.bondLineWidth = 1.2
S7 = 'S7'


def find_fragments(adj):
    """
    Implementation of a modified depth first search on the adjacency matrix provided to identify isolated graphs within
    the superstructure. This allows us to easily track the number of isolated fragments and the size of each of these
    fragments. This implementation does not care about the specific values within the adjacency matrix, but effectively
    treats the adjacency matrix as boolean.

    :param adj: dok_matrix  -- NxN sparse matrix in dictionary of keys format that contains all of the connectivity
        information for the current lignification state
    :return: two lists where the list indices of each correspond to a unique fragment:
                A list of sets: the list contains a set for each fragment, comprised of the unique integer identifiers
                                for the monomers contained within the fragment,
                A list of ints containing the number of number of branch points found in each fragment
    """
    remaining_nodes = list(range(adj.get_shape()[0]))
    current_node = 0
    connected_fragments = [set()]
    connection_stack = []

    branches_in_frags = []
    num_branches = 0

    csr_adj = adj.tocsr(copy=True)

    while current_node is not None:
        # Indicate that we are currently visiting this node by removing it
        remaining_nodes.remove(current_node)

        # Add to the current_fragment
        current_fragment = connected_fragments[-1]

        # Look for what's connected to this row
        connections = {node for node in csr_adj[current_node].indices}
        # if more than two units are connected, there is a branch
        len_connections = len(connections)
        if len_connections > 2:
            num_branches += len_connections - 2

        # Add these connections to our current fragment
        current_fragment.update({current_node})

        # Visit any nodes that the current node is connected to that still need to be visited
        connection_stack.extend([node for node in connections if (node in remaining_nodes and
                                                                  node not in connection_stack)])

        # Get the next node that should be visited
        if len(connection_stack) != 0:
            current_node = connection_stack.pop()
        elif len(remaining_nodes) != 0:
            current_node = remaining_nodes[0]
            # great ready for next fragment
            connected_fragments.append(set())
            branches_in_frags.append(num_branches)
            num_branches = 0
        else:
            current_node = None
            branches_in_frags.append(num_branches)
    return connected_fragments, branches_in_frags


def fragment_size(frags):
    """
    A rigorous way to analyze_adj_matrix the size of fragments that have been identified using the find_fragments(adj)
    tool. Makes a dictionary of monomer identities mapped to the length of the fragment that contains them.

    Example usage:
    > frags = [{0}, {1}]
    > result = fragment_size(frags)
    {0: 1, 1: 1}

    > frags = [{0, 4, 2}, {1, 3}]
    > result = fragment_size(frags)
    {0: 3, 2: 3, 4: 3, 1: 2, 3: 2}

    > frags = [{0, 1, 2, 3, 4}]
    > result = fragment_size(frags)
    {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}

    :param frags: list of sets; the set (list) of monomer identifier sets that were output from
                  find_fragments, or the monomers that are connected to each other
    :return: dict mapping the integer identity of each monomer to the length of the fragment that it is found in
    """
    # Note: this is only used by the test, so maybe delete
    sizes = {}
    for fragment in frags:
        length = len(fragment)
        for node in fragment:
            sizes[node] = length
    return sizes


def quick_frag_size(monomer):
    """
    An easy check on a specific monomer to tell if it is a monomer or involved in an oligomer. This is used over the
    detailed fragment_size(frags) calculation in the simulation of lignification for performance benefits.

    :param monomer: Monomer object that we want to know if it is bound to anything else
                     (i.e. if it is truly a monomer still)
    :return: string, either 'monomer' or 'oligomer' (as the global variable) if it is connected to nothing else,
             or isn't respectively
    """
    if monomer.type == G and monomer.open == {4, 5, 8}:  # Guaiacol monomer
        return MONOMER
    elif monomer.type == S and monomer.open == {4, 8}:  # Syringol monomer
        return MONOMER
    elif monomer.type == C and monomer.open == {4, 5, 8}:  # Caffeoyl monomer
        return MONOMER
    return OLIGOMER


def break_bond_type(adj, bond_type):
    """
    Function for removing all of a certain type of bond from the adjacency matrix. This is primarily used for the
    analysis at the end of the simulations when in silico RCF should occur. The update happens via conditional removal
    of the matching values in the adjacency matrix.

    Example use cases:
    > a = dok_matrix((5,5))
    > a[1,0] = 4; a[0,1] = 8; a[2,3] = 8; a[3,2] = 8;
    > break_bond_type(a, BO4).todense()
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 8, 0],
     [0, 0, 8, 0, 0],
     [0, 0, 0, 0, 0]]

    > a = dok_matrix([[0, 4, 0, 0, 0],
    >                 [8, 0, 1, 0, 0],
    >                 [0, 8, 0, 0, 0],
    >                 [0, 0, 0, 0, 0],
    >                 [0, 0, 0, 0, 0]])
    > break_bond_type(a, B1_ALT).todense()
    [[0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 8, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

    :param adj: dok_matrix, the adjacency matrix for the lignin polymer that has been simulated, and needs
        certain bonds removed
    :param bond_type: str, the string containing the bond type that should be broken. These are the standard
        nomenclature, except for B1_ALT, which removes the previous bond between the beta position and another monomer
        on the monomer that is bound through 1
    :return: dok_matrix, new adjacency matrix after bonds were broken
    """
    # Copy the matrix into a new matrix
    new_adj = adj.todok(copy=True)

    breakage = {B1: (lambda row, col: (adj[(row, col)] == 1 and adj[(col, row)] == 8) or (adj[(row, col)] == 8 and
                                                                                          adj[(col, row)] == 1)),
                B1_ALT: (lambda row, col: (adj[(row, col)] == 1 and adj[(col, row)] == 8) or
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

    for adj_bond_loc in adj.keys():
        adj_row = adj_bond_loc[0]
        adj_col = adj_bond_loc[1]

        if breakage[bond_type](adj_row, adj_col) and bond_type != B1_ALT:
            new_adj[(adj_row, adj_col)] = 0
            new_adj[(adj_col, adj_row)] = 0
        elif breakage[bond_type](adj_row, adj_col):
            if adj[(adj_row, adj_col)] == 1:
                # The other 8 is in this row
                remove_prev_bond(adj, adj_row, new_adj)
            else:
                # The other 8 is in the other row
                remove_prev_bond(adj, adj_col, new_adj)
    return new_adj


def remove_prev_bond(adj, search_loc, new_adj):
    idx = 0  # make IDE happy
    data = adj.tocoo().getrow(search_loc).data
    cols = adj.tocoo().getrow(search_loc).indices
    for i, idx in enumerate(cols):
        if data[i] == 8:
            break
    new_adj[(search_loc, idx)] = 0
    new_adj[(idx, search_loc)] = 0


def count_bonds(adj):
    """
    Counter for the different bonds that are present in the adjacency matrix. Primarily used for getting easy analysis
    of the properties of a simulated lignin from the resulting adjacency matrix.

    :param adj: dok_matrix   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: OrderedDict mapping bond strings to the frequency of that specific bond
    """
    bound_count_dict = OrderedDict({BO4: 0,  BB: 0, B5: 0, B1: 0, C5O4: 0, AO4: 0, C5C5: 0})
    bonding_dict = {(4, 8): BO4, (8, 4): BO4, (8, 1): B1, (1, 8): B1, (8, 8): BB, (5, 5): C5C5,
                    (8, 5): B5, (5, 8): B5, (7, 4): AO4, (4, 7): AO4, (5, 4): C5O4, (4, 5): C5O4}

    adj_array = triu(adj.toarray(), k=1)

    # Don't double count by looking only at the upper triangular keys
    for el in dok_matrix(adj_array).keys():
        row = el[0]
        col = el[1]

        bond = (adj[(row, col)], adj[(col, row)])
        bound_count_dict[bonding_dict[bond]] += 1

    return bound_count_dict


def count_oligomer_yields(adj):
    """
    Use the depth first search implemented in find_fragments(adj) to locate individual fragments and branching
    Related values are also calculated.

    :param adj: scipy dok_matrix, the adjacency matrix for the lignin polymer that has been simulated
    :return: four dicts: an OrderedDict for olig_len_dict (olig_len: num_oligs); the keys are common to all
                             dicts so one ordered dict should be sufficient. The other three dicts are:
                                 olig_length: the total number of monomers involved in oligomers
                                 olig_length: total number of branch points in oligomers of that length
                                 olig_length: the branching coefficient for the oligomers of that length
    """
    oligomers, branches_in_oligs = find_fragments(adj)

    temp_olig_len_dict = defaultdict(int)
    temp_olig_branch_dict = defaultdict(int)
    for oligomer, num_branches in zip(oligomers, branches_in_oligs):
        temp_olig_len_dict[len(oligomer)] += 1
        temp_olig_branch_dict[len(oligomer)] += num_branches

    # create one ordered dict, and three regular dicts
    olig_lengths = list(temp_olig_len_dict.keys())
    olig_lengths.sort()
    olig_len_dict = OrderedDict()
    olig_monos_dict = {}
    olig_branch_dict = {}
    olig_branch_coeff_dict = {}
    for olig_len in olig_lengths:
        num_oligs = temp_olig_len_dict[olig_len]
        num_monos_in_olig_length = num_oligs * olig_len
        num_branches = temp_olig_branch_dict[olig_len]
        olig_len_dict[olig_len] = num_oligs
        olig_monos_dict[olig_len] = num_monos_in_olig_length
        olig_branch_dict[olig_len] = num_branches
        olig_branch_coeff_dict[olig_len] = num_branches / num_monos_in_olig_length

    return olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict


def analyze_adj_matrix(adjacency):
    """
    Performs the analysis for a single simulation to extract the relevant macroscopic properties, such as both the
    simulated frequency of different oligomer sizes and the number of each different type of bond before and after in
    silico RCF. The specific code to handle each of these properties is written in the count_bonds(adj) and
    count_oligomer_yields(adj) specifically.

    :param adjacency: scipy dok_matrix  -- the adjacency matrix for the lignin polymer that has been simulated
    :return: A dictionary of results: Chain Lengths, RCF Yields, Bonds, and RCF Bonds
    """

    # Remove any excess b1 bonds from the matrix, e.g. bonds that should be
    # broken during synthesis
    adjacency = break_bond_type(adjacency, B1_ALT)

    # Examine the initial polymers before any bonds are broken
    olig_yield_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adjacency)
    bond_distributions = count_bonds(adjacency)

    # Simulate the RCF process at complete conversion by breaking all of the
    # alkyl C-O bonds that were formed during the reaction
    rcf_adj = break_bond_type(break_bond_type(break_bond_type(adjacency, BO4), AO4), C5O4)

    # Now count the bonds and yields remaining
    rcf_yield_dict, rcf_monos_dict, rcf_branch_dict, rcf_branch_coeff_dict = count_oligomer_yields(rcf_adj)
    rcf_bonds = count_bonds(rcf_adj)

    return {BONDS: bond_distributions, CHAIN_LEN: olig_yield_dict, CHAIN_MONOS:  olig_monos_dict,
            CHAIN_BRANCHES: olig_branch_dict, CHAIN_BRANCH_COEFF: olig_branch_coeff_dict,
            RCF_BONDS: rcf_bonds, RCF_YIELDS: rcf_yield_dict, RCF_MONOS: rcf_monos_dict,
            RCF_BRANCHES: rcf_branch_dict, RCF_BRANCH_COEFF: rcf_branch_coeff_dict}


def update_events(state_dict, adj, last_event, event_dict, rate_vec, rate_dict, max_mon=500):
    """
    This method determines what the possible events are in a given state, where the state is the current simulation
    state. Most of the additional parameters in this method are added for performance benefits rather than necessity.

    :param state_dict: dict that maps the index of each monomer in the simulation to the monomer itself and the
        event_dict that would be effected by a change to the monomer key. This makes it easy to quickly
        determine which of the event_dict in the simulation need to be updated and which should not be changed.
    :param adj: dok_matrix  -- The current state of the simulation represented by the adjacency matrix containing all
        of the monomers and the bonds between them (if any)
    :param last_event: The previous Event that occurred, which will tell us what monomers were effected. When combined
        with the state dictionary, this allows for efficient updating of the set of event_dict that are possible
    :param event_dict: dict  -- The set of all possible unique event_dict that must be updated and returned from this
        method, implemented in a hash map where the event hash value is the key
    :param rate_vec: dict  -- The rates of all of the unique event_dict implemented in a hash map where the Event
        hash value is the key
    :param rate_dict: dict  -- The dictionary of the possible rates involved in each reaction
    :param max_mon: int -- The maximum number of monomers that should be stored in the simulation
    :return: n/a, updates state_dict, adj, event_dict, rate_vec
    """

    # Map the monomer active state to the possible event_dict it can do
    possible_events = {0: [[OX, 1, rate_dict[OX]]],
                       4: [[B1, 2, rate_dict[B1], [1, 8]],
                           [C5O4, 2, rate_dict[C5O4], [4, 5]],
                           [AO4, 2, rate_dict[AO4], [4, 7]],
                           [BO4, 2, rate_dict[BO4], [4, 8]],
                           [C5C5, 2, rate_dict[C5C5], [5, 5]],
                           [B5, 2, rate_dict[B5], [5, 8]],
                           [BB, 2, rate_dict[BB], [8, 8]]],
                       7: [[Q, 1, rate_dict[Q]],
                           [AO4, 2, rate_dict[AO4], [7, 4]]],
                       -1: [[]]
                       }
    # Only do these for bonding and oxidation event_dict, any growth does not actually change the possible event_dict
    if last_event.key != GROW:
        # Remove the last event that we just did from the set of event_dict that can be performed
        last_event_hash = hash(last_event)
        del (event_dict[last_event_hash])
        del (rate_vec[last_event_hash])

        # Make sure to keep track of which partners have been "cleaned" already - i.e. what monomers have already had
        #     all of the old event_dict removed
        cleaned_partners = set()

        # Get indices of monomers that were acted upon
        affected_monomers = last_event.index

        # Update event_dict from the perspective of each monomer that was just affected individually
        # This prevents having to re-search the entire space every time, saving significant computation
        for mon_id in affected_monomers:
            # Get the affected monomer
            mon = state_dict[mon_id][MONOMER]

            # Get the sets of activated monomers that we could bind with
            ox = set()
            quinone = set()

            other_ids = [x for x in state_dict if x != mon_id]
            for other in other_ids:
                other_mon_type = state_dict[other][MONOMER]
                # Don't allow connections that would cyclize the polymer!
                if other_mon_type.active == 4 and other_mon_type.identity not in mon.connectedTo:
                    ox.add(other_mon_type)
                elif other_mon_type.active == 7 and other_mon_type.identity not in mon.connectedTo:
                    quinone.add(state_dict[other][MONOMER])
            bonding_partners = {BO4: ox, B5: ox, C5O4: ox, C5C5: ox, BB: ox, B1: ox, AO4: quinone}

            # Obtain the event_dict that are affected by a change to the monomer that was just acted on
            events_to_be_modified = state_dict[mon_id][AFFECTED]

            # Get the codes for the event_dict that are possible based on how the current monomer behaves
            active_pos = mon.active
            new_event_list = possible_events[active_pos]

            # Take any event_dict to be modified out of the set so that we can replace with the updated event_dict
            for event in events_to_be_modified:
                ev_hash = hash(event)
                if ev_hash in event_dict:
                    del (event_dict[ev_hash])
                    del (rate_vec[ev_hash])

            # Overwrite the old event_dict that could have been modified from this monomer being updated
            state_dict[mon_id][AFFECTED] = set()
            cur_n, _ = adj.get_shape()

            for rxn_event in new_event_list:
                if rxn_event and rxn_event[1] == 1:  # Unimolecular reaction event
                    size = quick_frag_size(mon)

                    rate = rxn_event[2][mon.type][size] / cur_n

                    # Add the event to the event_dict modifiable by changing the monomer, and update the set of all
                    # event_dict at the next time step
                    state_dict[mon_id][AFFECTED].add(Event(rxn_event[0], [mon.identity], rate))

                elif rxn_event and rxn_event[1] == 2:  # Bimolecular reaction event
                    bond = tuple(rxn_event[3])
                    alt = (bond[1], bond[0])
                    for partner in bonding_partners[rxn_event[0]]:
                        # Sanitize the set of event_dict that can be effected
                        if partner not in cleaned_partners:
                            # Remove any old event_dict from
                            state_dict[partner.identity][AFFECTED].difference_update(events_to_be_modified)
                            cleaned_partners.add(partner)

                        index = [mon.identity, partner.identity]
                        back = [partner.identity, mon.identity]

                        # Add the bond from one active unit to the other in the default config
                        size = (quick_frag_size(mon), quick_frag_size(partner))
                        if bond[0] in mon.open and bond[1] in partner.open:
                            try:
                                # todo: delete when questions re olig-olig b04 bond is resolved
                                if size == (OLIGOMER, OLIGOMER):
                                    if rxn_event[0] == BO4:
                                        print(f"{rxn_event[0]} reaction between oligomers with {mon.identity} and "
                                              f"{partner.identity}")
                                rate = rxn_event[2][(mon.type, partner.type)][size] / (cur_n ** 2)
                            except KeyError:
                                raise InvalidDataError(f"Error while attempting to update event_dict: event "
                                                       f"{rxn_event[0]} between indices {mon.identity} and "
                                                       f"{partner.identity} ")

                            # Add this to both the monomer and it's bonding partners list of event_dict that need to be
                            # modified upon manipulation of either monomer
                            # this -> other
                            state_dict[mon_id][AFFECTED].add(Event(rxn_event[0], index, rate, bond))
                            state_dict[partner.identity][AFFECTED].add(Event(rxn_event[0], index, rate, bond))

                            # Switch the order
                            # other -> this
                            state_dict[mon_id][AFFECTED].add(Event(rxn_event[0], back, rate, alt))
                            state_dict[partner.identity][AFFECTED].add(Event(rxn_event[0], back, rate, alt))

                        # Add the bond from one monomer to the other in the reverse config if not symmetric
                        if rxn_event[0] != BB and rxn_event[0] != C5C5:  # non-symmetric bond
                            if bond[1] in mon.open and bond[0] in partner.open:
                                # Adjust the rate using the correct monomer types
                                try:
                                    rate = rxn_event[2][(partner.type, mon.type)][(size[1], size[0])] / (cur_n ** 2)
                                except KeyError:
                                    raise InvalidDataError(f"Error on determining the rate for rxn_event type "
                                                           f"{rxn_event[0]}, bonding index {mon.identity} to "
                                                           f"{partner.identity}")
                                # this -> other alt
                                state_dict[mon_id][AFFECTED].add(Event(rxn_event[0], index, rate, alt))
                                state_dict[partner.identity][AFFECTED].add(Event(rxn_event[0], index, rate, alt))

                                # Switch the order
                                # other -> this alt
                                state_dict[mon_id][AFFECTED].add(Event(rxn_event[0], back, rate, bond))
                                state_dict[partner.identity][AFFECTED].add(Event(rxn_event[0], back, rate, bond))
                    # END LOOP OVER PARTNERS
                # END UNIMOLECULAR/BIMOLECULAR BRANCH
            # END LOOP OVER NEW REACTION POSSIBILITIES
            for event in state_dict[mon_id][AFFECTED]:
                ev_hash = hash(event)
                event_dict[ev_hash] = event
                rate_vec[ev_hash] = event.rate
        # END LOOP OVER MONOMERS THAT WERE AFFECTED BY LAST EVENT
    else:
        cur_n, _ = adj.get_shape()

        # If the system has grown to the maximum size, delete the event for adding more monomers
        if cur_n >= max_mon:
            last_event_hash = hash(last_event)
            del (event_dict[last_event_hash])
            del (rate_vec[last_event_hash])

        # Reflect the larger system volume
        for i in rate_vec:
            if event_dict[i].key != GROW:
                rate_vec[i] = rate_vec[i] * (cur_n - 1) / cur_n

        # Add an event to oxidize the monomer that was just added to the simulation
        oxidation = Event(OX, [cur_n - 1], rate_dict[OX][state_dict[cur_n - 1][MONOMER].type][MONOMER])
        state_dict[cur_n - 1][AFFECTED].add(oxidation)
        ev_hash = hash(oxidation)
        event_dict[ev_hash] = oxidation
        rate_vec[ev_hash] = oxidation.rate / cur_n


def connect_monos(mon1, mon2):
    if mon1.parent == mon1:
        if mon2.parent == mon2:
            parent = mon1 if mon1.identity <= mon2.identity else mon2
            mon1.parent = parent
            mon2.parent = parent
            parent.size = mon1.size + mon2.size
            return parent
        else:
            parent = connect_monos(mon1, mon2.parent)
            mon1.parent = parent
            mon2.parent = parent
            return parent
    else:
        parent = connect_monos(mon1.parent, mon2)
        mon2.parent = parent
        mon1.parent = parent
        return parent


def do_event(event, state, adj, sg_ratio=None, random_seed=None):
    """
    The second key component of the lignin implementation of the Monte Carlo algorithm, this method actually executes
    the chosen event on the current state and modifies it to reflect the updates.

    :param event: The event object that should be executed on the current state
    :param state: dict, The dictionary of dictionaries that contains the state information for each monomer
    :param adj: dok_matrix, The adjacency matrix in the current state
    :param sg_ratio: float needed if and only if:
                         a) there are S and G and only S and G, and
                         b) new state_dict will be added
    :param random_seed: if a positive integer is provided, it will be used for reproducible results (for testing)
    :return: N/A - mutates the list of state and adjacency matrix instead
    """

    indices = event.index
    # state is indirectly updated by changing monomers
    monomers = [state[i][MONOMER] for i in state]
    if len(indices) == 2:  # Doing bimolecular reaction, need to adjust adj

        # Get the tuple of values corresponding to bond and state updates and
        # unpack them
        vals = event.eventDict[event.key]
        state_updates = vals[0]
        bond_updates = event.bond
        order = event.activeDict[bond_updates]

        # Get the monomers that were being reacted in the correct order
        mon0 = monomers[indices[0]]
        mon1 = monomers[indices[1]]

        connect_monos(mon0, mon1)

        # Make the update to the state and adjacency matrix,
        # Rows are perspective of bonds FROM indices[0] and columns perspective of bonds TO indices[0]
        adj[(indices[0], indices[1])] = bond_updates[0]
        adj[(indices[1], indices[0])] = bond_updates[1]

        # remove the position that was just active
        mon0.open -= {bond_updates[0]}
        mon1.open -= {bond_updates[1]}

        # Update the activated nature of the monomer
        mon0.active = state_updates[order[0]]
        mon1.active = state_updates[order[1]]

        # Add any additional opened positions based on what just reacted
        mon0.open |= set(vals[1 + order[0]])
        mon1.open |= set(vals[1 + order[1]])

        if mon0.active == 7 and mon1.type == C:
            mon0.active = 0
            mon0.open -= {7}

        if mon1.active == 7 and mon0.type == C:
            mon1.active = 0
            mon1.open -= {7}

        # Decided to break bond between alpha and ring position later (i.e. after all synthesis occurred) when a B1
        # bond is formed This is primarily to make it easier to see what the fragment that needs to break is for
        # visualization purposes
        mon0.connectedTo.update(mon1.connectedTo)
        for mon in monomers:
            if mon.identity in mon0.connectedTo:
                mon.connectedTo = mon0.connectedTo

    elif len(indices) == 1:
        if event.key == Q:
            mon = monomers[indices[0]]
            mon.active = 0
            mon.open.remove(7)
            mon.open.add(1)
        elif event.key == OX:
            mon = monomers[indices[0]]

            # Make the monomer appear oxidized
            mon.active = 4
        else:
            raise InvalidDataError(f'Unexpected event: {event.key} for index {indices[0]}')
    else:
        if event.key == GROW:
            current_size, _ = adj.get_shape()

            # Add another monomer to the adjacency matrix
            adj.resize((current_size + 1, current_size + 1))

            # Add another monomer to the state
            if monomers and monomers[-1].type == C:
                mon_type = C
            else:
                try:
                    pct = sg_ratio / (1 + sg_ratio)
                    if random_seed:
                        # to prevent the same choice every iteration, add a changing integer
                        np.random.seed(random_seed + current_size)
                        rand_num = np.around(np.random.rand(), MAX_NUM_DECIMAL)
                    else:
                        rand_num = np.random.rand()
                    mon_type = INT_TO_TYPE_DICT[int(rand_num < pct)]
                except TypeError:
                    if sg_ratio is None:
                        sg_note = " the default value 'None'."
                    else:
                        sg_note = f": {sg_ratio}"
                    raise InvalidDataError(f"A numeric sg_ratio must be supplied. Instead, found{sg_note}")

            new_mon = Monomer(mon_type, current_size)
            state[current_size] = {MONOMER: new_mon, AFFECTED: set()}


def run_kmc(rate_dict, initial_state, initial_events, n_max=10, t_max=10, dynamics=False, random_seed=None,
            sg_ratio=None):
    """
    Performs the Gillespie algorithm using the specific event and update implementations described by do_event and
    update_events specifically. The initial state and event_dict in that state are constructed and passed to the run_kmc
    method, along with the possible rates of different bond formation event_dict, the maximum number of monomers that
    should be included in the simulation and the total simulation time.

    :param rate_dict:  dict   -- contains the reaction rate of each of the possible event_dict
    :param initial_state: dict  -- The dictionary mapping the index of each monomer to a dictionary with the monomer
        and the set of event_dict that a change to this monomer would impact
    :param initial_events: list of dicts dictionary -- The dictionary mapping event hash values to those event_dict
    :param n_max:   int   -- The maximum number of monomers in the simulation
    :param t_max: float -- The final simulation time (units depend on units of rates)
    :param dynamics: boolean -- if True, will keep values for every time step
    :param random_seed: None or hashable value to aid testing
    :param sg_ratio: needed if there is S and G and nothing else
    :return: dict with the simulation times, adjacency matrix, and list of monomers at the end of the simulation
    """
    # probably not going to use the initial_state again, but copying anyway so it remains pristine
    cur_state = copy.deepcopy(initial_state)

    # Current number of monomers
    num_monos = len(cur_state.keys())
    adj = dok_matrix((num_monos, num_monos))
    t = [0, ]

    # Calculate the rates of all of the event_dict available at the current state; ordered dict for consistency
    r_vec = OrderedDict()

    # Build the dictionary of event_dict
    event_dict = OrderedDict()
    for event in initial_events:
        event_hash = hash(event)
        r_vec[event_hash] = event.rate / num_monos
        event_dict[event_hash] = event

    if dynamics:
        adj_list = [adj.copy()]
        mon_list = [[copy.copy(cur_state[i][MONOMER]) for i in cur_state]]
    else:  # just to make IDE happy that won't use before defined
        adj_list = []
        mon_list = []

    # Run the Gillespie algorithm
    while t[-1] < t_max and len(event_dict) > 0:
        # Find the total rate for all of the possible event_dict and choose which event to do
        # Less direct conversion of r_vec to lists (keys in hashes, vals in all_rates) for consistency
        hashes = list(r_vec.keys())
        hashes.sort()
        all_rates = []
        for r_vec_key in hashes:
            all_rates.append(r_vec[r_vec_key])
        r_tot = np.sum(all_rates)

        if random_seed:
            # don't want same dt for every iteration, so add to seed with each iteration
            np.random.seed(random_seed + len(t))
            # don't let machine precision change the dt on different platforms
            rand_num = np.around(np.random.rand(), MAX_NUM_DECIMAL)
        else:
            rand_num = np.random.rand()
        j = np.random.choice(hashes, p=all_rates / r_tot)
        event = event_dict[j]
        # See how much time has passed before this event happened; rounding to reduce platform dependency
        dt = round_sig_figs((1 / r_tot) * np.log(1 / rand_num))
        t.append(t[-1] + dt)

        # Do the event and update the state
        do_event(event, cur_state, adj, sg_ratio, random_seed=random_seed)

        if dynamics:
            adj_list.append(adj.copy())
            mon_list.append([copy.copy(cur_state[i][MONOMER]) for i in cur_state])

        # Check the new state for what events are possible
        update_events(cur_state, adj, event, event_dict, r_vec, rate_dict, max_mon=n_max)

    if dynamics:
        return {TIME: t, MONO_LIST: mon_list, ADJ_MATRIX: adj_list}

    return {TIME: t, MONO_LIST: [cur_state[i][MONOMER] for i in cur_state], ADJ_MATRIX: adj}


# noinspection PyTypeChecker
def generate_mol(adj, node_list):
    """
    Based on standard molfile format https://www.daylight.com/meetings/mug05/Kappler/ctfile.pdf
    :param adj: dok_matrix
    :param node_list: list
    :return: mol_str, str in standard molfile
    """
    # define dictionary for atoms within each monomer
    atom_blocks = {G: ('C 0 0 0 0 \n' +  # 1
                       'C 0 0 0 0 \n' +  # 2
                       'C 0 0 0 0 \n' +  # 3
                       'C 0 0 0 0 \n' +  # 4
                       'C 0 0 0 0 \n' +  # 5
                       'C 0 0 0 0 \n' +  # 6
                       'C 0 0 0 0 \n' +  # 7
                       'C 0 0 0 0 \n' +  # 8
                       'C 0 0 0 0 \n' +  # 9
                       'O 0 0 0 0 \n' +  # 9-OH
                       'O 0 0 0 0 \n' +  # 3-OMe
                       'C 0 0 0 0 \n' +  # 3-OMe
                       'O 0 0 0 0 \n'),  # 4-OH
                   S: ('C 0 0 0 0 \n' +  # 1
                       'C 0 0 0 0 \n' +  # 2
                       'C 0 0 0 0 \n' +  # 3
                       'C 0 0 0 0 \n' +  # 4
                       'C 0 0 0 0 \n' +  # 5
                       'C 0 0 0 0 \n' +  # 6
                       'C 0 0 0 0 \n' +  # 7
                       'C 0 0 0 0 \n' +  # 8
                       'C 0 0 0 0 \n' +  # 9
                       'O 0 0 0 0 \n' +  # 9-OH
                       'O 0 0 0 0 \n' +  # 3-OMe
                       'C 0 0 0 0 \n' +  # 3-OMe
                       'O 0 0 0 0 \n' +  # 4-OH
                       'O 0 0 0 0 \n' +  # 5-OMe
                       'C 0 0 0 0 \n'),  # 5-OMe
                   C: ('C 0 0 0 0 \n' +  # 1
                       'C 0 0 0 0 \n' +  # 2
                       'C 0 0 0 0 \n' +  # 3
                       'C 0 0 0 0 \n' +  # 4
                       'C 0 0 0 0 \n' +  # 5
                       'C 0 0 0 0 \n' +  # 6
                       'C 0 0 0 0 \n' +  # 7
                       'C 0 0 0 0 \n' +  # 8
                       'C 0 0 0 0 \n' +  # 9
                       'O 0 0 0 0 \n' +  # 9-OH
                       'O 0 0 0 0 \n' +  # 3-OH
                       'O 0 0 0 0 \n'),  # 4-OH
                   G4: ('C 0 0 0 0 \n' +  # 1
                        'C 0 0 0 0 \n' +  # 2
                        'C 0 0 0 0 \n' +  # 3
                        'C 0 0 0 0 \n' +  # 4
                        'C 0 0 0 0 \n' +  # 5
                        'C 0 0 0 0 \n' +  # 6
                        'C 0 0 0 0 \n' +  # 7
                        'C 0 0 0 0 \n' +  # 8
                        'C 0 0 0 0 \n' +  # 9
                        'O 0 0 0 0 \n' +  # 9-OH
                        'O 0 0 0 0 \n' +  # 3-OMe
                        'C 0 0 0 0 \n' +  # 3-OMe
                        'O 0 0 0 0 RAD=2\n'),  # 4-O
                   S4: ('C 0 0 0 0 \n' +  # 1
                        'C 0 0 0 0 \n' +  # 2
                        'C 0 0 0 0 \n' +  # 3
                        'C 0 0 0 0 \n' +  # 4
                        'C 0 0 0 0 \n' +  # 5
                        'C 0 0 0 0 \n' +  # 6
                        'C 0 0 0 0 \n' +  # 7
                        'C 0 0 0 0 \n' +  # 8
                        'C 0 0 0 0 \n' +  # 9
                        'O 0 0 0 0 \n' +  # 9-OH
                        'O 0 0 0 0 \n' +  # 3-OMe
                        'C 0 0 0 0 \n' +  # 3-OMe
                        'O 0 0 0 0 RAD=2\n' +  # 4-O
                        'O 0 0 0 0 \n' +  # 5-OMe
                        'C 0 0 0 0 \n')}  # 5-OMe

    # Similarly define dictionary for bonds within each monomer -
    # NOTE: THESE MAY NEED TO CHANGE DEPENDING ON INTER-UNIT LINKAGES

    bond_blocks = {G7: ('1 1  2  \n' +  # Aromatic ring 1->2
                        '2 2  3  \n' +  # Aromatic ring 2->3
                        '1 3  4  \n' +  # Aromatic ring 3->4
                        '1 4  5  \n' +  # Aromatic ring 4->5
                        '2 5  6  \n' +  # Aromatic ring 5->6
                        '1 6  1  \n' +  # Aromatic ring 6->1
                        '2 1  7  \n' +  # Quinone methide propyl tail 1->A
                        '1 7  8  \n' +  # Propyl tail A->B
                        '1 8  9  \n' +  # Propyl tail B->G
                        '1 9  10 \n' +  # Gamma hydroxyl G->OH
                        '1 3  11 \n' +  # 3 methoxy 3->O
                        '1 11 12 \n' +  # 3 methoxy O->12
                        '2 4  13 \n'),  # 4 ketone 4->O
                   G: ('2 1  2  \n' +  # Aromatic ring 1->2
                       '1 2  3  \n' +  # Aromatic ring 2->3
                       '2 3  4  \n' +  # Aromatic ring 3->4
                       '1 4  5  \n' +  # Aromatic ring 4->5
                       '2 5  6  \n' +  # Aromatic ring 5->6
                       '1 6  1  \n' +  # Aromatic ring 6->1
                       '1 1  7  \n' +  # Ring - propyl tail 1->A
                       '2 7  8  \n' +  # Alkene propyl tail A->B
                       '1 8  9  \n' +  # Propyl tail B->G
                       '1 9  10 \n' +  # Gamma hydroxyl G->OH
                       '1 3  11 \n' +  # 3 methoxy 3->O
                       '1 11 12 \n' +  # 3 methoxy O->12
                       '1 4  13 \n'),  # 4 hydroxyl 4->OH
                   S7: ('1 1  2  \n' +  # Aromatic ring 1->2
                        '2 2  3  \n' +  # Aromatic ring 2->3
                        '1 3  4  \n' +  # Aromatic ring 3->4
                        '1 4  5  \n' +  # Aromatic ring 4->5
                        '2 5  6  \n' +  # Aromatic ring 5->6
                        '1 6  1  \n' +  # Aromatic ring 6->1
                        '2 1  7  \n' +  # Quinone methide 1->A
                        '1 7  8  \n' +  # Propyl tail A->B
                        '1 8  9  \n' +  # Propyl tail B->G
                        '1 9  10 \n' +  # Gamma hydroxyl G->OH
                        '1 3  11 \n' +  # 3 methoxy 3->O
                        '1 11 12 \n' +  # 3 methoxy O->12
                        '2 4  13 \n' +  # 4 ketone 4->O
                        '1 5  14 \n' +  # 5 methoxy 5->O
                        '1 14 15 \n'),  # 5 methoxy O->15
                   S: ('2 1  2  \n' +  # Aromatic ring 1->2
                       '1 2  3  \n' +  # Aromatic ring 2->3
                       '2 3  4  \n' +  # Aromatic ring 3->4
                       '1 4  5  \n' +  # Aromatic ring 4->5
                       '2 5  6  \n' +  # Aromatic ring 5->6
                       '1 6  1  \n' +  # Aromatic ring 6->1
                       '1 1  7  \n' +  # Ring - propyl tail 1->A
                       '2 7  8  \n' +  # Alkene propyl tail A->B
                       '1 8  9  \n' +  # Propyl tail B->G
                       '1 9  10 \n' +  # Gamma hydroxyl G->OH
                       '1 3  11 \n' +  # 3 methoxy 3->O
                       '1 11 12 \n' +  # 3 methoxy O->12
                       '1 4  13 \n' +  # 4 hydroxyl 4->OH
                       '1 5  14 \n' +  # 5 methoxy 5->O
                       '1 14 15 \n'),  # 5 methoxy O->15
                   C: ('2 1  2  \n' +  # Aromatic ring 1->2
                       '1 2  3  \n' +  # Aromatic ring 2->3
                       '2 3  4  \n' +  # Aromatic ring 3->4
                       '1 4  5  \n' +  # Aromatic ring 4->5
                       '2 5  6  \n' +  # Aromatic ring 5->6
                       '1 6  1  \n' +  # Aromatic ring 6->1
                       '1 1  7  \n' +  # Ring - propyl tail 1->A
                       '2 7  8  \n' +  # Alkene propyl tail A->B
                       '1 8  9  \n' +  # Propyl tail B->G
                       '1 9  10 \n' +  # Gamma hydroxyl G->OH
                       '1 3  11 \n' +  # 3 hydroxyl 3->O
                       '1 4  12 \n')}  # 4 hydroxyl 4->OH

    mol_str = '\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\nM  V30 BEGIN CTAB\n'  # Header information
    mol_atom_blocks = 'M  V30 BEGIN ATOM\n'
    mol_bond_blocks = 'M  V30 BEGIN BOND\n'
    atom_line_num = 1
    bond_line_num = 1
    mono_start_idx_bond = []
    mono_start_idx_atom = []
    removed = {BONDS: 0, ATOMS: 0}

    site_positions = {1: {x: 0 for x in [G, S, C]},
                      4: {C: 11, S: 12, G: 12},
                      5: {x: 4 for x in [G, S, C]},
                      7: {x: 6 for x in [G, S, C]},
                      8: {x: 7 for x in [G, S, C]},
                      10: {x: 9 for x in [G, S, C]}}
    alpha_beta_alkene_location = 7
    alpha_ring_location = 6
    alpha = 7

    # to make IDE happy:
    atom_block = None
    bond_block = None

    # Build the individual monomers before they are linked by anything
    for i, mon in enumerate(node_list):
        if mon.type == G or mon.type == S:
            if mon.active == 0 or mon.active == -1:
                if mon.type == G:
                    atom_block = atom_blocks[G]
                    bond_block = bond_blocks[G]
                else:
                    atom_block = atom_blocks[S]
                    bond_block = bond_blocks[S]
            elif mon.active == 4:
                if mon.type == G:
                    atom_block = atom_blocks[G4]
                    bond_block = bond_blocks[G]
                else:
                    atom_block = atom_blocks[S4]
                    bond_block = bond_blocks[S]
            elif mon.active == 7:
                if mon.type == G:
                    atom_block = atom_blocks[G]
                    bond_block = bond_blocks[G7]
                else:
                    atom_block = atom_blocks[S]
                    bond_block = bond_blocks[S7]
        elif mon.type == C:
            atom_block = atom_blocks[C]
            bond_block = bond_blocks[C]
        else:
            raise ValueError("Expected monomer types are {LIGNIN_SUBUNITS} but encountered type '{mon.type}'")

        # Extract each of the individual atoms from this monomer to add to the aggregate file
        lines = atom_block.splitlines(keepends=True)

        # Figure out what atom and bond number this monomer is starting at
        mono_start_idx_bond.append(bond_line_num)
        mono_start_idx_atom.append(atom_line_num)

        # Loop through the lines of the atom block and add the necessary prefixes to the lines, using a continuing
        #     atom index
        for line in lines:
            mol_atom_blocks += f'M  V30 {atom_line_num} {line}'
            atom_line_num += 1
        # END ATOM AGGREGATION

        # Extract each of the individual bonds that defines the monomer skeleton and add it into the aggregate string
        lines = bond_block.splitlines(keepends=True)

        # Recall where this monomer started
        # So that we can add the defined bond indices to this start index to get the bond defs
        start_index = mono_start_idx_atom[-1] - 1

        # Loop through the lines of the bond block and add necessary prefixes to the lines, then modify as needed after
        for line in lines:
            # Extract the defining information about the monomer skeleton
            bond_vals = re.split(' +', line)

            # The first element is the bond order, followed by the indices of the atoms that are connected by this bond.
            # These indices need to be updated based on the true index of this monomer, not the 1 -> ~15 indices that
            #     it started with
            bond_order = bond_vals[0]
            bond_connects = [int(bond_vals[1]) + start_index, int(bond_vals[2]) + start_index]

            # Now save the true string for defining this bond, along with the cumulative index of the bond
            mol_bond_blocks += f'M  V30 {bond_line_num} {bond_order} {bond_connects[0]} {bond_connects[1]} \n'
            bond_line_num += 1
        # END BOND AGGREGATION
    # END MONOMER AGGREGATION

    # Now that we have all of the monomers in one string, we just need to add the bonds and atoms that are defined by
    #     the adjacency matrix
    bonds = mol_bond_blocks.splitlines(keepends=True)
    atoms = mol_atom_blocks.splitlines(keepends=True)

    break_alkene = {(4, 8): True, (5, 8): True, (8, 8): True, (5, 5): False, (1, 8): True, (4, 7): False, (4, 5): False}
    hydrate = {(4, 8): True, (5, 8): False, (8, 8): False, (5, 5): False, (1, 8): True, (4, 7): False, (4, 5): False}
    beta = {(4, 8): 1, (8, 4): 0, (5, 8): 1, (8, 5): 0, (1, 8): 1, (8, 1): 0}
    make_alpha_ring = {(4, 8): False, (5, 8): True, (8, 8): True, (5, 5): False, (1, 8): False, (4, 7): False,
                       (4, 5): False}

    # Start by looping through the adjacency matrix, one bond at a time (corresponds to a pair iterator)
    paired_adj = zip(*[iter(dict(adj))] * 2)

    for pair in paired_adj:
        # Find the types of bonds and indices associated with each of the elements in the adjacency matrix
        # Indices are extracted as tuples (row,col) and we just want the row for each
        mono_indices = [x[0] for x in pair]

        # Get the monomers corresponding to the indices in this bond
        mons = [None, None]

        for mon in node_list:
            if mon.identity == mono_indices[0]:
                mons[0] = mon
            elif mon.identity == mono_indices[1]:
                mons[1] = mon

        # Now just extract the bond types (the value from the adj matrix)
        bond_loc = [int(adj[p]) for p in pair]

        # Get the indices of the atoms being bound -> Count from where the monomer starts, and add however many is
        #     needed to reach the desired position for that specific monomer type and bonding site
        atom_indices = [mono_start_idx_atom[mono_indices[i]] + site_positions[bond_loc[i]][mons[i].type]
                        for i in range(2)]

        # Make the string to add to the molfile
        bond_string = f'M  V30 {bond_line_num} 1 {atom_indices[0]} {atom_indices[1]} \n'
        bond_line_num += 1

        # Append the newly created bond to the file
        bonds.append(bond_string)

        # Check if the alkene needs to be modified to a single bond
        bond_loc_tuple = tuple(sorted(bond_loc))
        if break_alkene[bond_loc_tuple]:
            for i in range(2):
                if adj[pair[i]] == 8 and mons[i].active != 7:  # Monomer index is bound through beta
                    # Find the bond index corresponding to alkene bond
                    alkene_bond_index = mono_start_idx_bond[mono_indices[i]] + alpha_beta_alkene_location - \
                                        removed[BONDS]

                    # Get all of the required bond information (index,order,monIdx1,monIdx2)
                    bond_vals = re.split(' +', bonds[alkene_bond_index])[2:]
                    try:
                        assert (int(bond_vals[0]) == alkene_bond_index + removed[BONDS])
                    except AssertionError:
                        print(f'Expected index: {bond_vals[0]}, Index obtained: {alkene_bond_index}')

                    # Decrease the bond order by 1
                    bonds[alkene_bond_index] = f'M  V30 {bond_vals[0]} 1 {bond_vals[2]} {bond_vals[3]} \n'

        # Check if we need to add water to the alpha position
        if hydrate[bond_loc_tuple] and 7 not in adj[mono_indices[beta[tuple(bond_loc)]]].values() and \
                mons[beta[tuple(bond_loc)]].active != 7:
            if mons[int(not beta[tuple(bond_loc)])].type != C:
                # We should actually only be hydrating BO4 bonds when the alpha position is unoccupied (handled by
                # second clause above)

                # Find the location of the alpha position
                alpha_idx = mono_start_idx_atom[mono_indices[beta[tuple(bond_loc)]]] + site_positions[7][
                    mons[beta[tuple(bond_loc)]].type]

                # Add the alpha hydroxyl O atom
                atoms.append(f'M  V30 {atom_line_num} O 0 0 0 0 \n')
                atom_line_num += 1

                # Make the bond
                bonds.append(f'M  V30 {bond_line_num} 1 {alpha_idx} {atom_line_num - 1} \n')
                bond_line_num += 1
            else:
                # Make the benzodioxane linkage
                alpha_idx = mono_start_idx_atom[mono_indices[beta[tuple(bond_loc)]]] + site_positions[7][
                    mons[beta[tuple(bond_loc)]].type]
                hydroxy_index = mono_start_idx_atom[mono_indices[int(not beta[tuple(bond_loc)])]] + (
                        site_positions[4][mons[beta[tuple(bond_loc)]].type] - 1)  # subtract 1 to move from 4-OH to 3-OH

                # Make the bond
                bonds.append(f'M  V30 {bond_line_num} 1 {alpha_idx} {hydroxy_index} \n')
                bond_line_num += 1

        # Check if there will be a ring involving the alpha position
        if make_alpha_ring[bond_loc_tuple]:
            other_site = {(5, 8): 4, (8, 8): 10}
            for i in range(2):
                if adj[pair[i]] == 8:  # This index is bound through beta (will get alpha connection)
                    # Find the location of the alpha position and the position that cyclizes with alpha
                    alpha_idx = mono_start_idx_atom[mono_indices[i]] + site_positions[7][mons[i].type]
                    other_idx = mono_start_idx_atom[mono_indices[int(not i)]] + \
                        site_positions[other_site[bond_loc_tuple]][mons[int(not i)].type]

                    bonds.append(f'M  V30 {bond_line_num} 1 {alpha_idx} {other_idx} \n')
                    bond_line_num += 1

        # For the B1 bond:
        #     1) Disconnect the original 1 -> A bond that existed from the not beta monomer
        #     2) Convert the new primary alcohol to an aldehyde
        if sorted(bond_loc) == [1, 8]:
            # TODO: make sure all B1 bonds are correctly forms
            warning("There are problems in how this program currently builds molecules with B1 bonds. Carefully "
                    "check any output.")
            index_for_one = int(not beta[tuple(bond_loc)])
            # Convert the alpha alcohol on one's tail to an aldehyde
            alpha_idx = mono_start_idx_atom[mono_indices[index_for_one]
                                            ] + site_positions[alpha][mons[index_for_one].type]

            # Temporarily join the bonds so that we can find the string
            temp = ''.join(bonds)
            matches = re.findall(f'M {2}V30 [0-9]+ 1 {alpha_idx} [0-9]+', temp)

            # Find the bond connecting the alpha to the alcohol
            others = []
            for possibility in matches:
                bound_atoms = re.split(' +', possibility)[4:]
                others.extend([int(x) for x in bound_atoms if int(x) != alpha_idx])

            # TODO: Fix the section below--it does not work; for now, simply catch the error and provide a
            #   descriptive failure message (instead of a stack trace)
            # The oxygen atom should have the greatest index of the atoms bound to the alpha position because it
            #     was added last
            try:
                oxygen_atom_index = max(others)
                bonds = re.sub(f'1 {alpha_idx} {oxygen_atom_index}',
                               f'2 {alpha_idx} {oxygen_atom_index}', temp).splitlines(keepends=True)

                # Find where the index for the bond is and remove it from the array
                alpha_ring_bond_index = mono_start_idx_bond[mono_indices[index_for_one]
                                                            ] + alpha_ring_location - removed[BONDS]
                del (bonds[alpha_ring_bond_index])
                removed[BONDS] += 1
            except ValueError:
                raise InvalidDataError("This program cannot currently generate a molecule with this beta-1 bond. Sorry!"
                                       "\n    Try running again, as this bond type is only occasionally created.")

    mol_bond_blocks = ''.join(bonds)
    mol_atom_blocks = ''.join(atoms)

    mol_atom_blocks += 'M  V30 END ATOM \n'
    mol_bond_blocks += 'M  V30 END BOND \n'
    counts = f'M  V30 COUNTS {atom_line_num - 1 - removed[ATOMS]} {bond_line_num - 1 - removed[BONDS]} 0 0 0\n'
    mol_str += counts + mol_atom_blocks + mol_bond_blocks + 'M  V30 END CTAB\nM  END'

    return mol_str


def write_patch(open_file, patch_name, segname, resid1, resid2=None):
    """
    Simple script to consistently format patch output for tcl script
    :param open_file: {TextIOWrapper}
    :param patch_name: str
    :param segname: str
    :param resid1: int
    :param resid2: int
    :return: what to write to file
    """
    if resid2:
        open_file.write(f"patch {patch_name} {segname}:{resid1} {segname}:{resid2}\n")
    else:
        open_file.write(f"patch {patch_name} {segname}:{resid1}\n")


def gen_tcl(orig_adj, monomers, tcl_fname=DEF_TCL_FNAME, psf_fname=DEF_PSF_FNAME, chain_id=DEF_CHAIN_ID,
            toppar_dir=DEF_TOPPAR, out_dir=None):
    """
    This takes a computed adjacency matrix and monomer list and writes out a script to generate a psf file of the
    associated structure, suitable for feeding into the LigninBuilder plugin of VMD
    (https://github.com/jvermaas/LigninBuilder).

    :param orig_adj: dok_matrix, Adjacency matrix generated by the kinetic Monte Carlo process
    :param monomers: list of Monomer objects, Monomer list from the kinetic Monte Carlo process
    :param tcl_fname: str, desired output filename
    :param psf_fname: str, desired basename for psf and pdb files to be generated by VMD
    :param chain_id: str, desired `chainID` to be used in the resulting output segment name for the generated lignin
    :param toppar_dir: location where the topology files top_lignin.top and top_all36_cgenff.rtf are expected
    :param out_dir: directory where the .tcl file should be saved
    :return:
    """
    adj = orig_adj.copy()
    # In the unlikely event that chain_id starts with a space (won't happen if provided through create_lignin),
    #     get rid of it, as it would cause problems. Also make it uppercase, as that is what PDBs use.
    chain_id = chain_id.strip().upper()
    # in the unlikely event that chain_id is empty, which would also cause problems, give it the default, instead
    #     of just failing at the end, after all the work of generating the lignin. Also truncate a too-long chainID.
    if not chain_id or len(chain_id) > 1:
        if len(chain_id) > 1:
            chain_id = chain_id[0]
        else:
            chain_id = DEF_CHAIN_ID
        warning(f"ChainID's for PDBs should be one character. Will use: '{chain_id}' as the chainID.")
    residue_letter = {G: 'G', S: 'S', H: 'H', C: 'C'}
    f_out = create_out_fname(tcl_fname, base_dir=out_dir)
    # add a mac/linux dir separator if there isn't already a directory separator, and if there is to be a subdirectory
    #   (not None or "")
    if toppar_dir and len(toppar_dir) > 0:
        if toppar_dir[-1] != '/' and (toppar_dir[-1] != '\\'):
            toppar_dir += "/"
    else:
        toppar_dir = ""
    with open(f_out, "w") as f:
        f.write(f"package require psfgen\n"
                f"topology {toppar_dir}{'top_all36_cgenff.rtf'}\n"
                f"topology {toppar_dir}{'top_lignin.top'}\n"
                f"segment {chain_id} {{\n")
        for monomer in monomers:
            resid = monomer.identity + 1
            res_name = residue_letter[monomer.type]
            f.write(f"    residue {resid} {res_name}\n")
        f.write(f"}}\n")

        for row in (adj == 1).nonzero()[0]:
            col = (adj.getrow(row) == 8).nonzero()[1]
            if len(col):
                col = col[0]
                adj[(row, col)] *= -1
        for bond_matrix_tuple in adj.keys():
            if bond_matrix_tuple[0] > bond_matrix_tuple[1]:
                continue
            psf_patch_resid1 = bond_matrix_tuple[0] + 1
            psf_patch_resid2 = bond_matrix_tuple[1] + 1
            flipped_bond_matrix_tuple = (bond_matrix_tuple[1], bond_matrix_tuple[0])
            bond_loc1 = int(adj[bond_matrix_tuple])
            bond_loc2 = int(adj[flipped_bond_matrix_tuple])
            if bond_loc1 == 8 and bond_loc2 == 4:  # Beta-O-4 linkage
                write_patch(f, "BO4", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 4 and bond_loc2 == 8:  # Reverse beta-O-4 linkage.
                write_patch(f, "BO4", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 5 and monomers[bond_matrix_tuple[1]].type == G:  # B5G linkage
                write_patch(f, "B5G", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 8 and monomers[bond_matrix_tuple[0]].type == G:  # Reverse B5G linkage
                write_patch(f, "B5G", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 5 and monomers[bond_matrix_tuple[1]].type == C:  # B5C linkage
                write_patch(f, "B5C", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 8 and monomers[bond_matrix_tuple[0]].type == C:  # Reverse B5C linkage
                write_patch(f, "B5C", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 5 and bond_loc2 == 5:  # 55 linkage
                write_patch(f, "B5C", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 7 and bond_loc2 == 4:  # alpha-O-4 linkage
                write_patch(f, "AO4", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 4 and bond_loc2 == 7:  # Reverse alpha-O-4 linkage
                write_patch(f, "AO4", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 4 and bond_loc2 == 5:  # 4O5 linkage
                write_patch(f, "4O4", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 4:  # Reverse 4O5 linkage
                write_patch(f, "4O4", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 1:  # Beta-1 linkage
                write_patch(f, "B1", chain_id, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 1 and bond_loc2 == 8:  # Reverse beta-1 linkage
                write_patch(f, "B1", chain_id, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == -8 and bond_loc2 == 4:  # Beta-1 linkage remnant
                write_patch(f, "O4AL", chain_id, psf_patch_resid2)
            elif bond_loc2 == -8 and bond_loc1 == 4:  # Reverse beta-1 remnant
                write_patch(f, "O4AL", chain_id, psf_patch_resid1)
            elif bond_loc1 == -8 and bond_loc2 == 1:  # Beta-1 linkage remnant (C1 variant)
                write_patch(f, "C1AL", chain_id, psf_patch_resid2)
            elif bond_loc2 == -8 and bond_loc1 == 1:  # Reverse beta-1 remnant (C1 variant)
                write_patch(f, "C1AL", chain_id, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 8:  # beta-beta linkage
                write_patch(f, "BB", chain_id, psf_patch_resid1, psf_patch_resid2)
            else:
                raise InvalidDataError(f"Encountered unexpected linkage: adj_matrix loc: {bond_matrix_tuple}, "
                                       f"bond locations: {bond_loc1} and {bond_loc2}, monomer types: "
                                       f"{monomers[bond_matrix_tuple[0]].type} and "
                                       f"{monomers[bond_matrix_tuple[1]].type}")
        f.write(f"regenerate angles dihedrals\nwritepsf {psf_fname}.psf\n")

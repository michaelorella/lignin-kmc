# !/usr/bin/env python
# coding=utf-8

"""
Written:        2018-06-06 Michael Orella
Last Edited:    2019-11-11 by Heather Mayes

Code base for simulating the in planta polymerization of monolignols through Gillepsie algorithm adaptations.
"""

import scipy.sparse as sp
import numpy as np
import copy
from common_wrangler.common import InvalidDataError
from ligninkmc.event import Event
from ligninkmc.monomer import Monomer
from ligninkmc.kmc_common import (AO4, B1, B5, BB, BO4, C5C5, C5O4, OX, Q, GROW, TIME, DIMER, MONOMER, AFFECTED,
                                  ADJ_MATRIX, MONO_LIST)


def quick_frag_size(monomer):
    """
    An easy check on a specific monomer to tell if it is a monomer or involved in a dimer. This is used over the
    detailed fragment_size(frags) calculation in the simulation of lignification for performance benefits. However,
    extensions beyond dimers would be difficult, if it is found that there are significant impacts on chain
    length > dimer

    :param monomer: monomer -- Monomer -- the monomer object that we want to know if it is bound to anything else
        (i.e. if it is truly a monomer still)
    :return: String either 'monomer' or 'dimer' if it is connected to nothing else, or isn't respectively
    """
    if monomer.type == 0 and monomer.open == {4, 5, 8}:  # Guaiacol monomer
        return MONOMER
    elif monomer.type == 1 and monomer.open == {4, 8}:  # Syringol monomer
        return MONOMER
    elif monomer.type == 2 and monomer.open == {4, 5, 8}:  # Caffeoyl monomer
        return MONOMER
    return DIMER


def update_events(monomers, adj, last_event, events, rate_vec, rate_dict, max_mon=500):
    """
    The meat of the implementation for lignification specific KMC. This method determines what the possible events are
    in a given state, where the state is the current simulation state. Most of the additional parameters in this method
    are added for performance benefits rather than necessity.

    :param monomers: monomers is a dictionary that maps the index of each monomer in the simulation to the monomer
        itself and the events that would be effected by a change to the monomer key. This makes it easy to quickly
        determine which of the events in the simulation need to be updated and which should not be changed.
    :param adj: dok_matrix  -- The current state of the simulation represented by the adjacency matrix containing all
        of the monomers and the bonds between them (if any)
    :param last_event: The previous Event that occurred, which will tell us what monomers were effected. When combined
        with the state dictionary, this allows for efficient updating of the set of events that are possible
    :param events: dictionary  -- The set of all possible unique events that must be updated and returned from this
        method, implemented in a hash map where the event hash value is the key
    :param rate_vec: dictionary  -- The rates of all of the unique events implemented in a hash map where the Event
        hash value is the key
    :param rate_dict: dictionary  -- The dictionary of the possible rates involved in each reaction, where the possible
        reactions are OX,B1,C5O4,AO4,C5C5,B5,Q,AO4, and . These are calculated a priori from DFT
    :param max_mon: int -- The maximum number of monomers that should be stored in the simulation
    :return: v
    """

    # Map the monomer active state to the possible events it can do
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
    # Only do these for bonding and oxidation events, any growth does not actually change the possible events
    if last_event.key != GROW:
        # Remove the last event that we just did from the set of events that can be performed
        le_hash = hash(last_event)
        del (events[le_hash])
        del (rate_vec[le_hash])

        # Make sure to keep track of which partners have been "cleaned" already - i.e. what monomers have already had
        #     all of the old events removed
        cleaned_partners = set()

        # Get indices of monomers that were acted upon
        affected_monomers = last_event.index

        # Update events from the perspective of each monomer that was just affected individually
        # This prevents having to re-search the entire space every time, saving significant computation
        for mon_id in affected_monomers:
            # Get the affected monomer
            mon = monomers[mon_id][MONOMER]

            # Get the sets of activated monomers that we could bind with
            ox = set()
            quinone = set()

            other_ids = [x for x in monomers if x != mon_id]
            for other in other_ids:
                other_mon_type = monomers[other][MONOMER]
                # Don't allow connections that would cyclize the polymer!
                if other_mon_type.active == 4 and other_mon_type.identity not in mon.connectedTo:
                    ox.add(other_mon_type)
                elif other_mon_type.active == 7 and other_mon_type.identity not in mon.connectedTo:
                    quinone.add(monomers[other][MONOMER])
            bonding_partners = {BO4: ox, B5: ox, C5O4: ox, C5C5: ox, BB: ox, B1: ox, AO4: quinone}

            # Obtain the events that are affected by a change to the monomer that was just acted on
            events_to_be_modified = monomers[mon_id][AFFECTED]

            # Get the codes for the events that are possible based on how the current monomer behaves
            active_pos = mon.active
            new_event_list = possible_events[active_pos]

            # Take any events to be modified out of the set so that we can replace with the updated events
            for event in events_to_be_modified:
                ev_hash = hash(event)
                if ev_hash in events:
                    del (events[ev_hash])
                    del (rate_vec[ev_hash])

            # Overwrite the old events that could have been modified from this monomer being updated
            monomers[mon_id][AFFECTED] = set()
            cur_n, _ = adj.get_shape()

            for rxn_event in new_event_list:
                if rxn_event and rxn_event[1] == 1:  # Unimolecular reaction event
                    size = quick_frag_size(mon)

                    rate = rxn_event[2][mon.type][size] / cur_n

                    # Add the event to the set of events modifiable by changing the monomer, and update the set of all
                    # events at the next time step
                    monomers[mon_id][AFFECTED].add(Event(rxn_event[0], [mon.identity], rate))

                elif rxn_event and rxn_event[1] == 2:  # Bimolecular reaction event
                    bond = tuple(rxn_event[3])
                    alt = (bond[1], bond[0])
                    for partner in bonding_partners[rxn_event[0]]:
                        # Sanitize the set of events that can be effected
                        if partner not in cleaned_partners:
                            # Remove any old events from
                            monomers[partner.identity][AFFECTED].difference_update(events_to_be_modified)
                            cleaned_partners.add(partner)

                        index = [mon.identity, partner.identity]
                        back = [partner.identity, mon.identity]

                        # Add the bond from one monomer to the other in the default config
                        size = (quick_frag_size(mon), quick_frag_size(partner))
                        # # Here to show that now 2+/2+ can make beta-O4 monomers
                        # if size == (DIMER, DIMER):
                        #     if rxn_event[0] == BO4:
                        #         print("If the option to link 2+ to 2+ oligomers by a beta-O-4"
                        #                "bond is included, it can happen!")
                        if bond[0] in mon.open and bond[1] in partner.open:
                            try:
                                rate = rxn_event[2][(mon.type, partner.type)][size] / (cur_n ** 2)
                            except KeyError:
                                raise InvalidDataError(f"Error while attempting to update events: "
                                                       f"event {rxn_event[0]} between indices {mon.identity} and "
                                                       f"{partner.identity} ")

                            # Add this to both the monomer and it's bonding partners list of events that need to be
                            # modified upon manipulation of either monomer
                            monomers[mon_id][AFFECTED].add(Event(rxn_event[0], index, rate, bond))  # this -> other
                            monomers[partner.identity][AFFECTED].add(
                                Event(rxn_event[0], index, rate, bond))  # this -> other

                            # Switch the order
                            # other -> this
                            monomers[mon_id][AFFECTED].add(Event(rxn_event[0], back, rate, alt))
                            # other -> this
                            monomers[partner.identity][AFFECTED].add(Event(rxn_event[0], back, rate, alt))

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
                                monomers[mon_id][AFFECTED].add(Event(rxn_event[0], index, rate, alt))
                                monomers[partner.identity][AFFECTED].add(
                                    Event(rxn_event[0], index, rate, alt))  # this -> other alt

                                # Switch the order
                                # other -> this alt
                                monomers[mon_id][AFFECTED].add(Event(rxn_event[0], back, rate, bond))
                                # other -> this alt
                                monomers[partner.identity][AFFECTED].add(Event(rxn_event[0], back, rate, bond))
                    # END LOOP OVER PARTNERS
                # END UNIMOLECULAR/BIMOLECULAR BRANCH
            # END LOOP OVER NEW REACTION POSSIBILITIES
            for event in monomers[mon_id][AFFECTED]:
                ev_hash = hash(event)
                events[ev_hash] = event
                rate_vec[ev_hash] = event.rate
        # END LOOP OVER MONOMERS THAT WERE AFFECTED BY LAST EVENT
    else:
        cur_n, _ = adj.get_shape()

        # If the system has grown to the maximum size, delete the event for adding more monomers
        if cur_n >= max_mon:
            le_hash = hash(last_event)
            del (events[le_hash])
            del (rate_vec[le_hash])

        # Reflect the larger system volume
        for i in rate_vec:
            if events[i].key != GROW:
                rate_vec[i] = rate_vec[i] * (cur_n - 1) / cur_n

        # Add an event to oxidize the monomer that was just added to the
        # simulation
        oxidation = Event(OX, [cur_n - 1], rate_dict[OX][monomers[cur_n - 1][MONOMER].type][MONOMER])
        monomers[cur_n - 1][AFFECTED].add(oxidation)
        ev_hash = hash(oxidation)
        events[ev_hash] = oxidation
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


def connected_size(mon=None):
    if mon == mon.parent:
        return mon.size
    else:
        return connected_size(mon=mon.parent)


def do_event(event, state, adj, sg_ratio=None):
    """
    The second key component of the lignin implementation of the Monte Carlo algorithm, this method actually executes
    the chosen event on the current state and modifies it to reflect the updates.

    :param event: The event object that should be executed on the current state
    :param state: dict, The dictionary of dictionaries that contains the state information for each monomer
    :param adj: dok_matrix, The adjacency matrix in the current state
    :param sg_ratio: float needed if and only if:
                         a) there are S and G and only S and G, and
                         b) new monomers will be added
    :return: N/A - mutates the list of monomers and adjacency matrix instead
    """

    indices = event.index
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

        if mon0.active == 7 and mon1.type == 2:
            mon0.active = 0
            mon0.open -= {7}

        if mon1.active == 7 and mon0.type == 2:
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
            if monomers and monomers[-1].type == 2:
                mon_type = 2
            else:
                try:
                    pct = sg_ratio / (1 + sg_ratio)
                    mon_type = int(np.random.rand() < pct)
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
    update_events specifically. The initial state and events in that state are constructed and passed to the run_kmc
    method, along with the possible rates of different bond formation events, the maximum number of monomers that
    should be included in the simulation and the total simulation time.

    :param rate_dict:  dict   -- contains the reaction rate of each of the possible events
    :param initial_state: dict  -- The dictionary mapping the index of each monomer to a dictionary with the monomer
        and the set of events that a change to this monomer would impact
    :param initial_events: dictionary -- The dictionary mapping event hash values to those events
    :param n_max:   int   -- The maximum number of monomers in the simulation
    :param t_max: float -- The final simulation time (units depend on units of rates)
    :param dynamics: boolean -- if True, will keep values for every time step
    :param random_seed: None or hashable value to aid testing
    :param sg_ratio: needed if there is S and G and nothing else
    :return: dict with the simulation times, adjacency matrix, and list of monomers at the end of the simulation
    """

    state = copy.deepcopy(initial_state)
    events = copy.deepcopy(initial_events)

    # Current number of monomers
    n = len(state.keys())
    adj = sp.dok_matrix((n, n))
    t = [0, ]

    # Calculate the rates of all of the events available at the current state
    r_vec = {}

    # Build the dictionary of events
    event_dict = {}
    for event in sorted(events):
        r_vec[hash(event)] = event.rate / n
        event_dict[hash(event)] = event

    if dynamics:
        adj_list = [adj.copy()]
        mon_list = [[copy.copy(state[i][MONOMER]) for i in state]]
    else:  # just to make IDE happy that won't use before defined
        adj_list = []
        mon_list = []

    # Run the Gillespie algorithm
    while t[-1] < t_max and len(event_dict) > 0:
        # Find the total rate for all of the possible events and choose which event to do
        hashes = list(r_vec.keys())
        # <class 'list'>: [635582509440568386, 7038641671649327019, 4826763781572526596, -1046983068416491331]
        all_rates = list(r_vec.values())
        r_tot = np.sum(all_rates)

        if random_seed:
            np.random.seed(random_seed)
        j = np.random.choice(hashes, p=all_rates / r_tot)
        event = event_dict[j]

        # See how much time has passed before this event happened
        dt = (1 / r_tot) * np.log(1 / np.random.rand(1))

        t.extend(t[-1] + dt)

        # Do the event and update the state
        do_event(event, state, adj, sg_ratio)

        if dynamics:
            adj_list.append(adj.copy())
            mon_list.append([copy.copy(state[i][MONOMER]) for i in state])

        # Check the new state for what events are possible
        update_events(state, adj, event, event_dict, r_vec, rate_dict, max_mon=n_max)

    if dynamics:
        return {TIME: t, MONO_LIST: mon_list, ADJ_MATRIX: adj_list}

    return {TIME: t, MONO_LIST: [state[i][MONOMER] for i in state], ADJ_MATRIX: adj}

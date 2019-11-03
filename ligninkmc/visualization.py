# !/usr/bin/env python
# coding=utf-8

import re
import networkx as nx
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from ligninkmc.kmc_common import (G, S, C, S4, G4, G7, ATOMS, BONDS)
from common_wrangler.common import (InvalidDataError)

DrawingOptions.bondLineWidth = 1.2
S7 = 'S7'


def generate_mol(adj, node_list):
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
    monomer_start_idx_bond = []
    monomer_start_idx_atom = []
    removed = {BONDS: 0, ATOMS: 0}

    site_positions = {1: {x: 0 for x in [0, 1, 2]},
                      4: {2: 11, 1: 12, 0: 12},
                      5: {x: 4 for x in [0, 1, 2]},
                      7: {x: 6 for x in [0, 1, 2]},
                      8: {x: 7 for x in [0, 1, 2]},
                      10: {x: 9 for x in [0, 1, 2]}}
    alpha_beta_alkene_location = 7
    alpha_ring_location = 6
    alpha = 7

    # to make IDE happy:
    atom_block = None
    bond_block = None

    # Build the individual monomers before they are linked by anything
    for i, mon in enumerate(node_list):
        if mon.type == 0 or mon.type == 1:
            if mon.active == 0 or mon.active == -1:
                if mon.type == 0:
                    atom_block = atom_blocks[G]
                    bond_block = bond_blocks[G]
                else:
                    atom_block = atom_blocks[S]
                    bond_block = bond_blocks[S]
            elif mon.active == 4:
                if mon.type == 0:
                    atom_block = atom_blocks[G4]
                    bond_block = bond_blocks[G]
                else:
                    atom_block = atom_blocks[S4]
                    bond_block = bond_blocks[S]
            elif mon.active == 7:
                if mon.type == 0:
                    atom_block = atom_blocks[G]
                    bond_block = bond_blocks[G7]
                else:
                    atom_block = atom_blocks[S]
                    bond_block = bond_blocks[S7]
        elif mon.type == 2:
            atom_block = atom_blocks[C]
            bond_block = bond_blocks[C]
        else:
            raise ValueError("Expected types for monomer to be 0, 1, or 2, but encountered type '{}'".format(mon.type))

        # Extract each of the individual atoms from this monomer to add to the aggregate file
        lines = atom_block.splitlines(keepends=True)

        # Figure out what atom and bond number this monomer is starting at
        monomer_start_idx_bond.append(bond_line_num)
        monomer_start_idx_atom.append(atom_line_num)

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
        start_index = monomer_start_idx_atom[-1] - 1

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
        monomer_indices = [x[0] for x in pair]

        # Get the monomers corresponding to the indices in this bond
        mons = [None, None]

        for mon in node_list:
            if mon.identity == monomer_indices[0]:
                mons[0] = mon
            elif mon.identity == monomer_indices[1]:
                mons[1] = mon

        # Now just extract the bond types
        bond = [adj[p] for p in pair]

        # Get the indices of the atoms being bound -> Count from where the monomer starts, and add however many is
        #     needed to reach the desired position for that specific monomer type and bonding site
        atom_indices = [monomer_start_idx_atom[monomer_indices[i]] + site_positions[bond[i]][mons[i].type]
                        for i in range(2)]

        # Make the string to add to the molfile
        bond_string = f'M  V30 {bond_line_num} 1 {atom_indices[0]} {atom_indices[1]} \n'
        bond_line_num += 1

        # Append the newly created bond to the file
        bonds.append(bond_string)

        # Check if the alkene needs to be modified to a single bond
        if break_alkene[tuple(sorted(bond))]:
            for index in range(2):
                if adj[pair[index]] == 8 and mons[index].active != 7:  # Monomer index is bound through beta
                    # Find the bond index corresponding to alkene bond
                    alkeneBondIndex = monomer_start_idx_bond[monomer_indices[index]] + alpha_beta_alkene_location - removed[
                        BONDS]

                    # Get all of the required bond information (index,order,monIdx1,monIdx2)
                    bond_vals = re.split(' +', bonds[alkeneBondIndex])[2:]
                    try:
                        assert (int(bond_vals[0]) == alkeneBondIndex + removed[BONDS])
                    except AssertionError:
                        print(f'Expected index: {bond_vals[0]} Index obtained: {alkeneBondIndex}')

                    # Decrease the bond order by 1
                    bonds[alkeneBondIndex] = f'M  V30 {bond_vals[0]} 1 {bond_vals[2]} {bond_vals[3]} \n'

        # Check if we need to add water to the alpha position

        if hydrate[tuple(sorted(bond))] and 7 not in adj[monomer_indices[beta[tuple(bond)]]].values() and mons[
            beta[tuple(bond)]].active != 7:
            if mons[int(not beta[tuple(bond)])].type != 2:
                # We should actually only be hydrating BO4 bonds when the alpha position is unoccupied (handled by second clause above)

                # Find the location of the alpha position
                alpha_index = monomer_start_idx_atom[monomer_indices[beta[tuple(bond)]]] + site_positions[7][
                    mons[beta[tuple(bond)]].type]

                # Add the alpha hydroxyl O atom
                atoms.append(f'M  V30 {atom_line_num} O 0 0 0 0 \n')
                atom_line_num += 1

                # Make the bond
                bonds.append(f'M  V30 {bond_line_num} 1 {alpha_index} {atom_line_num - 1} \n')
                bond_line_num += 1
            else:
                # Make the benzodioxane linkage
                alpha_index = monomer_start_idx_atom[monomer_indices[beta[tuple(bond)]]] + site_positions[7][
                    mons[beta[tuple(bond)]].type]
                hydroxy_index = monomer_start_idx_atom[monomer_indices[int(not beta[tuple(bond)])]] + (
                        site_positions[4][mons[beta[tuple(bond)]].type] - 1)  # subtract 1 to move from 4-OH to 3-OH

                # Make the bond
                bonds.append(f'M  V30 {bond_line_num} 1 {alpha_index} {hydroxy_index} \n')
                bond_line_num += 1

        # Check if there will be a ring involving the alpha position
        if make_alpha_ring[tuple(sorted(bond))]:
            other_site = {(5, 8): 4, (8, 8): 10}
            for index in range(2):
                if adj[pair[index]] == 8:  # This index is bound through beta (will get alpha connection)
                    # Find the location of the alpha position and the position that cyclizes with alpha
                    alpha_index = monomer_start_idx_atom[monomer_indices[index]] + site_positions[7][mons[index].type]
                    otherIndex = monomer_start_idx_atom[monomer_indices[int(not index)]] + \
                                 site_positions[other_site[tuple(sorted(bond))]][mons[int(not index)].type]

                    bonds.append(f'M  V30 {bond_line_num} 1 {alpha_index} {otherIndex} \n')
                    bond_line_num += 1

        # All kinds of fun things need to happen for the B1 bond --
        # 1 ) Disconnect the original 1 -> A bond that existed from the not beta monomer
        # 2 ) Convert the new primary alcohol to an aldehyde
        if sorted(bond) == [1, 8]:
            index_for_one = int(not beta[tuple(bond)])
            # Convert the alpha alcohol on one's tail to an aldehyde
            alpha_index = monomer_start_idx_atom[monomer_indices[index_for_one]
                                                 ] + site_positions[alpha][mons[index_for_one].type]

            # Temporarily join the bonds so that we can find the string
            temp = ''.join(bonds)
            matches = re.findall(f'M {2}V30 [0-9]+ 1 {alpha_index} [0-9]+', temp)

            # Find the bond connecting the alpha to the alcohol
            others = []
            for possibility in matches:
                bound_atoms = re.split(' +', possibility)[4:]
                others.extend([int(x) for x in bound_atoms if int(x) != alpha_index])

            # The oxygen atom should have the greatest index of the atoms bound to the alpha position because it
            #     was added last
            oxygen_atom_index = max(others)
            bonds = re.sub(f'1 {alpha_index} {oxygen_atom_index}',
                           f'2 {alpha_index} {oxygen_atom_index}', temp).splitlines(keepends=True)

            # Find where the index for the bond is and remove it from the array
            alpha_ring_bond_index = monomer_start_idx_bond[monomer_indices[index_for_one]
                                                           ] + alpha_ring_location - removed[
                                        BONDS]
            del (bonds[alpha_ring_bond_index])
            removed[BONDS] += 1

    mol_bond_blocks = ''.join(bonds)
    mol_atom_blocks = ''.join(atoms)

    mol_atom_blocks += 'M  V30 END ATOM \n'
    mol_bond_blocks += 'M  V30 END BOND \n'
    counts = f'M  V30 COUNTS {atom_line_num - 1 - removed[ATOMS]} {bond_line_num - 1 - removed[BONDS]} 0 0 0\n'
    mol_str += counts + mol_atom_blocks + mol_bond_blocks + 'M  V30 END CTAB\nM  END'

    return mol_str


def generate_psfgen(adj, monomers, fname="psfgen.tcl", segname="L", toppar_directory="toppar/"):
    """
    This takes a computed adjacency matrix and monomer list and writes out a script to generate a psf file of the
    associated structure, suitable for feeding into the LigninBuilder plugin of VMD
    (https://github.com/jvermaas/LigninBuilder).

    :param adj: Adjacency matrix generated by the kinetic Monte Carlo process
    :param monomers: Monomer list from the kinetic Monte Carlo process
    :param fname: desired output filename
    :param segname: desired output segment name for the generated lignin
    :param toppar_directory: location where the topology files top_lignin.top and top_all36_cgenff.rtf are expected
    :return:
    """
    adj = adj.copy()
    resnames = {0: G, 1: S, 2: C}
    fout = open(fname, "w")
    fout.write("package require psfgen\ntopology %stop_all36_cgenff.rtf\ntopology %stop_lignin.top\n" % (
        toppar_directory, toppar_directory))
    fout.write("segment %s {\n" % segname)
    for monomer in monomers:
        resid = monomer.identity + 1
        res_name = resnames[monomer.type]
        fout.write("\tresidue %d %s\n" % (resid, res_name))
    fout.write("}\n")
    # Since B-1 linkages actually involve three monomers, we signal that the previous beta-O-4/B-1 linkage required
    #     for B-1 is broken by flipping the sign.
    for row in (adj == 1).nonzero()[0]:
        col = (adj.getrow(row) == 8).nonzero()[1]
        if len(col):
            col = col[0]
            adj[(row, col)] *= -1
    for key in adj.keys():
        # Each linkage shows up as two non-zero values in the adjacency matrix. We only need one.
        if key[0] > key[1]:
            continue
        alt_key = (key[1], key[0])
        mono1 = int(adj[key])
        mono2 = int(adj[alt_key])
        if mono1 == 8 and mono2 == 4:  # Beta-O-4 linkage.
            fout.write("patch BO4 %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 4 and mono2 == 8:  # Reverse beta-O-4 linkage.
            fout.write("patch BO4 %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == 8 and mono2 == 5 and monomers[key[1]].type == 0:  # B5G linkage.
            fout.write("patch B5G %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 5 and mono2 == 8 and monomers[key[0]].type == 0:  # Reverse B5G linkage.
            fout.write("patch B5G %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == 8 and mono2 == 5 and monomers[key[1]].type == 2:  # B5C linkage.
            fout.write("patch B5C %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 5 and mono2 == 8 and monomers[key[0]].type == 2:  # Reverse B5C linkage.
            fout.write("patch B5C %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == 5 and mono2 == 5:  # 55 linkage
            fout.write("patch 55 %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 7 and mono2 == 4:  # alpha-O-4 linkage.
            fout.write("patch AO4 %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 4 and mono2 == 7:  # Reverse alpha-O-4 linkage.
            fout.write("patch AO4 %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == 4 and mono2 == 5:  # 4O5 linkage.
            fout.write("patch 4O5 %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 5 and mono2 == 4:  # Reverse 4O5 linkage.
            fout.write("patch 4O5 %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == 8 and mono2 == 1:  # Beta-1 linkage.
            fout.write("patch B1 %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        elif mono1 == 1 and mono2 == 8:  # Reverse beta-1 linkage.
            fout.write("patch B1 %s:%d %s:%d\n" % (segname, key[1] + 1, segname, key[0] + 1))
        elif mono1 == -8 and mono2 == 4:  # Beta-1 linkage remnant
            fout.write("patch O4AL %s:%d\n" % (segname, key[1] + 1))
        elif mono2 == -8 and mono1 == 4:  # Reverse beta-1 remnant.
            fout.write("patch O4AL %s:%d\n" % (segname, key[0] + 1))
        elif mono1 == -8 and mono2 == 1:  # Beta-1 linkage remnant (C1 variant)
            fout.write("patch C1AL %s:%d\n" % (segname, key[1] + 1))
        elif mono2 == -8 and mono1 == 1:  # Reverse beta-1 remnant. (C1 variant)
            fout.write("patch C1AL %s:%d\n" % (segname, key[0] + 1))
        elif mono1 == 8 and mono2 == 8:  # beta-beta linkage.
            fout.write("patch BB %s:%d %s:%d\n" % (segname, key[0] + 1, segname, key[1] + 1))
        else:
            print("This should never have happened! Abort!")
            print(key, mono1, mono2)
            print(monomers[key[0]].type, monomers[key[1]].type)
            raise ValueError
    fout.write("regenerate angles dihedrals\nwritepsf %s.psf" % segname)
    fout.close()


def mol_to_svg(mol, mol_size=(450, 150), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except KeyError:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(mol_size[0], mol_size[1])

    # opts = drawer.drawOptions()
    # for i in range(mc.GetNumAtoms()):
    #    opts.atomLabels[i] = mc.GetAtomWithIdx(i).GetSymbol()+str(i+1)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def generate_graph_representation(adj, node_list):
    """
    Creates a directed graph in NetworkX for a simplified representation of the lignin topology.
    In general, the graph edges point *from* interactions at C4 *to* their respective linkage sites.
    In exceptional cases (5-5, B-B linkages), both edges are created.

    Nodes express their monomer identity in the "t" field, and the edges are keyed according to linkage in the
    "l" field.

    An example script using matplotlib to draw the representation:

    import matplotlib.pyplot as plt
    import networkx as nx
    dir_graph = generateGraphRepresentation(l['adjacency_matrix'], l['monomers'])
    pos = nx.nx_agraph.graphviz_layout(dir_graph)
    nodes = dir_graph.nodes()
    nodecolors = [dir_graph.node[n]['t'] for n in nodes]
    edges = dir_graph.edges()
    edgecolors = [dir_graph[u][v]['l'] for u,v in edges]
    edgemap = plt.get_cmap('Set1')
    nodemap = plt.get_cmap('tab10')
    nx.draw_networkx(dir_graph, pos=pos, edges=edges, node_color=nodecolors, edge_color=edgecolors,
            cmap=nodemap, vmin=0, vmax=2, edge_cmap=edgemap, edge_vmin=0, edge_vmax=8)
    plt.savefig("test.png")

    :param adj:
    :param node_list:
    :return:
    """

    dir_graph = nx.DiGraph()  # Directional graph so we can easily see a bit of structure within the graph.
    for i, mon in enumerate(node_list):
        dir_graph.add_node(i, t=mon.type)
    adj = adj.copy()
    # Since B-1 linkages actually involve three monomers, we signal that the beta-O-4 linkage required for B-1 is
    #     broken by flipping the sign.
    for row in (adj == 1).nonzero()[0]:
        col = (adj.getrow(row) == 8).nonzero()[1]
        if len(col):
            col = col[0]
            adj[(row, col)] *= -1
    for key in sp.tril(adj).todok().keys():
        alt_key = (key[1], key[0])
        mono1 = int(adj[key])
        mono2 = int(adj[alt_key])
        if mono1 == 8 and mono2 == 4:  # Beta-O-4 linkage.
            dir_graph.add_edge(key[1], key[0], length=0)
        elif mono1 == 4 and mono2 == 8:  # Reverse beta-O-4 linkage.
            dir_graph.add_edge(key[0], key[1], length=0)
        elif mono1 == 8 and mono2 == 5:  # B5 linkage.
            dir_graph.add_edge(key[1], key[0], length=1)
        elif mono1 == 5 and mono2 == 8:  # Reverse B5 linkage.
            dir_graph.add_edge(key[0], key[1], length=1)
        elif mono1 == 5 and mono2 == 5:  # 55 linkage
            dir_graph.add_edge(key[1], key[0], length=2)
            dir_graph.add_edge(key[0], key[1], length=2)
        elif mono1 == 7 and mono2 == 4:  # alpha-O-4 linkage.
            dir_graph.add_edge(key[1], key[0], length=3)
        elif mono1 == 4 and mono2 == 7:  # Reverse alpha-O-4 linkage.
            dir_graph.add_edge(key[0], key[1], length=3)
        elif mono1 == 4 and mono2 == 5:  # 4O5 linkage.
            dir_graph.add_edge(key[0], key[1], length=4)
        elif mono1 == 5 and mono2 == 4:  # Reverse 4O5 linkage.
            dir_graph.add_edge(key[1], key[0], length=4)
        elif mono1 == 8 and mono2 == 1:  # Beta-1 linkage.
            dir_graph.add_edge(key[1], key[0], length=5)
        elif mono1 == 1 and mono2 == 8:  # Reverse beta-1 linkage.
            dir_graph.add_edge(key[0], key[1], length=5)
        elif mono1 == -8 and mono2 == 4:  # Beta-1 linkage remnant
            dir_graph.add_edge(key[1], key[1], length=6)
        elif mono2 == -8 and mono1 == 4:  # Reverse beta-1 remnant.
            dir_graph.add_edge(key[0], key[0], length=6)
        elif mono1 == -8 and mono2 == 1:  # Beta-1 linkage remnant (C1 variant)
            dir_graph.add_edge(key[1], key[1], length=7)
        elif mono2 == -8 and mono1 == 1:  # Reverse beta-1 remnant. (C1 variant)
            dir_graph.add_edge(key[0], key[0], length=7)
        elif mono1 == 8 and mono2 == 8:  # beta-beta linkage.
            dir_graph.add_edge(key[0], key[1], length=8)
            dir_graph.add_edge(key[1], key[0], length=8)
        else:
            raise InvalidDataError(f"Error when generating graph representation, on key: {key}, mono1: {mono1}, "
                                   f"mono2: {mono2}")
    return dir_graph

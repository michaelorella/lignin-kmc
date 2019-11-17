# !/usr/bin/env python
# coding=utf-8

import re
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from ligninkmc.kmc_common import (G, S, C, S4, G4, G7, ATOMS, BONDS)
from common_wrangler.common import (InvalidDataError, create_out_fname, warning)

DrawingOptions.bondLineWidth = 1.2
S7 = 'S7'


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
                    alkene_bond_index = mono_start_idx_bond[mono_indices[i]
                                                            ] + alpha_beta_alkene_location - removed[BONDS]

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
            if mons[int(not beta[tuple(bond_loc)])].type != 2:
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
                    other_idx = mono_start_idx_atom[mono_indices[int(not i)]
                                                    ] + site_positions[other_site[bond_loc_tuple]
                                                                       ][mons[int(not i)].type]

                    bonds.append(f'M  V30 {bond_line_num} 1 {alpha_idx} {other_idx} \n')
                    bond_line_num += 1

        # All kinds of fun things need to happen for the B1 bond --
        # 1 ) Disconnect the original 1 -> A bond that existed from the not beta monomer
        # 2 ) Convert the new primary alcohol to an aldehyde
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

            # TODO: Fix the section below--it does not work; for now, do not allow this functionality
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
                raise InvalidDataError("This program cannot currently generate a molecule with this beta-1 bond. "
                                       "Sorry!")

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


def gen_psfgen(orig_adj, monomers, fname="psfgen.tcl", segname="L", toppar_dir="toppar/", out_dir=None):
    """
    This takes a computed adjacency matrix and monomer list and writes out a script to generate a psf file of the
    associated structure, suitable for feeding into the LigninBuilder plugin of VMD
    (https://github.com/jvermaas/LigninBuilder).

    :param orig_adj: Adjacency matrix generated by the kinetic Monte Carlo process
    :param monomers: Monomer list from the kinetic Monte Carlo process
    :param fname: desired output filename
    :param segname: desired output segment name for the generated lignin
    :param toppar_dir: location where the topology files top_lignin.top and top_all36_cgenff.rtf are expected
    :param out_dir: subdirectory name where VMD should look for the toppar files
    :return:
    """
    adj = orig_adj.copy()
    resnames = {0: 'G', 1: 'S', 2: 'C'}
    f_out = create_out_fname(fname, base_dir=out_dir)
    # add a mac/linux dir separator if there isn't already a directory separator, and if there is to be a subdirectory
    #   (not None or "")
    if toppar_dir and len(toppar_dir) > 0:
        if toppar_dir[-1] != '/' and (toppar_dir[-1] != '\\'):
            toppar_dir += "/"
    else:
        toppar_dir = ""
    with open(f_out, "w") as f:
        print(f"Writing psfgen {f_out}")
        f.write(f"package require psfgen\n"
                f"topology {toppar_dir}{'top_all36_cgenff.rtf'}\n" 
                f"topology {toppar_dir}{'top_lignin.top'}\n"
                f"segment {segname} {{\n")
        for monomer in monomers:
            resid = monomer.identity + 1
            res_name = resnames[monomer.type]
            f.write(f"    residue {resid} {res_name}\n")
            # print(monomer)
        f.write(f"}}\n")

        # # step through in a consistent order; may be a more elegant way, but this works
        # adj_dict = dict(adj)
        # adj_keys = list(adj_dict.keys())
        # adj_keys.sort()
        # new_adj_dict = OrderedDict()
        # for adj_key in adj_keys:
        #     val = adj_dict[adj_key]
        #     # Since B-1 linkages involve three monomers, signal that the previous beta-O-4/B-1 linkage required
        #     #     for B-1 is broken by flipping the sign.
        #     if val == 1:
        #         val *= -1
        #     new_adj_dict[adj_key] = val
        # print("finished keys")
        #
        # # print("yo yo")
        # Since B-1 linkages actually involve three monomers, we signal that the previous beta-O-4/B-1 linkage required
        #     for B-1 is broken by flipping the sign.
        for row in (adj == 1).nonzero()[0]:
            col = (adj.getrow(row) == 8).nonzero()[1]
            if len(col):
                col = col[0]
                adj[(row, col)] *= -1
        for bond_matrix_tuple in adj.keys():
            # The adjacency matrix keys are always a tuple (of 2, row & col); sometimes they are equal to each other
            #    (e.g. oxidation)
            if bond_matrix_tuple[0] > bond_matrix_tuple[1]:
                continue
            psf_patch_resid1 = bond_matrix_tuple[0] + 1
            psf_patch_resid2 = bond_matrix_tuple[1] + 1
            flipped_bond_matrix_tuple = (bond_matrix_tuple[1], bond_matrix_tuple[0])
            bond_loc1 = int(adj[bond_matrix_tuple])
            bond_loc2 = int(adj[flipped_bond_matrix_tuple])
            if bond_loc1 == 8 and bond_loc2 == 4:  # Beta-O-4 linkage
                write_patch(f, "BO4", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 4 and bond_loc2 == 8:  # Reverse beta-O-4 linkage.
                write_patch(f, "BO4", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 5 and monomers[bond_matrix_tuple[1]].type == 0:  # B5G linkage
                write_patch(f, "B5G", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 8 and monomers[bond_matrix_tuple[0]].type == 0:  # Reverse B5G linkage
                write_patch(f, "B5G", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 5 and monomers[bond_matrix_tuple[1]].type == 2:  # B5C linkage
                write_patch(f, "B5C", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 8 and monomers[bond_matrix_tuple[0]].type == 2:  # Reverse B5C linkage
                write_patch(f, "B5C", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 5 and bond_loc2 == 5:  # 55 linkage
                write_patch(f, "B5C", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 7 and bond_loc2 == 4:  # alpha-O-4 linkage
                write_patch(f, "AO4", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 4 and bond_loc2 == 7:  # Reverse alpha-O-4 linkage
                write_patch(f, "AO4", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 4 and bond_loc2 == 5:  # 4O5 linkage
                write_patch(f, "4O4", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 5 and bond_loc2 == 4:  # Reverse 4O5 linkage
                write_patch(f, "4O4", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 1:  # Beta-1 linkage
                write_patch(f, "B1", segname, psf_patch_resid1, psf_patch_resid2)
            elif bond_loc1 == 1 and bond_loc2 == 8:  # Reverse beta-1 linkage
                write_patch(f, "B1", segname, psf_patch_resid2, psf_patch_resid1)
            elif bond_loc1 == -8 and bond_loc2 == 4:  # Beta-1 linkage remnant
                write_patch(f, "O4AL", segname, psf_patch_resid2)
            elif bond_loc2 == -8 and bond_loc1 == 4:  # Reverse beta-1 remnant
                write_patch(f, "O4AL", segname, psf_patch_resid1)
            elif bond_loc1 == -8 and bond_loc2 == 1:  # Beta-1 linkage remnant (C1 variant)
                write_patch(f, "C1AL", segname, psf_patch_resid2)
            elif bond_loc2 == -8 and bond_loc1 == 1:  # Reverse beta-1 remnant (C1 variant)
                write_patch(f, "C1AL", segname, psf_patch_resid1)
            elif bond_loc1 == 8 and bond_loc2 == 8:  # beta-beta linkage
                write_patch(f, "BB", segname, psf_patch_resid1, psf_patch_resid2)
            else:
                raise InvalidDataError(f"Encountered unexpected linkage: adj_matrix loc: {bond_matrix_tuple}, "
                                       f"bond locations: {bond_loc1} and {bond_loc2}, monomer types: "
                                       f"{monomers[bond_matrix_tuple[0]].type} and "
                                       f"{monomers[bond_matrix_tuple[1]].type}")
        f.write(f"regenerate angles dihedrals\nwritepsf {segname}.psf\n")

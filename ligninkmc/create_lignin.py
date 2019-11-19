#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launches steps required to build lignin
"""
import argparse
import os
import sys
import numpy as np
from scipy import triu
from scipy.sparse import dok_matrix
from collections import (defaultdict, OrderedDict)
from configparser import ConfigParser
from rdkit.Chem import (MolToSmiles, MolFromMolBlock, AddHs)
from rdkit.Chem.AllChem import (Compute2DCoords, EmbedMolecule, ETKDG)
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem.rdmolfiles import (MolToPDBFile, SDWriter)
from rdkit.Chem.rdMolInterchange import MolToJSON
from common_wrangler.common import (warning, process_cfg, MAIN_SEC, GOOD_RET, INPUT_ERROR, KB, H,
                                    KCAL_MOL_TO_J_PART, InvalidDataError, INVALID_DATA, OUT_DIR, make_dir,
                                    create_out_fname, str_to_file)
from ligninkmc import __version__
from ligninkmc.kmc_functions import run_kmc
from ligninkmc.visualization import (generate_mol, gen_psfgen)
from ligninkmc.kmc_common import (Event, Monomer, E_A_KCAL_MOL, E_A_J_PART, TEMP, INI_MONOS, MAX_MONOS, SIM_TIME,
                                  AFFECTED, GROW, DEF_E_A_KCAL_MOL, OX, MONOMER, OLIGOMER, LIGNIN_SUBUNITS,
                                  SG_RATIO, ADJ_MATRIX, RANDOM_SEED, AO4, B1, B1_ALT, B5, BB, BO4, C5C5, C5O4, S, G,
                                  CHAIN_LEN, BONDS, RCF_YIELDS, RCF_BONDS, MAX_NUM_DECIMAL, round_sig_figs,
                                  MONO_LIST, CHAIN_MONOS, CHAIN_BRANCHES, CHAIN_BRANCH_COEFF,
                                  RCF_MONOS, RCF_BRANCHES, RCF_BRANCH_COEFF)

# Config keys #
CONFIG_KEY = 'config_key'
OUT_FORMAT_LIST = 'output_format_list'
BASENAME = 'outfile_basename'
IMAGE_SIZE = 'image_size'
SAVE_JSON = 'json'
SAVE_PDB = 'pdb'
SAVE_PNG = 'png'
SAVE_SDF = 'sdf'
SAVE_SMI = 'smi'
SAVE_SVG = 'svg'
SAVE_TCL = 'tcl'
OUT_TYPE_LIST = [SAVE_JSON, SAVE_PDB, SAVE_PNG, SAVE_SDF, SAVE_SMI, SAVE_SVG, SAVE_TCL]
OUT_TYPE_STR = "', '".join(OUT_TYPE_LIST)
SAVE_FILES = 'save_files_boolean'


# Defaults #
DEF_TEMP = 298.15  # K
DEF_MAX_MONOS = 10  # number of monomers
DEF_SIM_TIME = 1  # simulation time in seconds
DEF_SG = 1
DEF_INI_MONOS = 2
DEF_ADD_RATE = 1e4
DEF_IMAGE_SIZE = (1200, 300)
DEF_BASENAME = 'lignin-kmc-out'

DEF_VAL = 'default_value'
DEF_CFG_VALS = {OUT_DIR: None, OUT_FORMAT_LIST: None, INI_MONOS: DEF_INI_MONOS, SIM_TIME: DEF_SIM_TIME,
                MAX_MONOS: DEF_MAX_MONOS, BASENAME: DEF_BASENAME, IMAGE_SIZE: DEF_IMAGE_SIZE,
                SG_RATIO: DEF_SG, TEMP: DEF_TEMP, RANDOM_SEED: None, E_A_KCAL_MOL: DEF_E_A_KCAL_MOL, E_A_J_PART: None,
                SAVE_FILES: False, SAVE_JSON: False, SAVE_PDB: False, SAVE_PNG: False, SAVE_SDF: False,
                SAVE_SMI: False, SAVE_SVG: False, SAVE_TCL: False,
                }

REQ_KEYS = {}

OPENING_MSG = f"Running Lignin-KMC version {__version__}. " \
              f"Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534\n"


################################################################################
# ANALYSIS CODE
################################################################################


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
    sizes = {}
    for fragment in frags:
        length = len(fragment)
        for node in fragment:
            sizes[node] = length
    return sizes


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


def adj_analysis_to_stdout(adj_results):
    """
    Describe the meaning of the summary dictionary
    :param adj_results: a dictionary from analyze_adj_matrix
    :return: n/a: prints to stdout
    """
    chain_len_results = adj_results[CHAIN_LEN]
    num_monos_created = sum(adj_results[CHAIN_MONOS].values())

    print(f"Lignin KMC created {num_monos_created} monomers, which formed:")
    print_olig_distribution(chain_len_results, adj_results[CHAIN_BRANCH_COEFF])

    lignin_bonds = adj_results[BONDS]
    print(f"composed of the following bond types and number:")
    print_bond_type_num(lignin_bonds)

    print("\nBreaking C-O bonds to simulate RCF results in:")
    print_olig_distribution(adj_results[RCF_YIELDS], adj_results[RCF_BRANCH_COEFF])

    print(f"with the following remaining bond types and number:")
    print_bond_type_num(adj_results[RCF_BONDS])


def print_bond_type_num(lignin_bonds):
    bond_summary = ""
    for bond_type, bond_num in lignin_bonds.items():
        bond_summary += f"   {bond_type.upper():>4}: {bond_num:4}"
    print(bond_summary)


def print_olig_distribution(chain_len_results, coeff):
    for olig_len, olig_num in chain_len_results.items():
        if olig_len == 1:
            print(f"{olig_num:>8} monomer(s) (chain length 1)")
        elif olig_len == 2:
            print(f"{olig_num:>8} dimer(s) (chain length 2)")
        elif olig_len == 3:
            print(f"{olig_num:>8} trimer(s) (chain length 3)")
        else:
            print(f"{olig_num:>8} oligomer(s) of chain length {olig_len}, with branching coefficient "
                  f"{round(coeff[olig_len], 3)}")


def degree(adj):
    """
    Determines the degree for each monomer within the polymer chain. The "degree" concept in graph theory
    is the number of edges connected to a node. In the context of lignin, that is simply the number of
    connected residues to a specific residue, and can be used to determine derived properties like the
    branching coefficient.
    :param adj: scipy dok_matrix   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: The degree for each monomer as a numpy array.
    """
    return np.bincount(adj.nonzero()[0])


def overall_branching_coefficient(adj):
    """
    Based on the definition in Dellon et al. (10.1021/acs.energyfuels.7b01150), this is the number of
       branched oligomers divided by the total number of monomers.
    This value is indifferent to the number of fragments in the output.

    :param adj: dok_matrix, the adjacency matrix for the lignin polymer that has been simulated
    :return: The branching coefficient that corresponds to the adjacency matrix
    """
    degrees = degree(adj)
    if len(degrees) == 0:
        return 0
    else:
        return np.sum(degrees >= 3) / len(degrees)


def get_bond_type_v_time_dict(adj_list, sum_len_larger_than=None):
    """
    given a list of adjs (one per timestep), flip nesting so have dictionaries of lists of type val vs. time
    for graphing, also created a 10+ list
    :param adj_list: list of adj dok_matrices
    :param sum_len_larger_than: None or an integer; if an integer, make a val_list that sums all lens >= that value
    :return: two dict of dicts
    """
    bond_type_dict = defaultdict(list)
    # a little more work for olig_len_dict, since each timestep does not contain all possible keys
    olig_len_dict = defaultdict(list)
    frag_count_dict_list = []  # first make list of dicts to get max bond_length
    for adj in adj_list:  # loop over each timestep
        # this is keys = timestep  values
        count_bonds_list = count_bonds(adj)
        for bond_type in count_bonds_list:
            bond_type_dict[bond_type].append(count_bonds_list[bond_type])
        olig_yield_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adj)
        frag_count_dict_list.append(olig_monos_dict)
    # since breaking bonds is not allowed, the longest oligomer will be from the last step; ordered, so last len
    max_olig_len = list(frag_count_dict_list[-1].keys())[-1]
    # can now get the dict of lists from list of dicts
    for frag_count_list in frag_count_dict_list:
        for olig_len in range(1, max_olig_len + 1):
            olig_len_dict[olig_len].append(frag_count_list.get(olig_len, 0))
    # now make a list of all values greater than a number, if given
    len_plus_list = None
    if sum_len_larger_than:
        if sum_len_larger_than in olig_len_dict.keys():
            len_plus_list = olig_len_dict[10].copy()
        for olig_len, val_list in olig_len_dict.items():
            if olig_len > sum_len_larger_than:
                len_plus_list = np.add(len_plus_list, val_list)
    return bond_type_dict, olig_len_dict, len_plus_list


def read_cfg(f_loc, cfg_proc=process_cfg):
    """
    Reads the given configuration file, returning a dict with the converted values supplemented by default values.

    :param f_loc: The location of the file to read.
    :param cfg_proc: The processor to use for the raw configuration values.  Uses default values when the raw
        value is missing.
    :return: A dict of the processed configuration file's data.
    """
    config = ConfigParser()
    good_files = config.read(f_loc)

    if not good_files:
        raise IOError(f"Could not find specified configuration file: {f_loc}")

    main_proc = cfg_proc(dict(config.items(MAIN_SEC)), DEF_CFG_VALS, REQ_KEYS)

    return main_proc


def parse_cmdline(argv=None):
    """
    Returns the parsed argument list and return code.
    :param argv: A list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description=f"Create lignin chain(s) composed of 'S' ({S}) and/or 'G' ({G}) "
                                                 f"monolignols, as described in:\n  Orella, M., "
                                                 'Gani, T. Z. H., Vermaas, J. V., Stone, M. L., Anderson, E. M., '
                                                 'Beckham, G. T., \n  Brushett, Fikile R., Roman-Leshkov, Y. (2019). '
                                                 'Lignin-KMC: A Toolkit for Simulating Lignin Biosynthesis.\n  '
                                                 'ACS Sustainable Chemistry & Engineering. '
                                                 'https://doi.org/10.1021/acssuschemeng.9b03534. C-Lignin can be \n  '
                                                 'modeled with the functions in this package, as shown in ipynb '
                                                 'examples in our project package on github \n  '
                                                 '(https://github.com/michaelorella/lignin-kmc/), but not currently '
                                                 'from the command line. If this \n  functionality is desired, please '
                                                 'start a new issue on the github.\n\n  '
                                                 'By default, the activation energies from this reference will be '
                                                 'used, as specified in Tables S1 and S2.\n  Alternately, the user '
                                                 f"may specify values, which should be specified as a dict of dict "
                                                 f"of dicts in a \n  specified configuration file (specified with '-c')"
                                                 f" using the '{E_A_KCAL_MOL}' or '{E_A_J_PART}'\n  parameters with "
                                                 f"corresponding units (kcal/mol or joules/particle, respectively), in "
                                                 f"a configuration file \n  (see '-c'). The format is (bond_type: "
                                                 f"monomer(s) involved: units involved: ea_vals), for example:\n      "
                                                 f"ea_dict = {{{OX}: {{0: {{{MONOMER}: 0.9, {OLIGOMER}: 6.3}}, "
                                                 f"1: ""{{{MONOMER}: 0.6, {OLIGOMER}: " f"2.2}}}}, ...}}\n  "
                                                 f"where 0: {LIGNIN_SUBUNITS[0]}, 1: {LIGNIN_SUBUNITS[1]}, "
                                                 f"2: {LIGNIN_SUBUNITS[2]}. The default output is a SMILES string "
                                                 f"printed to standard out.\n\n  All command-line options may "
                                                 f"alternatively be specified in a configuration file. Command-line "
                                                 f"(non-default) \n  selections will override configuration file "
                                                 f"specifications.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", help="The location of the configuration file in the 'ini' format. This file "
                                               "can be used to \noverwrite default values such as for energies.",
                        default=None, type=read_cfg)
    parser.add_argument("-d", "--out_dir", help="The directory where output files will be saved. The default is "
                                                "the current directory.", default=DEF_CFG_VALS[OUT_DIR])
    parser.add_argument("-f", "--output_format_list", help="The type(s) of output format to be saved. Provide as a "
                                                           "space- or comma-separated \nlist. The currently supported "
                                                           f"types are: '{OUT_TYPE_STR}'. \nThe '{SAVE_JSON}' "
                                                           f"option will save a json format of RDKit's 'mol' "
                                                           f"(molecule) object. The '{SAVE_TCL}' option will create "
                                                           f"a file for use with VMD to generate a psf file, as "
                                                           f"further described in Lignin-Builder, "
                                                           f"https://github.com/jvermaas/LigninBuilder, "
                                                           f"https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.8b05665"
                                                           f". \n\nA base name for the saved "
                                                           f"files can be provided with the '-o' option. Otherwise, "
                                                           f"the base \nname will be '{DEF_BASENAME}'.",
                        default=DEF_CFG_VALS[OUT_FORMAT_LIST])
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_CFG_VALS[INI_MONOS])
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME} s.", default=DEF_SIM_TIME)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-o", "--output_basename", help="The basename for output file(s). If an extension is provided, "
                                                        "it will determine \nthe type of output. Multiple output "
                                                        "formats can be selected with the '-f' option. \nThe default "
                                                        "format type is '.smi' (SMILES). Currently supported output"
                                                        f"types are: \n'{OUT_TYPE_STR}'.", default=DEF_BASENAME)
    parser.add_argument("-r", "--random_seed", help="A non-zero integer to be used as a seed value for testing.",
                        default=DEF_CFG_VALS[RANDOM_SEED])
    parser.add_argument("-s", "--image_size", help=f"The output size of svg or png files in pixels (provide two "
                                                   f"integers). The default \nis {DEF_IMAGE_SIZE} pixels.",
                        default=DEF_IMAGE_SIZE)
    parser.add_argument("-sg", "--sg_ratio", help=f"The S:G (guaiacol:syringyl) ratio. "
                                                  f"The default is {DEF_SG}.", default=DEF_SG)
    parser.add_argument("-t", "--temperature_in_k", help=f"The temperature (in K) at which to model lignin "
                                                         f"biosynthesis. The default is {DEF_TEMP} K. \nCAUTION: the "
                                                         f"default energy barriers were calculated at {DEF_TEMP} K. "
                                                         f"If they are used, \nthe resulting calculated reaction rates "
                                                         f"may not be valid.",
                        default=DEF_TEMP)

    args = None
    try:
        args = parser.parse_args(argv)
        # dict below to map config input and defaults to command-line input
        conf_arg_dict = {OUT_DIR: args.out_dir,
                         OUT_FORMAT_LIST: args.output_format_list,
                         INI_MONOS: args.initial_num_monomers,
                         SIM_TIME: args.length_simulation,
                         MAX_MONOS: args.max_num_monomers,
                         BASENAME: args.output_basename,
                         IMAGE_SIZE: args.image_size,
                         SG_RATIO: args.sg_ratio,
                         TEMP: args.temperature_in_k,
                         RANDOM_SEED: args.random_seed,
                         }
        if args.config is None:
            args.config = DEF_CFG_VALS.copy()
        # Now overwrite any config values with command-line arguments, only if those values are not the default
        for config_key, arg_val in conf_arg_dict.items():
            if not (arg_val == DEF_CFG_VALS[config_key]):
                args.config[config_key] = arg_val

    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def calc_rates(temp, ea_j_part_dict=None, ea_kcal_mol_dict=None):
    """
    Uses temperature and provided activation energy to calculate rates using the Eyring equation
    dictionary formats: dict = {rxn_type: {substrate(s): {sub_lengths (e.g. (monomer, monomer)): value, ...}, ...}, ...}

    Only ea_j_part_dict or ea_kcal_mol_dict are needed; if both are provided, only ea_j_part_dict will be used

    :param temp: float, temperature in K
    :param ea_j_part_dict: dictionary of activation energies in Joule/particle, in format noted above
    :param ea_kcal_mol_dict: dictionary of activation energies in Joule/particle, in format noted above
    :return: rxn_rates: dict of reaction rates units of 1/s
    """
    # want activation energies in J/particle; user can provide them in those units or in kcal/mol
    if ea_j_part_dict is None:
        ea_j_part_dict = {
            rxn_type: {substrate: {sub_len: ea_kcal_mol_dict[rxn_type][substrate][sub_len] * KCAL_MOL_TO_J_PART
                                   for sub_len in ea_kcal_mol_dict[rxn_type][substrate]}
                       for substrate in ea_kcal_mol_dict[rxn_type]} for rxn_type in ea_kcal_mol_dict}
    # TODO: check if need to change to solution state...
    rxn_rates = {}
    for rxn_type in ea_j_part_dict:
        rxn_rates[rxn_type] = {}
        for substrate in ea_j_part_dict[rxn_type]:
            rxn_rates[rxn_type][substrate] = {}
            for substrate_type in ea_j_part_dict[rxn_type][substrate]:
                # rounding to reduce difference due to solely to platform running package
                rate = KB * temp / H * np.exp(-ea_j_part_dict[rxn_type][substrate][substrate_type] / KB / temp)
                rxn_rates[rxn_type][substrate][substrate_type] = round_sig_figs(rate, sig_figs=15)
    return rxn_rates


def create_initial_monomers(pct_s, monomer_draw):
    """
    Make a monomer list (length of monomer_draw) based on the types determined by the monomer_draw list and pct_s
    :param pct_s: float ([0:1]), fraction of  monomers that should be type "1" ("S")
    :param monomer_draw: a list of floats ([0:1)) to determine if the monomer should be type "0" ("G", val < pct_s) or
                         "1" ("S", otherwise)
    :return: list of Monomer objects of specified type
    """
    # TODO: If want more than 2 monomer options, need to change logic; that will require an overhaul, since
    #       sg_ratio is often used
    # if mon_choice < pct_s, make it an S; that is, the evaluation comes back True (=1='S');
    #     otherwise, get False = 0 = 'G'. Since only two options (True/False) only works for 2 monomers
    return [Monomer(int(mono_type_draw < pct_s), i) for i, mono_type_draw in enumerate(monomer_draw)]


def create_initial_events(initial_monomers, rxn_rates):
    """
    # Create event_dict that will oxidize every monomer
    :param initial_monomers: a list of Monomer objects
    :param rxn_rates: dict of dict of dicts of reaction rates in 1/s
    :return: a list of oxidation Event objects to initialize the state by allowing oxidation of every monomer
    """
    return [Event(OX, [mon.identity], rxn_rates[OX][mon.type][MONOMER]) for mon in initial_monomers]


def create_initial_state(initial_events, initial_monomers):
    return {i: {MONOMER: initial_monomers[i], AFFECTED: {initial_events[i]}} for i in range(len(initial_monomers))}


def produce_output(result, cfg):
    # Default out is SMILES
    block = generate_mol(result[ADJ_MATRIX], result[MONO_LIST])
    mol = MolFromMolBlock(block)
    smi_str = MolToSmiles(mol) + '\n'
    # if SMI is to be saved, don't output to stdout
    if cfg[SAVE_SMI]:
        fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=SAVE_SMI)
        str_to_file(smi_str, fname, print_info=True)
    else:
        print("\nSMILES representation: \n", MolToSmiles(mol), "\n")
    if cfg[SAVE_TCL]:
        # This is a separated from below as it does not require 2D or 3D coordinates
        fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=SAVE_TCL)
        gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=fname, segname="L",
                   toppar_dir='toppar', out_dir=cfg[OUT_DIR])
    if cfg[SAVE_PDB] or cfg[SAVE_JSON] or cfg[SAVE_PNG] or cfg[SAVE_SDF] or cfg[SAVE_SVG]:
        Compute2DCoords(mol)
        mol3d = None  # making IDE happy; no other purpose
        if cfg[SAVE_PDB] or cfg[SAVE_SDF]:
            mol3d = AddHs(mol)
            EmbedMolecule(mol3d, ETKDG())
        for save_type in [SAVE_PDB, SAVE_JSON, SAVE_PNG, SAVE_SDF, SAVE_SVG]:
            if cfg[save_type]:
                fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=save_type)
                if save_type == SAVE_PDB:
                    MolToPDBFile(mol3d, fname)
                elif save_type == SAVE_JSON:
                    json_str = MolToJSON(mol)
                    str_to_file(json_str + '\n', fname)
                elif save_type == SAVE_PNG or save_type == SAVE_SVG:
                    MolToFile(mol, fname, size=cfg[IMAGE_SIZE])
                elif save_type == SAVE_SDF:
                    writer = SDWriter(fname)
                    writer.write(mol3d)
                print(f"Wrote file: {fname}")


def validate_input(cfg):
    """
    Checking for errors at the beginning, so don't waste time starting calculations that will not be able to complete

    :param cfg: dict of configuration values
    :return: will raise an error if invalid data is encountered
    """
    if (cfg[E_A_KCAL_MOL] == DEF_E_A_KCAL_MOL) and cfg[TEMP] != DEF_TEMP:
        warning(f"Caution: the default energy barriers, which were calculated at {DEF_TEMP}, have been selected for "
                f"use at a different specified temperature ({cfg[TEMP]} K). The resulting calculated reaction "
                f"rates may not be valid.")

    # Don't use "if cfg[RANDOM_SEED]:", because that won't catch the user giving the value 0, which they might think
    #    would be a valid random seed, but won't work for this package because of later "if cfg[RANDOM_SEED]:" checks
    if cfg[RANDOM_SEED] is not None:
        try:
            # numpy seeds must be 0 and 2**32 - 1. Raise an error if the input cannot be converted to an int. Also raise
            #   an error for 0, since that will return False that a seed was provided in the logic in this package
            cfg[RANDOM_SEED] = int(cfg[RANDOM_SEED])
            if cfg[RANDOM_SEED] <= 0 or cfg[RANDOM_SEED] > (2**32 - 1):
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Invalid input provided for '{RANDOM_SEED}': '{cfg[RANDOM_SEED]}'. If you "
                                   f"would like to obtain consistent output by using a random seed, provide a "
                                   f"positive integer value no greater than 2**32 - 1.")

    for req_pos_num in [SG_RATIO, SIM_TIME]:
        try:
            cfg[req_pos_num] = float(cfg[req_pos_num])
            if cfg[req_pos_num] < 0:
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Found '{cfg[req_pos_num]}' input for '{req_pos_num}'. The {req_pos_num} must be "
                                   f"a positive number.")

    for num_monos in [INI_MONOS, MAX_MONOS]:
        try:
            cfg[num_monos] = int(cfg[num_monos])
            if cfg[num_monos] < 0:
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Found '{cfg[num_monos]}' input for '{num_monos}'. The {num_monos} must be a "
                                   f"positive integer.")

    try:
        # Will be a string to process unless it is the default
        if isinstance(cfg[IMAGE_SIZE], str):
            raw_vals = cfg[IMAGE_SIZE].replace(",", " ").replace("(", "").replace(")", "").split()
            if len(raw_vals) != 2:
                raise ValueError
            cfg[IMAGE_SIZE] = (int(raw_vals[0]), int(raw_vals[1]))
    except ValueError:
        raise InvalidDataError(f"Found '{cfg[req_pos_num]}' input for '{req_pos_num}'. The {req_pos_num} must be "
                               f"a positive number.")

    # Check for valid output requests
    check_if_files_to_be_saved(cfg)


def check_if_files_to_be_saved(cfg):
    """
    Evaluate input for requests to save output and check for valid specified locations
    :param cfg: dict of configuration values
    :return: if the cfg designs that files should be created, returns an updated cfg dict, and raises errors if
              invalid data in encountered
    """
    if cfg[OUT_FORMAT_LIST]:
        # remove any periods to aid comparison; might as well also change comma to space and then split on just space
        out_format_list = cfg[OUT_FORMAT_LIST].replace(".", " ").replace(",", " ")
        format_set = set(out_format_list.split())
    else:
        format_set = set()

    if cfg[BASENAME] and (cfg[BASENAME] != DEF_BASENAME):
        # If cfg[BASENAME] is not just the base name, make it so, saving a dir or ext in their spots
        out_path, base_name = os.path.split(cfg[BASENAME])
        if out_path and cfg[OUT_DIR]:
            cfg[OUT_DIR] = os.path.join(cfg[OUT_DIR], out_path)
        elif out_path:
            cfg[OUT_DIR] = out_path
        base, ext = os.path.splitext(base_name)
        cfg[BASENAME] = base
        format_set.add(ext.replace(".", ""))

    if len(format_set) > 0:
        for format_type in format_set:
            if format_type in OUT_TYPE_LIST:
                cfg[SAVE_FILES] = True
                cfg[format_type] = True
            else:
                raise InvalidDataError(f"Invalid extension provided: '{format_type}'. The currently supported types "
                                       f"are: '{OUT_TYPE_STR}'")

    # if out_dir does not already exist, recreate it, only if we will actually need it
    if cfg[SAVE_FILES] and cfg[OUT_DIR]:
        make_dir(cfg[OUT_DIR])


def main(argv=None):
    """
    Runs the main program.

    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    print(OPENING_MSG)
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    cfg = args.config

    try:
        # tests at the beginning to catch errors early
        validate_input(cfg)

        # need rates before we can start modeling reactions
        rxn_rates = calc_rates(cfg[TEMP], ea_j_part_dict=cfg[E_A_J_PART], ea_kcal_mol_dict=cfg[E_A_KCAL_MOL])

        # decide on initial monomers, based on given SG_RATIO
        pct_s = cfg[SG_RATIO] / (1 + cfg[SG_RATIO])
        ini_num_monos = cfg[INI_MONOS]
        if cfg[RANDOM_SEED]:
            np.random.seed(cfg[RANDOM_SEED])
            monomer_draw = np.around(np.random.rand(ini_num_monos), MAX_NUM_DECIMAL)
        else:
            monomer_draw = np.random.rand(ini_num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)

        # initial event must be oxidation to create reactive species; all monomers get a chance of being oxidized
        initial_events = create_initial_events(initial_monomers, rxn_rates)

        # After the initial_monomers and initial_events have been created, they are grouped into the initial state.
        initial_state = create_initial_state(initial_events, initial_monomers)
        initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE, bond=cfg[SG_RATIO]))

        # begin simulation
        result = run_kmc(rxn_rates, initial_state, initial_events, n_max=cfg[MAX_MONOS], t_max=cfg[SIM_TIME],
                         sg_ratio=cfg[SG_RATIO])
        # show results
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        adj_analysis_to_stdout(summary)

        # Outputs
        produce_output(result, cfg)

    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launches steps required to build lignin
Multiple output options, from tcl files to plots
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import (defaultdict)
from configparser import ConfigParser
from common_wrangler.common import (MAIN_SEC, GOOD_RET, INPUT_ERROR, KB, H, KCAL_MOL_TO_J_PART,
                                    INVALID_DATA, OUT_DIR, InvalidDataError, warning, process_cfg, make_dir,
                                    create_out_fname, str_to_file, round_sig_figs)
from rdkit.Chem import (MolToSmiles, MolFromMolBlock)
from rdkit.Chem.AllChem import (Compute2DCoords)
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem.rdMolInterchange import MolToJSON
from ligninkmc import __version__
from ligninkmc.kmc_common import (Event, Monomer, E_BARRIER_KCAL_MOL, E_BARRIER_J_PART, TEMP, INI_MONOS, MAX_MONOS,
                                  SIM_TIME, AFFECTED, GROW, DEF_E_BARRIER_KCAL_MOL, OX, MONOMER, OLIGOMER,
                                  LIGNIN_SUBUNITS, ADJ_MATRIX, RANDOM_SEED, S, G, CHAIN_LEN, BONDS, ADD_RATE,
                                  RCF_YIELDS, RCF_BONDS, MAX_NUM_DECIMAL, MONO_LIST, CHAIN_MONOS, CHAIN_BRANCH_COEFF,
                                  RCF_BRANCH_COEFF, CHAIN_ID, DEF_CHAIN_ID, PSF_FNAME, DEF_PSF_FNAME, DEF_TOPPAR,
                                  TOPPAR_DIR, MANUSCRIPT_RATES, DEF_RXN_RATES)
from ligninkmc.kmc_functions import (run_kmc, generate_mol, gen_psfgen, count_bonds,
                                     count_oligomer_yields, analyze_adj_matrix)


# Config keys #
CONFIG_KEY = 'config_key'
OUT_FORMAT_LIST = 'output_format_list'
BASENAME = 'outfile_basename'
IMAGE_SIZE = 'image_size'
SAVE_JSON = 'json'
SAVE_PNG = 'png'
SAVE_SMI = 'smi'
SAVE_SVG = 'svg'
SAVE_TCL = 'tcl'
OUT_TYPE_LIST = [SAVE_JSON, SAVE_PNG,  SAVE_SMI, SAVE_SVG, SAVE_TCL]
OUT_TYPE_STR = "', '".join(OUT_TYPE_LIST)
SAVE_FILES = 'save_files_boolean'
ENERGY_BARRIER_FLAG = 'energy_barriers_flag'
ADD_RATES = 'add_rates_list'
RXN_RATES = 'reaction_rates_at_298K'
SG_RATIOS = 'sg_ratio_list'
NUM_REPEATS = 'num_repeats'
DYNAMICS = 'dynamics_flag'

PLOT_COLORS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.6, 0), (0.6, 0, 0.6), (1, 0.549, 0),
               (0, 0.6, 0.6), (1, 0.8, 0), (0.6078, 0.2980, 0), (0.6, 0, 0), (0, 0, 0.6)]


# Defaults #
DEF_TEMP = 298.15  # K
DEF_MAX_MONOS = 10  # number of monomers
DEF_SIM_TIME = 3600  # simulation time in seconds
DEF_SG = 1
DEF_INI_MONOS = 2
# Estimated addition rate below is based on: https://www.pnas.org/content/early/2019/10/25/1904643116.abstract
#     p. 23121, col2, "0.1 fmol s^-1" for 100micro-m2 surface area; estimated area for lignin modeling is 100nm^2
#     thus 0.1 fmols/ second * 1.00E-15 mol/fmol * 6.022E+23 particles/mol  * 100 nm^2/100microm^2 = 6 monomers/s
#     as an upper limit--rounded down to 1.0 monomers/s--this is just an estimate
DEF_ADD_RATE = 1.0
DEF_IMAGE_SIZE = (1200, 300)
DEF_BASENAME = 'lignin-kmc-out'

DEF_VAL = 'default_value'
DEF_CFG_VALS = {OUT_DIR: None, OUT_FORMAT_LIST: None, ADD_RATES: [DEF_ADD_RATE], INI_MONOS: DEF_INI_MONOS,
                MAX_MONOS: DEF_MAX_MONOS, SIM_TIME: DEF_SIM_TIME, SG_RATIOS: [DEF_SG], TEMP: DEF_TEMP,
                RANDOM_SEED: None, BASENAME: DEF_BASENAME, IMAGE_SIZE: DEF_IMAGE_SIZE, DYNAMICS: False,
                E_BARRIER_KCAL_MOL: DEF_E_BARRIER_KCAL_MOL, E_BARRIER_J_PART: None, SAVE_FILES: False,
                SAVE_JSON: False, SAVE_PNG: False, SAVE_SMI: False, SAVE_SVG: False, SAVE_TCL: False,
                CHAIN_ID: DEF_CHAIN_ID, PSF_FNAME: DEF_PSF_FNAME, TOPPAR_DIR: DEF_TOPPAR,
                }

REQ_KEYS = {}

OPENING_MSG = f"Running Lignin-KMC version {__version__}. " \
              f"Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534\n"

# y_axis_val_dicts={'monomers': avg_num_monos, oligomers': avg_num_oligs}
# y_axis_std_dev_dicts={'monomers': std_dev_monos, oligomers': std_dev_oligs}
# y_val_key_list=['monomers', 'oligomers']
# x_axis_label='Time step'
# y_axis_label='Number'

# y_val_key_list=BOND_TYPE_LIST
# x_axis_label='SG Ratio'
# y_axis_label='Bond Type Yield (%)'


def plot_bond_error_bars(x_axis, y_axis_val_dicts, y_axis_std_dev_dicts, y_val_key_list, x_axis_label, y_axis_label,
                         plot_title, plot_fname):
    plt.figure(figsize=(3.5, 3.5))
    for y_idx, y_key in enumerate(y_val_key_list):
        plt.errorbar(x_axis, y_axis_val_dicts[y_key], yerr=y_axis_std_dev_dicts[y_key], linestyle='none', marker='.',
                     markersize=10, markerfacecolor=PLOT_COLORS[y_idx], markeredgecolor=PLOT_COLORS[y_idx],
                     label=y_key, capsize=3, ecolor=PLOT_COLORS[y_idx])

    if len(x_axis) > 1:
        plt.xscale('log')

    [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
    plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1, length=4)
    plt.ylabel(y_axis_label, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=14)
    plt.ylim([0.0, 1.0])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
    plt.title(plot_title)
    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)
    print(f"Wrote file: {plot_fname}")
    plt.close()


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
    # a little more work for olig_len_monos_dict, since each timestep does not contain all possible keys
    olig_len_monos_dict = defaultdict(list)
    olig_len_count_dict = defaultdict(list)
    olig_count_dict_list = []
    frag_count_dict_list = []  # first make list of dicts to get max bond_length
    for adj in adj_list:  # loop over each timestep
        # this is keys = timestep  values
        count_bonds_list = count_bonds(adj)
        for bond_type in count_bonds_list:
            bond_type_dict[bond_type].append(count_bonds_list[bond_type])
        olig_yield_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adj)
        olig_count_dict_list.append(olig_yield_dict)
        frag_count_dict_list.append(olig_monos_dict)
    # since breaking bonds is not allowed, the longest oligomer will be from the last step; ordered, so last len
    max_olig_len = list(frag_count_dict_list[-1].keys())[-1]
    # can now get the dict of lists from list of dicts
    for frag_count_list, olig_count_list in zip(frag_count_dict_list, olig_count_dict_list):
        for olig_len in range(1, max_olig_len + 1):
            olig_len_monos_dict[olig_len].append(frag_count_list.get(olig_len, 0))
            olig_len_count_dict[olig_len].append(olig_count_list.get(olig_len, 0))
    # now make a list of all values greater than a number, if given
    # first initialize as None so something can be returned, even if we are not summing over a particular number
    len_monos_plus_list = None
    len_count_plus_list = None
    if sum_len_larger_than:
        num_time_steps = len(adj_list)
        len_monos_plus_list = np.zeros(num_time_steps)
        len_count_plus_list = np.zeros(num_time_steps)
        # both dicts have same keys, so no worries
        for olig_len, val_list in olig_len_monos_dict.items():
            if olig_len >= sum_len_larger_than:
                len_monos_plus_list = np.add(len_monos_plus_list, val_list)
                len_count_plus_list = np.add(len_count_plus_list, olig_len_count_dict[olig_len])
    return bond_type_dict, olig_len_monos_dict, len_monos_plus_list, olig_len_count_dict, len_count_plus_list


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
                                                 f"Gani, T. Z. H., Vermaas, J. V., Stone, M. L., Anderson, E. M., "
                                                 f"Beckham, G. T., \n  Brushett, Fikile R., Roman-Leshkov, Y. (2019). "
                                                 f"Lignin-KMC: A Toolkit for Simulating Lignin Biosynthesis.\n  "
                                                 f"ACS Sustainable Chemistry & Engineering. "
                                                 f"https://doi.org/10.1021/acssuschemeng.9b03534. C-Lignin can be \n  "
                                                 f"modeled with the functions in this package, as shown in ipynb "
                                                 f"examples in our project package on github \n  "
                                                 f"(https://github.com/michaelorella/lignin-kmc/), but not currently "
                                                 f"from the command line. If this \n  functionality is desired, "
                                                 f"please start a new issue on the github.\n\n  By default, the Gibbs "
                                                 f"free energy barriers from this reference will be used, as specified "
                                                 f"in Tables S1 and S2.\n  Alternately, the user may specify values, "
                                                 f"which should be specified as a dict of dict of dicts in a \n  "
                                                 f"specified configuration file (specified with '-c') using the "
                                                 f"'{E_BARRIER_KCAL_MOL}' or '{E_BARRIER_J_PART}'\n  parameters with "
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
    parser.add_argument("-a", "--add_rates", help=f"A comma-separated list of the rates of monomer addition to the "
                                                  f"system (in monomers/second), \nto be used when the '{MAX_MONOS}' "
                                                  f"('-m' option) is larger than '{INI_MONOS}' \n('-i' option), thus "
                                                  f"specifying monomer addition. The simulation will end when either "
                                                  f"there \nare no more possible reactions (including monomer "
                                                  f"addition) or when the '{SIM_TIME}' \n('-l' option) is reached, "
                                                  f"whichever comes first. Note: if there are spaces in the list of "
                                                  f"\naddition rates, the list must be enclosed in quotes to be read "
                                                  f"as a single string. The \ndefault list contains the single "
                                                  f"addition rate of {DEF_ADD_RATE} monomers/s.",
                        default=[DEF_ADD_RATE])
    parser.add_argument("-c", "--config", help="The location of the configuration file in the 'ini' format. This file "
                                               "can be used to \noverwrite default values such as for energies.",
                        default=None, type=read_cfg)
    parser.add_argument("-d", "--out_dir", help="The directory where output files will be saved. The default is "
                                                "the current directory.", default=DEF_CFG_VALS[OUT_DIR])
    # Todo: add this functionality
    parser.add_argument("-dy", "--dynamics_flag", help=f"Select this option if dynamics (results per timestep) are "
                                                       f"requested. If chosen, plots of \nmonomers and oligomers vs "
                                                       f"timestep, and bond type percent vs timestep, will be saved. "
                                                       f"\nNote that this option significantly increases simulation "
                                                       f"time.", action="store_true")
    parser.add_argument("-e", "--energy_barriers_flag", help=f"By default, the reaction rates will be based on the "
                                                             f"energy barriers in kcal/mol to be used \nto calculate "
                                                             f"reaction rates at {DEF_TEMP} K. If this flag is used, "
                                                             f"the rates use to produce the \noriginal manuscript "
                                                             f"plots will be used (missing one term). Alternate sets "
                                                             f"of energy \nbarriers can be specified; see the main "
                                                             f"help message.", action="store_true")
    parser.add_argument("-f", "--output_format_list", help="The type(s) of output format to be saved. Provide as a "
                                                           "space- or comma-separated list. \nNote: if the list has "
                                                           "spaces, it must be enclosed in quotes, to be treated as "
                                                           "a single \nstring. The currently supported "
                                                           f"types are: '{OUT_TYPE_STR}'. \nThe '{SAVE_JSON}' "
                                                           f"option will save a json format of RDKit's 'mol' "
                                                           f"(molecule) object. The '{SAVE_TCL}' \noption will create "
                                                           f"a file for use with VMD to generate a psf file and 3D "
                                                           f"molecules, \nas described in LigninBuilder, "
                                                           f"https://github.com/jvermaas/LigninBuilder, \n"
                                                           f"https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.8b05665"
                                                           f". \nA base name for the saved "
                                                           f"files can be provided with the '-o' option. Otherwise, "
                                                           f"the \nbase name will be '{DEF_BASENAME}'.",
                        default=DEF_CFG_VALS[OUT_FORMAT_LIST])
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_CFG_VALS[INI_MONOS])
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME} s.", default=DEF_SIM_TIME)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-o", "--output_basename", help=f"The base name for output file(s). If an extension is "
                                                        f"provided, it will determine \nthe type of output. Currently "
                                                        f"supported output types are: \n'{OUT_TYPE_STR}'. Multiple "
                                                        f"output formats can be selected with the \n'-f' option. If "
                                                        f"the '-f' option is selected and no output base name "
                                                        f"provided, a default \nbase name of '{DEF_BASENAME}' will be "
                                                        f"used.", default=DEF_BASENAME)
    parser.add_argument("-r", "--random_seed", help="A positive integer to be used as a seed value for testing. The "
                                                    "default is not to use a \nseed, to allow pseudorandom lignin "
                                                    "creation.", default=DEF_CFG_VALS[RANDOM_SEED])
    parser.add_argument("-s", "--image_size", help=f"The output size of svg or png files in pixels. The default size "
                                                   f"is {DEF_IMAGE_SIZE} pixels. \nTo use a different size, provide "
                                                   f"two integers, separated by a space or a comma. \nNote: if the "
                                                   f"list of two numbers has any spaces in it, it must be enclosed "
                                                   f"in quotes.",
                        default=DEF_IMAGE_SIZE)
    parser.add_argument("-sg", "--sg_ratios", help=f"A comma-separated list of the S:G (guaiacol:syringyl) ratios to "
                                                   f"be tested. \nIf there are spaces, the list must be enclosed in "
                                                   f"quotes to be read as a single string. \nThe default list "
                                                   f"contains the single value {DEF_SG}.", default=[DEF_SG])
    parser.add_argument("-t", "--temperature_in_k", help=f"The temperature (in K) at which to model lignin "
                                                         f"biosynthesis. The default is {DEF_TEMP} K.",
                        default=DEF_TEMP)
    parser.add_argument("--chain_id", help=f"Option for use when generating a tcl script: the chainID to be used in "
                                           f"generating a psf \nand/or pdb file from a tcl script (see LigninBuilder). "
                                           f"This should be one character. If a \nlonger ID is provided, it will be "
                                           f"truncated to the first character. The default value is {DEF_CHAIN_ID}.",
                        default=DEF_CHAIN_ID)
    parser.add_argument("--psf_fname", help=f"Option for use when generating a tcl script: the file name for psf and "
                                            f"pdb files that will \nbe produced from running a tcl produced by this "
                                            f"package (see LigninBuilder). The default \nvalue is {DEF_PSF_FNAME}.",
                        default=DEF_PSF_FNAME)
    parser.add_argument("--toppar_dir", help=f"Option for use when generating a tcl script: the directory name where "
                                             f"VMD should look for \nthe toppar file(s) when running the tcl file in "
                                             f"VMD (see LigninBuilder). The default value \nis '{DEF_TOPPAR}'.",
                        default=DEF_TOPPAR)

    args = None
    try:
        args = parser.parse_args(argv)
        # dict below to map config input and defaults to command-line input
        conf_arg_dict = {OUT_DIR: args.out_dir,
                         OUT_FORMAT_LIST: args.output_format_list,
                         ADD_RATES: args.add_rates,
                         DYNAMICS: args.dynamics_flag,
                         ENERGY_BARRIER_FLAG: args.energy_barriers_flag,
                         INI_MONOS: args.initial_num_monomers,
                         SIM_TIME: args.length_simulation,
                         MAX_MONOS: args.max_num_monomers,
                         BASENAME: args.output_basename,
                         IMAGE_SIZE: args.image_size,
                         SG_RATIOS: args.sg_ratios,
                         TEMP: args.temperature_in_k,
                         RANDOM_SEED: args.random_seed,
                         CHAIN_ID: args.chain_id,
                         PSF_FNAME: args.psf_fname,
                         TOPPAR_DIR: args.toppar_dir,
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

        # only print the e if it has meaningful info; 2 simply is system exit from parser;
        #    tests that have triggered System Exit are caught and explained below
        if not e.args[0] == 2:
            warning(e)

        # Easy possible error is to have a space in a list; check for it
        check_arg_list = []
        for arg_str in ['-f', '--output_format_list', '-s', '--image_size']:
            if arg_str in argv:
                check_arg_list.append(arg_str)
        if len(check_arg_list) > 0:
            check_list = "', '".join(check_arg_list)
            warning(f"Check your entry/entries for '{check_list}'. If spaces separate list entries, "
                    f"enclose the whole list in quotes, or separate with commas only.")

        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def calc_rates(temp, ea_j_part_dict=None, ea_kcal_mol_dict=None):
    """
    Uses temperature and provided Gibbs free energy barriers (at 298.15 K and 1 atm) to calculate rates using the
        Eyring equation
    Dict formats: dict = {rxn_type: {substrate(s): {sub_lengths (e.g. (monomer, monomer)): value, ...}, ...}, ...}

    Only ea_j_part_dict or ea_kcal_mol_dict are needed; if both are provided, only ea_j_part_dict will be used

    :param temp: float, temperature in K
    :param ea_j_part_dict: dictionary of Gibbs free energy barriers in Joule/particle, in format noted above
    :param ea_kcal_mol_dict: dictionary of Gibbs free energy barriers in kcal/mol, in format noted above
    :return: rxn_rates: dict of reaction rates units of 1/s
    """
    # want Gibbs free energy barriers in J/particle for later calculation;
    #     user can provide them in those units or in kcal/mol
    if ea_j_part_dict is None:
        ea_j_part_dict = {
            rxn_type: {substrate: {sub_len: ea_kcal_mol_dict[rxn_type][substrate][sub_len] * KCAL_MOL_TO_J_PART
                                   for sub_len in ea_kcal_mol_dict[rxn_type][substrate]}
                       for substrate in ea_kcal_mol_dict[rxn_type]} for rxn_type in ea_kcal_mol_dict}
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
    #       sg_ratio is often used. However, until we have Gibbs free energy barriers for bonding between more than
    #       just S and G, no need to update
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
    if cfg[SAVE_PNG] or cfg[SAVE_SVG] or cfg[SAVE_JSON]:
        # PNG and SVG make 2D images and thus need coordinates
        # JSON will save coordinates--zero's if not computed; might as well compute and save non-zero values
        Compute2DCoords(mol)
    for save_format in [SAVE_TCL, SAVE_JSON, SAVE_PNG, SAVE_SVG]:
        if cfg[save_format]:
            fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=save_format)
            if save_format == SAVE_TCL:
                gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], tcl_fname=fname, chain_id=cfg[CHAIN_ID],
                           psf_fname=cfg[PSF_FNAME], toppar_dir=cfg[TOPPAR_DIR], out_dir=cfg[OUT_DIR])
            if save_format == SAVE_JSON:
                json_str = MolToJSON(mol)
                str_to_file(json_str + '\n', fname)
            elif save_format == SAVE_PNG or save_format == SAVE_SVG:
                MolToFile(mol, fname, size=cfg[IMAGE_SIZE])
            print(f"Wrote file: {fname}")


def validate_input(cfg):
    """
    Checking for errors at the beginning, so don't waste time starting calculations that will not be able to complete

    :param cfg: dict of configuration values
    :return: will raise an error if invalid data is encountered
    """
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

    # Convert list entries. Will already be lists if defaults are used. Otherwise, they will be strings.
    list_args = [ADD_RATES, SG_RATIOS]
    arg, arg_val = "", ""  # to make IDE happy
    try:
        for arg in list_args:
            arg_val = cfg[arg]
            # Will be a string to process unless it is the default
            if isinstance(arg_val, str):
                raw_vals = arg_val.replace(",", " ").replace("(", "").replace(")", "").split()
                cfg[arg] = [float(val) for val in raw_vals]
            else:
                cfg[arg] = arg_val
            for val in cfg[arg]:
                if val < 0:
                    raise ValueError
                # okay for sg_ratio to be zero, but not add_rate
                elif val == 0 and arg == ADD_RATES:
                    raise ValueError
    except ValueError:
        raise InvalidDataError(f"Found {arg_val} for '{arg}'. This entry must be able to be "
                               f"converted to a list of positive floats.")

    for req_pos_num in [SIM_TIME]:
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
        raise InvalidDataError(f"Found '{cfg[IMAGE_SIZE]}' input for '{IMAGE_SIZE}'. The {IMAGE_SIZE} must be "
                               f"two positive numbers, separated either by a comma or a space.")

    # Check for valid output requests
    check_if_files_to_be_saved(cfg)

    # determine rates to use
    if cfg[E_BARRIER_KCAL_MOL] == DEF_CFG_VALS[E_BARRIER_KCAL_MOL] and (cfg[E_BARRIER_J_PART] ==
                                                                        DEF_CFG_VALS[E_BARRIER_J_PART]):
        if cfg[ENERGY_BARRIER_FLAG]:
            cfg[RXN_RATES] = MANUSCRIPT_RATES
        else:
            cfg[RXN_RATES] = DEF_RXN_RATES
    else:
        if cfg[ENERGY_BARRIER_FLAG]:
            warning("Both the {ENERGY_BARRIER_FLAG} option and energy barriers have been provided. The "
                    "{ENERGY_BARRIER_FLAG} option will be ignored, and reaction rates will be calculated from the "
                    "provided energy barriers.")
        cfg[RXN_RATES] = calc_rates(cfg[TEMP], ea_j_part_dict=cfg[E_BARRIER_J_PART],
                                    ea_kcal_mol_dict=cfg[E_BARRIER_KCAL_MOL])


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

        for add_rate in cfg[ADD_RATES]:
            sg_adjs = []
            add_rate_str = f'{add_rate:.{3}g}'.replace("+", "").replace(".", "-")
            if cfg[RXN_RATES] == MANUSCRIPT_RATES:
                add_rate_str += "_e"
            for sg_ratio in cfg[SG_RATIOS]:
                # num_monos = []
                # num_oligs = []
                # adj_repeats = []

                # decide on initial monomers, based on given sg_ratio
                pct_s = sg_ratio / (1 + sg_ratio)
                ini_num_monos = cfg[INI_MONOS]
                if cfg[RANDOM_SEED]:
                    np.random.seed(cfg[RANDOM_SEED])
                    monomer_draw = np.around(np.random.rand(ini_num_monos), MAX_NUM_DECIMAL)
                else:
                    monomer_draw = np.random.rand(ini_num_monos)
                initial_monomers = create_initial_monomers(pct_s, monomer_draw)

                # initial event must be oxidation to create reactive species; all monomers may be oxidized
                initial_events = create_initial_events(initial_monomers, cfg[RXN_RATES])

                # initial_monomers and initial_events are grouped into the initial state
                initial_state = create_initial_state(initial_events, initial_monomers)
                if cfg[MAX_MONOS] > cfg[INI_MONOS]:
                    initial_events.append(Event(GROW, [], rate=cfg[ADD_RATE]))
                elif cfg[MAX_MONOS] < cfg[INI_MONOS]:
                    warning(f"The specified maximum number of monomers ({cfg[MAX_MONOS]}) is less than the specified "
                            f"initial number of monomers ({cfg[INI_MONOS]}). \n The program will proceed with the "
                            f"with the initial number of monomers with no addition of monomers.")

                # begin simulation
                result = run_kmc(cfg[RXN_RATES], initial_state, initial_events, n_max=cfg[MAX_MONOS],
                                 t_max=cfg[SIM_TIME], sg_ratio=sg_ratio, dynamics=cfg[DYNAMICS])

                # save for potential plotting
                sg_adjs.append(result[ADJ_MATRIX])

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

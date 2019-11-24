#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Goals:
1) find num monos vs. num oligomers over time, for given number of repeats, sg ratios, and addition rates
2) compare bond types vs. sg ratios (performing repeats) over a range of addition rates
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from common_wrangler.common import (GOOD_RET, INPUT_ERROR, INVALID_DATA, InvalidDataError, warning,
                                    create_out_fname, make_dir, OUT_DIR)

from ligninkmc.create_lignin import (DEF_ADD_RATE, DEF_SIM_TIME, DEF_SG, OPENING_MSG, DEF_TEMP, create_initial_monomers,
                                     create_initial_events, create_initial_state, get_bond_type_v_time_dict)
from ligninkmc.kmc_common import (Event, S, G, GROW, DEF_RXN_RATES, ADJ_MATRIX, BO4, BB, B5, B1, C5O4, AO4, C5C5, BONDS,
                                  INI_MONOS, MAX_MONOS, RANDOM_SEED, SIM_TIME, MAX_NUM_DECIMAL, MON_MON, MON_OLI,
                                  OLI_MON, OLI_OLI, MONOMER, OLIGOMER, OX, Q)
from ligninkmc.kmc_functions import (run_kmc, analyze_adj_matrix)


__author__ = 'hmayes'

# Config keys #
ADD_RATES = 'add_rates_list'
RXN_RATES = 'reaction_rates_at_298K'
SG_RATIOS = 'sg_ratio_list'
NUM_REPEATS = 'num_repeats'

BOND_TYPE_LIST = [BO4, BB, B5, B1, C5O4, AO4, C5C5]
COLORS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.6, 0), (0.6, 0, 0.6), (1, 0.549, 0),
          (0, 0.6, 0.6), (1, 0.8, 0), (0.6078, 0.2980, 0), (0.6, 0, 0), (0, 0, 0.6)]
DEF_INI_MONOS = 5
DEF_MAX_MONOS = 200
DEF_NUM_REPEATS = 5

ORELLA_RATES = {C5O4: {(0, 0): {MON_MON: 5904261.55598695, MON_OLI: 80333.560184945, OLI_MON: 80333.560184945,
                                OLI_OLI: 31893540751.2937},
                       (1, 0): {MON_MON: 8626534.12830123, MON_OLI: 80333.560184945, OLI_MON: 80333.560184945,
                                OLI_OLI: 31893540751.2937}},
                C5C5: {(0, 0): {MON_MON: 1141805.97148106, MON_OLI: 22698.3606666981, OLI_MON: 22698.3606666981,
                                OLI_OLI: 68083872447.6958}},
                B5: {(0, 0): {MON_MON: 7941635722.59467, MON_OLI: 5435496317.66216, OLI_MON: 5435496317.66216,
                              OLI_OLI: 5435496317.66216},
                     (0, 1): {MON_MON: 7941635722.59467, MON_OLI: 5435496317.66216, OLI_MON: 5435496317.66216,
                              OLI_OLI: 5435496317.66216}},
                BB: {(0, 0): {MON_MON: 11603278571.9039, MON_OLI: 11603278571.9039, OLI_MON: 11603278571.9039,
                              OLI_OLI: 11603278571.9039},
                     (1, 0): {MON_MON: 2243920367.77638, MON_OLI: 2243920367.77638, OLI_MON: 2243920367.77638,
                              OLI_OLI: 2243920367.77638},
                     (1, 1): {MON_MON: 11603278571.9039, MON_OLI: 11603278571.9039, OLI_MON: 11603278571.9039,
                              OLI_OLI: 11603278571.9039},
                     (0, 1): {MON_MON: 2243920367.77638, MON_OLI: 2243920367.77638, OLI_MON: 2243920367.77638,
                              OLI_OLI: 2243920367.77638}},
                BO4: {(0, 0): {MON_MON: 2889268780.92427, MON_OLI: 3278522716.22094, OLI_MON: 3278522716.22094,
                               OLI_OLI: 3278522716.22094},
                      (1, 0): {MON_MON: 83919112.8376677, MON_OLI: 3278522716.22094, OLI_MON: 3278522716.22094,
                               OLI_OLI: 3278522716.22094},
                      (0, 1): {MON_MON: 108054134.329644, MON_OLI: 3278522716.22094, OLI_MON: 3278522716.22094,
                               OLI_OLI: 3278522716.22094},
                      # below is where an entry is missing
                      (1, 1): {MON_MON: 34644086.8574001, MON_OLI: 16228844.7506668, OLI_MON: 16228844.7506668}},
                AO4: {(0, 0): {MON_MON: 36.0239749057507, MON_OLI: 36.0239749057507, OLI_MON: 36.0239749057507,
                               OLI_OLI: 36.0239749057507},
                      (1, 0): {MON_MON: 36.0239749057507, MON_OLI: 36.0239749057507, OLI_MON: 36.0239749057507,
                               OLI_OLI: 36.0239749057507},
                      (0, 1): {MON_MON: 36.0239749057507, MON_OLI: 36.0239749057507, OLI_MON: 36.0239749057507,
                               OLI_OLI: 36.0239749057507},
                      (1, 1): {MON_MON: 36.0239749057507, MON_OLI: 36.0239749057507, OLI_MON: 36.0239749057507,
                               OLI_OLI: 36.0239749057507}},
                B1: {(0, 0): {MON_OLI: 44607678.613794, OLI_MON: 44607678.613794, OLI_OLI: 44607678.613794},
                     (1, 0): {MON_OLI: 3138443.59211371, OLI_MON: 3138443.59211371, OLI_OLI: 3138443.59211371},
                     (0, 1): {MON_OLI: 11107513.4850607, OLI_MON: 11107513.4850607, OLI_OLI: 11107513.4850607},
                     (1, 1): {MON_OLI: 2437439.37772669, OLI_MON: 2437439.37772669, OLI_OLI: 2437439.37772669}},
                OX: {0: {MONOMER: 2659877051606.15, OLIGOMER: 2889268780.92427},
                     1: {MONOMER: 3886264174644.99, OLIGOMER: 514384986527.191}},
                Q: {0: {MONOMER: 6699707.46979824, OLIGOMER: 6699707.46979824},
                    1: {MONOMER: 3138443.59211371, OLIGOMER: 3138443.59211371}}}


# noinspection DuplicatedCode
def plot_mono_olig_v_time(x_axis, avg_num_monos, std_dev_monos, avg_num_oligs, std_dev_oligs, plot_title, plot_fname):
    plt.figure(figsize=(3.5, 3.5))
    plt.errorbar(x_axis, avg_num_monos, yerr=std_dev_monos, linestyle='none', marker='.',
                 markersize=10, markerfacecolor=COLORS[0], markeredgecolor=COLORS[0], label='monomers',
                 capsize=3, ecolor=COLORS[0])
    plt.errorbar(x_axis, avg_num_oligs, yerr=std_dev_oligs, linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[1], markeredgecolor=COLORS[1], label='oligomers', capsize=3, ecolor=COLORS[1])
    if len(x_axis) > 1:
        plt.xscale('log')

    [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
    plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1, length=4)
    plt.ylabel('Number', fontsize=14)
    plt.xlabel('Time step', fontsize=14)
    # plt.ylim([0.0, 1.0])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
    plt.title(plot_title)
    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)
    print(f"Wrote file: {plot_fname}")


# noinspection DuplicatedCode
def plot_bond_error_bars(x_axis, avg_bond_info, std_bond_info, plot_title, plot_fname):
    plt.figure(figsize=(3.5, 3.5))
    plt.errorbar(x_axis, avg_bond_info[BO4], yerr=std_bond_info[BO4], linestyle='none', marker='.',
                 markersize=10, markerfacecolor=COLORS[0], markeredgecolor=COLORS[0], label=BO4,
                 capsize=3, ecolor=COLORS[0])
    plt.errorbar(x_axis, avg_bond_info[BB], yerr=std_bond_info[BB], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[1], markeredgecolor=COLORS[1], label=BB, capsize=3, ecolor=COLORS[1])
    plt.errorbar(x_axis, avg_bond_info[B5], yerr=std_bond_info[B5], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[2], markeredgecolor=COLORS[2], label=B5, capsize=3, ecolor=COLORS[2])
    plt.errorbar(x_axis, avg_bond_info[B1], yerr=std_bond_info[B1], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[3], markeredgecolor=COLORS[3], label=B1, capsize=3, ecolor=COLORS[3])
    plt.errorbar(x_axis, avg_bond_info[C5O4], yerr=std_bond_info[C5O4], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[4], markeredgecolor=COLORS[4], label=C5O4, capsize=3, ecolor=COLORS[4])
    plt.errorbar(x_axis, avg_bond_info[AO4], yerr=std_bond_info[AO4], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[5], markeredgecolor=COLORS[5], label=AO4, capsize=3, ecolor=COLORS[5])
    plt.errorbar(x_axis, avg_bond_info[C5C5], yerr=std_bond_info[C5C5], linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[6], markeredgecolor=COLORS[6], label=C5C5, capsize=3, ecolor=COLORS[6])
    if len(x_axis) > 1:
        plt.xscale('log')

    [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
    plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1, length=4)
    plt.ylabel('Bond Type Yield (%)', fontsize=14)
    plt.xlabel('SG Ratio', fontsize=14)
    plt.ylim([0.0, 1.0])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
    plt.title(plot_title)
    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)
    print(f"Wrote file: {plot_fname}")


def get_avg_percent_bonds(bond_list, num_opts, adj_lists, num_trials):
    """
    Given adj_list for a set of options, with repeats for each option, find the avg and std dev of percent of each
    bond type
    :param bond_list: list of strings representing each bond type
    :param num_opts: number of options specified (should be length of adj_lists)
    :param adj_lists: list of lists of adjs: outer is for each option, inner is for each repeat
    :param num_trials: number of repeats (should be length of inner adj_lists list)
    :return: avg_bonds, std_bonds: list of floats, list of floats: for each option tested, the average and std dev
                  of bond distributions (percentages)
    """
    analysis = []
    for i in range(num_opts):
        cur_adjs = adj_lists[i]
        analysis.append([analyze_adj_matrix(cur_adjs[j]) for j in range(num_trials)])

    bond_percents = {}
    avg_bonds = {}
    std_bonds = {}

    for bond_type in bond_list:
        bond_percents[bond_type] = [[analysis[j][i][BONDS][bond_type]/sum(analysis[j][i][BONDS].values())
                                     for i in range(num_trials)] for j in range(num_opts)]
        avg_bonds[bond_type] = [np.mean(bond_pcts) for bond_pcts in bond_percents[bond_type]]
        std_bonds[bond_type] = [np.sqrt(np.var(bond_pcts)) for bond_pcts in bond_percents[bond_type]]
    return avg_bonds, std_bonds


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
                                                 'https://doi.org/10.1021/acssuschemeng.9b03534, and plot data on \n'
                                                 '  number of monomers and oligomers, and on bond type distribution, '
                                                 'as a function of S:G ratio and monomer \n  addition rate.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", "--add_rates", help=f"A comma-separated list of the rates of monomer addition to the "
                                                  f"system (in monomers/second). \nIf there are spaces, the list must "
                                                  f"be enclosed in quotes to be read as a single string. \nThe default "
                                                  f"list contains the single addition rate of {DEF_ADD_RATE} "
                                                  f"monomers/s.", default=[DEF_ADD_RATE])
    parser.add_argument("-d", "--out_dir", help="The directory where output files will be saved. The default is "
                                                "the current directory.", default=None)
    parser.add_argument("-e", "--energy_barriers", help=f"By default, the reaction rates will be based on the energy "
                                                        f"barriers in kcal/mol to be used to calculate reaction "
                                                        f"rates at {DEF_TEMP} K. If this flag is used, the rates used "
                                                        f"to produce the original manuscript plots will be used "
                                                        f"(missing one term).", action="store_true")
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_INI_MONOS)
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME*1000} s.",
                        default=DEF_SIM_TIME*1000)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-n", "--num_repeats", help=f"The number of times each sg_ratio and add_rate will be tested. "
                                                    f"The default is {DEF_NUM_REPEATS}. The minimum value is 3.",
                        default=DEF_NUM_REPEATS)
    parser.add_argument("-r", "--random_seed", help="Optional: a positive integer to be used as a seed value for "
                                                    "testing.", default=None)
    parser.add_argument("-sg", "--sg_ratios", help=f"A comma-separated list of the S:G (guaiacol:syringyl) ratios to "
                                                   f"be tested. \nIf there are spaces, the list must be enclosed in "
                                                   f"quotes to be read as a single string. \nThe default list "
                                                   f"contains the single value {DEF_SG}.", default=[DEF_SG])

    args = None
    try:
        args = parser.parse_args(argv)

    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET

        # only print the e if it has meaningful info
        if not e.args[0] == 2:
            warning(e)

        # Easy possible error is to have a space in a list; check for it
        check_arg_list = []
        for arg_str in ["-a", "--add_rates", "-sg", "--sg_ratio"]:
            if arg_str in argv:
                check_arg_list.append(arg_str)
        if len(check_arg_list) > 0:
            check_list = "', '".join(check_arg_list)
            warning(f"Check your entry/entries for '{check_list}'. If spaces separate list entries, "
                    f"enclose the whole list in quotes, or separate with commas only.")

        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def validate_input(args):
    """
    As needed, convert strings read on command line to required type. Save these entries, or the defaults when
    not entered, in a configuration dictionary. The ability to use global variables as the dictionary keys makes
    it more useful than using args attributes.
    :param args: Namespace object with arguments read from the command line
    :return: config: dict of configuration values
    """
    cfg = {}
    # Convert list entries. Will already be lists if defaults are used. Otherwise, they will be strings.
    list_args = {ADD_RATES: args.add_rates,
                 SG_RATIOS: args.sg_ratios}
    arg_dict = {ADD_RATES: 'add_rates',
                SG_RATIOS: 'sg_ratios'}
    arg, arg_val = "", ""  # to make IDE happy
    try:
        for arg, arg_val in list_args.items():
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
        raise InvalidDataError(f"Found {arg_val} for '--{arg_dict[ADD_RATES]}'. This entry must be able to be "
                               f"converted to a list of positive floats.")

    # if out_dir does not already exist, recreate it, only if we will actually need it
    if args.out_dir:
        make_dir(args.out_dir)
    cfg[OUT_DIR] = args.out_dir

    if args.energy_barriers:
        cfg[RXN_RATES] = ORELLA_RATES
    else:
        cfg[RXN_RATES] = DEF_RXN_RATES

    # Required ints
    int_args = {INI_MONOS: args.initial_num_monomers, MAX_MONOS: args.max_num_monomers, NUM_REPEATS: args.num_repeats}
    arg_dict = {INI_MONOS: 'initial_num_monomers', MAX_MONOS: 'max_num_monomers', NUM_REPEATS: 'num_repeats'}
    arg, arg_val = "", ""  # to make IDE happy
    try:
        for arg, arg_val in int_args.items():
            if isinstance(arg_val, str):
                cfg[arg] = int(arg_val)
            else:
                cfg[arg] = arg_val
    except ValueError:
        raise InvalidDataError(f"For '--{arg_dict[arg]}', found '{arg_val}'. This entry must be able to be converted "
                               f"into an integer.")
    if cfg[NUM_REPEATS] < 3:
        cfg[NUM_REPEATS] = 3
        warning(f"This script must be run with at least 3 repeats (found {args.num_repeats} for '{NUM_REPEATS}'). "
                f"The script will proceed with 3 repeats")

    # Don't use "args.random_seed:", because that won't catch the user giving the value 0, which they might think
    #    would be a valid random seed, but won't work for this package because of later "if cfg[RANDOM_SEED]:" checks
    if args.random_seed is None:
        cfg[RANDOM_SEED] = None
    else:
        try:
            # numpy seeds must be 0 and 2**32 - 1. Raise an error if the input cannot be converted to an int. Also raise
            #   an error for 0, since that will return False that a seed was provided in the logic in this package
            cfg[RANDOM_SEED] = int(args.random_seed)
            if cfg[RANDOM_SEED] <= 0 or cfg[RANDOM_SEED] > (2**32 - 1):
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Invalid input provided for '{RANDOM_SEED}': '{cfg[RANDOM_SEED]}'. If you "
                                   f"would like to obtain consistent output by using a random seed, provide a "
                                   f"positive integer value no greater than 2**32 - 1.")

    try:
        arg_val = args.length_simulation
        if isinstance(arg_val, str):
            cfg[SIM_TIME] = float(arg_val)
        else:
            cfg[SIM_TIME] = arg_val
        if cfg[SIM_TIME] <= 0:
            raise ValueError
    except ValueError:
        raise InvalidDataError(f"For '--length_simulation', found '{arg_val}'. This entry must be able to be "
                               f"converted into a positive float.")
    return cfg


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

    try:
        # tests at the beginning to catch errors early
        cfg = validate_input(args)

        for add_rate in cfg[ADD_RATES]:
            sg_adjs = []
            add_rate_str = f'{add_rate:.{3}g}'.replace("+", "").replace(".", "-")
            for sg_ratio in cfg[SG_RATIOS]:
                num_monos = []
                num_oligs = []
                adj_repeats = []

                for _ in range(cfg[NUM_REPEATS]):
                    if cfg[RANDOM_SEED]:
                        np.random.seed(cfg[RANDOM_SEED])
                        monomer_draw = np.around(np.random.rand(cfg[INI_MONOS]), MAX_NUM_DECIMAL)
                    else:
                        monomer_draw = np.random.rand(cfg[INI_MONOS])
                    initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
                    initial_events = create_initial_events(initial_monomers, cfg[RXN_RATES])
                    initial_state = create_initial_state(initial_events, initial_monomers)
                    if cfg[MAX_MONOS] > cfg[INI_MONOS]:
                        initial_events.append(Event(GROW, [], rate=add_rate))
                    elif cfg[MAX_MONOS] < cfg[INI_MONOS]:
                        warning(f"The specified {MAX_MONOS} ({cfg[MAX_MONOS]}) is less than the specified {INI_MONOS} "
                                f"({cfg[INI_MONOS]}). \n          The program will proceed with the initial "
                                f"number of monomers with no addition of monomers.")
                    result = run_kmc(cfg[RXN_RATES], initial_state, initial_events,
                                     n_max=cfg[MAX_MONOS], sg_ratio=sg_ratio, t_max=cfg[SIM_TIME], dynamics=True,
                                     random_seed=cfg[RANDOM_SEED])
                    adj_list = result[ADJ_MATRIX]
                    # following will be used to analyze final bonds only
                    adj_repeats.append(adj_list[-1])
                    # only need num monos, num oligs, but we'll get everything
                    (bond_type_dict, olig_monos_dict, sum_monos_list, olig_count_dict,
                     sum_count_list) = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=2)

                    num_monos.append(olig_count_dict[1])
                    num_oligs.append(sum_count_list)

                sg_adjs.append(adj_repeats)
                # Arrays may be different lengths, so find shortest array
                min_len = len(num_monos[0])
                for mono_list in num_monos[1:]:
                    if len(mono_list) < min_len:
                        min_len = len(mono_list)
                # make lists of lists into np array
                sg_num_monos = np.asarray([np.array(num_list[:min_len]) for num_list in num_monos])
                # could save, but I'm just going to print
                av_num_monos = np.mean(sg_num_monos, axis=0)
                std_num_monos = np.std(sg_num_monos, axis=0)

                sg_num_oligs = np.asarray([np.array(num_list[:min_len]) for num_list in num_oligs])
                av_num_oligs = np.mean(sg_num_oligs, axis=0)
                std_num_oligs = np.std(sg_num_oligs, axis=0)

                timesteps = list(range(min_len))
                title = f"S:G Ratio {sg_ratio}, Add rate {add_rate_str} monomer/s"
                sg_str = f'{sg_ratio:.{3}g}'.replace("+", "").replace(".", "-")
                fname = create_out_fname(f'mono_v_olig_{sg_str}_{add_rate_str}', base_dir=cfg[OUT_DIR], ext='.png')
                plot_mono_olig_v_time(timesteps, av_num_monos, std_num_monos, av_num_oligs, std_num_oligs, title, fname)

            all_avg_bonds, all_std_bonds = get_avg_percent_bonds(BOND_TYPE_LIST, len(cfg[SG_RATIOS]), sg_adjs,
                                                                 cfg[NUM_REPEATS])

            title = f"Add rate {add_rate_str} monomer/second"
            fname = create_out_fname(f'bond_v_add_rate_{add_rate_str}', base_dir=cfg[OUT_DIR], ext='.png')
            plot_bond_error_bars(cfg[SG_RATIOS], all_avg_bonds, all_std_bonds, title, fname)

    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launches steps required to build lignin
Multiple output options, from tcl files to plots
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
                                  OLI_MON, OLI_OLI, MONOMER, OLIGOMER, OX, Q, MANUSCRIPT_RATES)
from ligninkmc.kmc_functions import (run_kmc, analyze_adj_matrix)


# Config keys #
ADD_RATES = 'add_rates_list'
RXN_RATES = 'reaction_rates_at_298K'
SG_RATIOS = 'sg_ratio_list'
NUM_REPEATS = 'num_repeats'

BOND_TYPE_LIST = [BO4, BB, B5, B1, C5O4, AO4, C5C5]

DEF_INI_MONOS = 5
DEF_MAX_MONOS = 200
DEF_NUM_REPEATS = 5


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

    parser.add_argument("-n", "--num_repeats", help=f"The number of times each sg_ratio and add_rate will be tested. "
                                                    f"The default is {DEF_NUM_REPEATS}. The minimum value is 3.",
                        default=DEF_NUM_REPEATS)

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

    if args.energy_barriers:
        cfg[RXN_RATES] = MANUSCRIPT_RATES
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
            if cfg[RXN_RATES] == MANUSCRIPT_RATES:
                add_rate_str += "_e"
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

                    # todo: delete
                    print(cfg[RXN_RATES][BO4])
                    result = run_kmc(cfg[RXN_RATES], initial_state, initial_events,
                                     n_max=cfg[MAX_MONOS], sg_ratio=sg_ratio, t_max=cfg[SIM_TIME],
                                     dynamics=args.dynamics, random_seed=cfg[RANDOM_SEED])
                    adj_list = result[ADJ_MATRIX]
                    if args.dynamics:
                        # following will be used to analyze final bonds only
                        adj_repeats.append(adj_list[-1])
                        # only need num monos, num oligs, but we'll get everything
                        (bond_type_dict, olig_monos_dict, sum_monos_list, olig_count_dict,
                         sum_count_list) = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=2)

                        num_monos.append(olig_count_dict[1])
                        num_oligs.append(sum_count_list)
                    else:
                        adj_repeats.append(adj_list)

                sg_adjs.append(adj_repeats)
                if args.dynamics:
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
                    plot_mono_olig_v_time(timesteps, av_num_monos, std_num_monos, av_num_oligs,
                                          std_num_oligs, title, fname)

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

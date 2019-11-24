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
from common_wrangler.common import (GOOD_RET,  INPUT_ERROR, INVALID_DATA, InvalidDataError, warning,
                                    create_out_fname, )

from ligninkmc.create_lignin import (DEF_ADD_RATE, DEF_SIM_TIME, DEF_SG, OPENING_MSG, create_initial_monomers,
                                     create_initial_events, create_initial_state, get_bond_type_v_time_dict)
from ligninkmc.kmc_common import (Event, S, G, GROW, DEF_RXN_RATES, ADJ_MATRIX, BO4, BB, B5, B1, C5O4, AO4, C5C5, BONDS)
from ligninkmc.kmc_functions import (run_kmc, analyze_adj_matrix)


__author__ = 'hmayes'

# Config keys #
BOND_TYPE_LIST = [BO4, BB, B5, B1, C5O4, AO4, C5C5]
COLORS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.6, 0), (0.6, 0, 0.6), (1, 0.549, 0),
          (0, 0.6, 0.6), (1, 0.8, 0), (0.6078, 0.2980, 0), (0.6, 0, 0), (0, 0, 0.6)]
DEF_INI_MONOS = 5
DEF_MAX_MONOS = 200


# noinspection DuplicatedCode
def plot_mono_olig_v_time(x_axis, avg_num_monos, std_dev_monos, avg_num_oligs, std_dev_oligs, plot_title, plot_fname):
    plt.figure(figsize=(3.5, 3.5))
    plt.errorbar(x_axis, avg_num_monos, yerr=std_dev_monos, linestyle='none', marker='.',
                 markersize=10, markerfacecolor=COLORS[0], markeredgecolor=COLORS[0], label='monomers',
                 capsize=3, ecolor=COLORS[0])
    plt.errorbar(x_axis, avg_num_oligs, yerr=std_dev_oligs, linestyle='none', marker='.', markersize=10,
                 markerfacecolor=COLORS[1], markeredgecolor=COLORS[1], label='oligomers', capsize=3, ecolor=COLORS[1])
    plt.xscale('log')

    [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
    plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1, length=4)
    plt.ylabel('Number', fontsize=14)
    plt.xlabel('Time step', fontsize=14)
    plt.ylim([0.0, 1.0])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
    plt.title(plot_title)
    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)


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


def get_avg_percent_bonds(bond_list, num_opts, result_list, num_trials):
    analysis = []
    for i in range(num_opts):
        opt_results = result_list[i]
        cur_adjs = [opt_results[j][ADJ_MATRIX] for j in range(num_trials)]
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


# noinspection DuplicatedCode
def plot_bond_form_error_bars(x_axis, avg_bond_info, std_bond_info, plot_title, plot_fname):
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
                                                 'https://doi.org/10.1021/acssuschemeng.9b03534, and plot data re '
                                                 'monomer or oligomer length and bond type distribution, as a '
                                                 'function of S:G ratio and monomer addition rate. Uses the default '
                                                 "reaction rates as described in 'create_lignin.'",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", "--add_rates", help=f"A comma-separated list of the rates of monomer addition to the "
                                                  f"system (in monomers/second). If there are spaces, the list must "
                                                  f"be enclosed in quotes to be read as a single string. The default "
                                                  f"list contains the single addition rate of {DEF_ADD_RATE} "
                                                  f"monomers/s.", default=[DEF_ADD_RATE])
    parser.add_argument("-d", "--out_dir", help="The directory where output files will be saved. The default is "
                                                "the current directory.", default=None)
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_INI_MONOS)
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME*1000} s.",
                        default=DEF_SIM_TIME*1000)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-r", "--random_seed", help="Optional: a positive integer to be used as a seed value for "
                                                    "testing.", default=None)
    parser.add_argument("-sg", "--sg_ratio", help=f"A comma-separated list of the S:G (guaiacol:syringyl) ratios to "
                                                  f"be tested, If there are spaces, the list must be enclosed in "
                                                  f"quotes to be read as a single string. The default list contains "
                                                  f"the single value {DEF_SG}.", default=[DEF_SG])

    args = None
    try:
        args = parser.parse_args(argv)

    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def validate_input(args):
    print(args)


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
        validate_input(args)

        ini_num_mons = 5
        max_num_mons = 20
        # sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
        # sg_opts = [0.1, 1, 10]
        sg_opts = [0.1]
        # add_rates = [1e8, 1e6, 1e4, 1e2, 1]
        # add_rates = [1e8, 1e4, 1]
        add_rates = [1e8]
        num_repeats = 5

        for add_rate in add_rates:
            for sg_ratio in sg_opts:
                num_monos = []
                num_oligs = []

                for _ in range(num_repeats):
                    monomer_draw = np.random.rand(ini_num_mons)
                    initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
                    initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
                    initial_state = create_initial_state(initial_events, initial_monomers)
                    initial_events.append(Event(GROW, [], rate=add_rate))
                    result = run_kmc(DEF_RXN_RATES, initial_state, initial_events, n_max=max_num_mons,
                                     sg_ratio=sg_ratio, t_max=12400000, dynamics=True)
                    adj_list = result[ADJ_MATRIX]
                    # only need num monos, num oligs, but we'll get everything
                    (bond_type_dict, olig_monos_dict, sum_monos_list, olig_count_dict,
                     sum_count_list) = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=2)

                    num_monos.append(olig_count_dict[1])
                    num_oligs.append(sum_count_list)

                # find shortest array
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
                title = f"S:G Ratio {sg_ratio}, Add rate {add_rate} monomer/s"
                sg_str = str(sg_ratio).replace(".", "-")
                add_str = str(add_rate)
                fname = create_out_fname(f'mono_v_olig_{sg_str}_{add_str}', base_dir=args.out_dir, ext='.png')
                plot_mono_olig_v_time(timesteps, av_num_monos, std_num_monos, av_num_oligs, std_num_oligs, title, fname)
                print("there")

        for add_rate in add_rates:
            sg_results = []
            for sg_ratio in sg_opts:
                results = []
                for _ in range(num_repeats):
                    monomer_draw = np.random.rand(ini_num_mons)
                    initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
                    initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
                    initial_state = create_initial_state(initial_events, initial_monomers)
                    initial_events.append(Event(GROW, [], rate=add_rate))
                    result = run_kmc(DEF_RXN_RATES, initial_state, initial_events, n_max=max_num_mons,
                                     sg_ratio=sg_ratio, t_max=12400000)
                    results.append(result)
                    print("hi")

                sg_results.append(results)
                print("there")
            all_avg_bonds, all_std_bonds = get_avg_percent_bonds(BOND_TYPE_LIST, len(sg_opts), sg_results, num_repeats)
            print(all_avg_bonds, all_std_bonds)

    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

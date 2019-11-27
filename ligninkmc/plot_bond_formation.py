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





def main(argv=None):
    """
    Runs the main program.

    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """

    cfg = []

    try:
        # tests at the beginning to catch errors early
        for sg_ratio in cfg[SG_RATIOS]:
            num_monos = []
            num_oligs = []
            adj_repeats = []

            for _ in range(cfg[NUM_REPEATS]):

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




    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

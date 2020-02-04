#!/usr/bin/env python3
import logging
import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import optimize
from ligninkmc.create_lignin import (DEF_TEMP, calc_rates, create_initial_monomers, create_initial_events,
                                     degree, create_initial_state, overall_branching_coefficient,
                                     adj_analysis_to_stdout, get_bond_type_v_time_dict)
from ligninkmc.kmc_common import (Event, Monomer, G, S, H, C, C5O4, OX, C5C5, B5, BB, BO4, AO4, B1, DEF_RXN_RATES,
                                  MON_OLI, MONOMER, GROW, TIME, MONO_LIST, ADJ_MATRIX, CHAIN_LEN, BONDS,
                                  RCF_YIELDS, RCF_BONDS, B1_ALT, DEF_E_BARRIER_KCAL_MOL, MAX_NUM_DECIMAL)
from ligninkmc.kmc_functions import (run_kmc, generate_mol, gen_tcl, find_fragments, fragment_size, break_bond_type,
                                     analyze_adj_matrix, count_oligomer_yields, count_bonds)

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


def save_svg(svg_fname):
    plt.savefig(svg_fname, format='svg', transparent=True, bbox_inches='tight')
    print("wrote:", svg_fname)
    plt.close()


# Tests #

class TestScaling(unittest.TestCase):
    def testFindScaling(self):
        # To test scaling: replicate batch runs with different numbers of monomers
        # Here, we are testing with equal amount of S and G (no C)
        times = []
        sg_ratio = 1
        pct_s = sg_ratio / (1 + sg_ratio)

        test_vals = np.linspace(50, 150, num=3, dtype='int32')
        num_repeats = 5
        for num_monos in test_vals:
            print(f"Starting batch simulation with {num_monos} monomers")
            times.append([])
            for i in range(num_repeats):
                random_seed = 8 + i
                np.random.seed(random_seed)
                print(f"    Starting repeat", i)
                # Generate the initial monomers and events (oxidation)
                monomer_draw = np.random.rand(num_monos)
                initial_monomers = create_initial_monomers(pct_s, monomer_draw)
                initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
                # Set the state and add the option to join initial monomers
                initial_state = create_initial_state(initial_events, initial_monomers)

                # Start timing the actual KMC part
                # noinspection PyUnboundLocalVariable
                start = time.time()
                run_kmc(DEF_RXN_RATES, initial_state, initial_events, sg_ratio=sg_ratio, random_seed=random_seed)
                end = time.time()
                times[-1].append(end - start)
            print(f'Average time to complete simulation with {num_monos:5n} monomers: '
                  f'{np.sum(times[-1]) / num_repeats:7.2f} seconds')

        # Now we want to fit the times that we just calculated to a generic power law expression $t = aN^b$ to find the
        # scaling of our algorithm.
        meas_t = [np.mean(one_time) for one_time in times]
        # sdev_t = [np.sqrt(np.var(one_time)) for one_time in times]
        meas_n = test_vals

        sim_t = lambda p, n: p[0] * np.power(n, p[1])
        loss = lambda p: np.linalg.norm(sim_t(p, meas_n) - meas_t)

        results = optimize.minimize(loss, np.asarray([1e-5, 2.5]), bounds=[[0, 1], [0, 10]], options={'disp': True})
        opt_p = results.x
        scaling_formula = f'$t = {opt_p[0]:3.1e}N^{{ {opt_p[1]:4.2f} }}$'
        print(f'Scaling: {scaling_formula}')

# Without hash:
# Average time to complete simulation with    50 monomers:   25.62 seconds
# gave up before further data

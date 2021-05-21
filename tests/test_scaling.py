#!/usr/bin/env python3
import logging
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from ligninkmc.create_lignin import (create_initial_monomers, create_initial_events, create_initial_state)
from ligninkmc.kmc_common import (DEF_RXN_RATES)
from ligninkmc.kmc_functions import (run_kmc)

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
        test_scaling = False
        if test_scaling:
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

            # Now we want to fit the times that we just calculated to a generic power law expression
            # $t = aN^b$ to find the scaling of our algorithm.
            meas_t = [np.mean(one_time) for one_time in times]
            # sdev_t = [np.sqrt(np.var(one_time)) for one_time in times]
            meas_n = test_vals

            sim_t = lambda p, n: p[0] * np.power(n, p[1])
            loss = lambda p: np.linalg.norm(sim_t(p, meas_n) - meas_t)

            results = optimize.minimize(loss, np.asarray([1e-5, 2.5]), bounds=[[0, 1], [0, 10]], options={'disp': True})
            opt_p = results.x
            scaling_formula = f'$t = {opt_p[0]:3.1e}N^{{ {opt_p[1]:4.2f} }}$'
            print(f'Scaling: {scaling_formula}')

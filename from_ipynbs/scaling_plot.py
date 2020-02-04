#!/usr/bin/env python
# coding: utf-8

# # Kinetic Monte Carlo Simulation of Lignin Polymerization--batch scaling calculation
# Written by: Michael Orella <br>
# 2019 January 24 <br>
# Updated by: Heather Mayes <br>
# 2019 November 27 <br>
#
# functions, global variables, and default values (described below) from the lignin-kmc Package 
from ligninkmc.create_lignin import (create_initial_monomers, create_initial_events, create_initial_state)
from ligninkmc.kmc_functions import (run_kmc)
from ligninkmc.kmc_common import (DEF_RXN_RATES)

# General Math and LinAlg
import numpy as np
from scipy import optimize

#Plotting
import matplotlib.pyplot as plt

# For performance
import time


def save_svg(svg_fname):
    plt.savefig(svg_fname, format='svg', transparent=True, bbox_inches='tight')
    print("wrote:", svg_fname)
    plt.close()


# ## Code Performance
# One of the first things that we were interested in reporting was the performance and scaling of this code package.
# This was done by replicating runs of the algorithm with different numbers of monomers included in the simulation
# under batch conditions.

# Here, we are testing with equal amount of S and G (no C)
times = []
sg_ratio = 1
pct_s = sg_ratio / (1 + sg_ratio)

test_vals = np.linspace(50, 150, num=3, dtype ='int32')
num_repeats = 5
for num_monos in test_vals:
    print(f"Starting batch simulation with {num_monos} monomers")
    times.append([])
    for i in range(num_repeats):
        print(f"    Starting repeat", i)
        # Generate the initial monomers and events (oxidation)
        monomer_draw = np.random.rand(num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        # Set the state and add the option to join initial monomers
        initial_state = create_initial_state(initial_events, initial_monomers)

        #Start timing the actual KMC part
        start = time.time()
        run_kmc(DEF_RXN_RATES, initial_state, initial_events, sg_ratio=sg_ratio)
        end = time.time()
        times[-1].append(end-start)
    print(f'Average time to complete simulation with {num_monos:5n} monomers: {np.sum(times[-1])/num_repeats:7.2f} seconds')


# Now we want to fit the times that we just calculated to a generic power law expression $t = aN^b$ to find the
# scaling of our algorithm.
meas_t = [np.mean(time) for time in times]
meas_n = test_vals

sim_t = lambda p, n: p[0] * np.power (n, p[1])
loss = lambda p: np.linalg.norm(sim_t(p, meas_n) - meas_t)

results = optimize.minimize(loss, np.asarray([1e-5, 2.5]), bounds=[[0,1], [0,10]], options={'disp': True})
opt_p = results.x
scaling_formula = f'$t = {opt_p[0]:3.1e}N^{{ {opt_p[1]:4.2f} }}$'
print(f'Scaling: {scaling_formula}')


# Now we should plot both the measured values and the fit all together
plt.figure(figsize=(3.5, 3.5))
plt.errorbar(test_vals, [np.mean(time) for time in times],
             yerr=[np.sqrt(np.var(time)) for time in times],
             capsize=3, ecolor='black', linestyle='None', marker='.', markerSize=15, color='black', zorder=1)
plt.plot(test_vals, sim_t(opt_p,meas_n), linestyle='--', color='r', linewidth=1.5, zorder=2)
plt.tick_params(axis='both', which ='major', labelsize=10, direction='in',
                pad=8, top = True, right=True, width=1.5, length=5)
plt.tick_params(axis='both', which='minor', direction='in',
                pad=8, top=True, right=True, width=1, length=3)
ax = plt.gca()
[ax.spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
ax.fontsize = 10
plt.xlabel('Number of Monomers', fontsize=10)
plt.ylabel('Execution Time (s)', fontsize=10)
plt.yscale('log')
plt.xscale('log')
plt.ylim([0.2, 200])
plt.xlim([40, 200])
plt.text(75, 0.4, scaling_formula, fontsize=10, color='red')
plt.text(200, 200, r'Measured', fontsize=10, color='black')
save_svg('temp_performance.svg')

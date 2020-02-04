#!/usr/bin/env python
# coding: utf-8

# # Kinetic Monte Carlo Simulation of Lignin Polymerization
# Written by: Michael Orella <br>
# 2019 January 24 <br>
# Updated by: Heather Mayes <br>
# 2019 November 27 <br>
# 
# The code in this notebook performs calculations analogous to those performed in ["Lignin-KMC: A Toolkit for
# Simulating Lignin Biosynthesis"](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534), which depend on the results that were obtained from [DFT calculations of monolignol coupling kinetics](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b02506).
# 
# **Note:** this notebook directly calls various functions in the package. Users can also run Lignin-KMC from the command line, without the need to directly interact with the python code. See the [README on github](https://github.com/michaelorella/lignin-kmc).

# functions, global variables, and default values (described below) from the lignin-kmc Package 
from ligninkmc.create_lignin import (calc_rates, create_initial_monomers, create_initial_events, 
                                     create_initial_state, DEF_ADD_RATE)
from ligninkmc.kmc_functions import (run_kmc, generate_mol, analyze_adj_matrix)
from ligninkmc.kmc_common import (GROW, OX, BO4, B5, BB, B1, C5O4, AO4, C5C5, G, S, C, Q, MONOMER, OLIGOMER,
                                  MONO_LIST, ADJ_MATRIX, BONDS, CHAIN_LEN, RCF_YIELDS, Event, Monomer)

# General Math and LinAlg
import numpy as np
from scipy import optimize

# Chemical Drawing
from rdkit.Chem import MolFromMolBlock
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import MolToImage, MolToFile

#Plotting
import matplotlib.pyplot as plt

# For performance
import time

#Parallelization
import joblib as par


# ## Input Data
# The kinetic monte carlo code relies on rates of chemical reactions. The reactions that we are interested in here are
# the coupling of radicals on either individual monomers or oligomers respectively. The energetics necessary to compute
# the reaction rates were calculated by Terry Gani using [DFT methods](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b02506) for SG lignin and similar methods for C lignin. Once the reaction energies were calculated, they could converted to reaction rates using the Eyring equation:
# $$ r_i = \dfrac{k_BT}{h}\exp\left(-\dfrac{\Delta G_i}{k_BT}\right) $$
# 
# Note: the temperature used must match the temperature at which the energy barriers were calculated.
# The these Gibbs free energy barriers (at 298.15 K and 1 atm), as reported in the [Lignin-KMC paper](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534), [Tables S1 and S2](https://pubs.acs.org/doi/suppl/10.1021/acssuschemeng.9b03534/suppl_file/sc9b03534_si_001.pdf), are stored in the package's global variable DEF_E_BARRIER_KCAL_MOL (imported above). 
# 
# The user may use other values. The assignment of energy barriers (in kcal/mol) to this global variable is shown
# below (using global variables for reaction types to help prevent typos from directly using strings) to provide a
# template for how to assign alternate values. Of course, the user only needs to (and must) supply energy barriers
# (or directly supply reaction rates) only for the monomer types to be modeled and the reaction types to be modeled
# at the temperature of interest. Here, the bond names stands for creation of that bond.
# 
# Throughout the code, monomers are kept track of individually through the state, so there are no reaction
# degeneracies occurring, and bond events can be tracked faithfully.
# 
# The monomer types are:  0: guaiacol, 1: syringyl, 2: caffeoyl
# The definition for the default Gibbs free energy barriers in kcal/mol (at 298.15 K and 1 atm), shown here as a 
# template for user-input values. The global variables MONOMER, OLIGOMER, BO4, etc., can be imported from kmc.kmc_common.
# OLIGOMER stands for dimer or longer ("2+" in the SI)

def_e_barrier_kcal_mol = {C5O4: {(G, G): {(MONOMER, MONOMER): 11.2, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (S, G): {(MONOMER, MONOMER): 10.9, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (C, C): {(MONOMER, MONOMER): 11.9, (MONOMER, OLIGOMER): 11.9,
                                          (OLIGOMER, MONOMER): 11.9, (OLIGOMER, OLIGOMER): 11.9}},
                          C5C5: {(G, G): {(MONOMER, MONOMER): 12.5, (MONOMER, OLIGOMER): 15.6,
                                          (OLIGOMER, MONOMER): 15.6, (OLIGOMER, OLIGOMER): 3.8},
                                 (C, C): {(MONOMER, MONOMER): 10.6, (MONOMER, OLIGOMER): 10.6,
                                          (OLIGOMER, MONOMER): 10.6, (OLIGOMER, OLIGOMER): 10.6}},
                          B5: {(G, G): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (G, S): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (C, C): {(MONOMER, MONOMER): 1.9, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8}},
                          BB: {(G, G): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (S, G): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (G, S): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (S, S): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (C, C): {(MONOMER, MONOMER): 7.2, (MONOMER, OLIGOMER): 7.2, (OLIGOMER, MONOMER): 7.2,
                                        (OLIGOMER, OLIGOMER): 7.2}},
                          BO4: {(G, G): {(MONOMER, MONOMER): 6.3, (MONOMER, OLIGOMER): 6.2, (OLIGOMER, MONOMER): 6.2,
                                         (OLIGOMER, OLIGOMER): 6.2},
                                (S, G): {(MONOMER, MONOMER): 9.1, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (G, S): {(MONOMER, MONOMER): 8.9, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (S, S): {(MONOMER, MONOMER): 9.8, (MONOMER, OLIGOMER): 10.4,
                                         (OLIGOMER, MONOMER): 10.4, (OLIGOMER, OLIGOMER): 10.4},
                                (C, C): {(MONOMER, MONOMER): 4.9, (MONOMER, OLIGOMER): 1.3,
                                         (OLIGOMER, MONOMER): 1.3, (OLIGOMER, OLIGOMER): 1.3}},
                          AO4: {(G, G): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (S, G): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (G, S): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (S, S): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (C, C): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7}},
                          B1: {(G, G): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6},
                               (S, G): {(MONOMER, OLIGOMER): 11.7, (OLIGOMER, MONOMER): 11.7,
                                        (OLIGOMER, OLIGOMER): 11.7},
                               (G, S): {(MONOMER, OLIGOMER): 10.7, (OLIGOMER, MONOMER): 10.7,
                                        (OLIGOMER, OLIGOMER): 10.7},
                               (S, S): {(MONOMER, OLIGOMER): 11.9, (OLIGOMER, MONOMER): 11.9,
                                        (OLIGOMER, OLIGOMER): 11.9},
                               (C, C): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6}},
                          OX: {G: {MONOMER: 0.9, OLIGOMER: 6.3}, S: {MONOMER: 0.6, OLIGOMER: 2.2},
                               C: {MONOMER: 0.9, OLIGOMER: 0.9}},
                          Q: {G: {MONOMER: 11.1, OLIGOMER: 11.1}, S: {MONOMER: 11.7, OLIGOMER: 11.7},
                             C: {MONOMER: 11.1, OLIGOMER: 11.1}}}


def save_svg(svg_fname):
    plt.savefig(svg_fname, format='svg', transparent=True, bbox_inches='tight')
    print("wrote:", svg_fname)
    plt.close()


# Calculate the rates of reaction in 1/s at the specified temp
temp = 298.15  # K
rxn_rates = calc_rates(temp, ea_kcal_mol_dict=def_e_barrier_kcal_mol)

# if the user has instead input Gibbs free energy barriers in Joules per particle (e.g., as `my_ea_j_part_dict`),
# instead invoke:
#    rxn_rates = calc_rates(temp, ea_j_part_dict=my_ea_j_part_dict)

# ## Code Performance
# One of the first things that we were interested in reporting was the performance and scaling of this code package. This was done by replicating runs of the algorithm with different numbers of monomers included in the simulation under batch conditions.

# Here, we are testing with equal amount of S and G (no C)
times = []
sg_ratio = 1
pct_s = sg_ratio / (1 + sg_ratio)

test_vals = np.linspace(50, 150, num=3, dtype ='int32')
num_repeats = 5
for num_monos in test_vals:
    times.append([])
    for i in range(num_repeats):
        print("Starting repeat", i)
        # Generate the initial monomers and events (oxidation)
        monomer_draw = np.random.rand(num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)
        initial_events = create_initial_events(initial_monomers, rxn_rates)

        # Set the state and add the option to join initial monomers
        initial_state = create_initial_state(initial_events, initial_monomers)
        initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE))

        #Start timing the actual KMC part
        start = time.time()
        run_kmc(rxn_rates, initial_state, initial_events, sg_ratio=sg_ratio)
        end = time.time()
        times[-1].append(end-start)
    print(f'Time to complete simulation with {num_monos:5n} monomers: {np.sum(times[-1])/num_repeats:7.2f} seconds')

# Now we want to fit the times that we just calculated to a generic power law expression $t = aN^b$ to find the scaling of our algorithm.

meas_t = [np.mean(time) for time in times]
meas_n = test_vals

sim_t = lambda p, n: p[0] * np.power (n, p[1])
loss = lambda p: np.linalg.norm(sim_t(p, meas_n) - meas_t)

results = optimize.minimize(loss, np.asarray([1e-5, 2.5]), bounds=[[0,1], [0,10]], options={'disp': True})
print(results)

opt_p = results.x

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
plt.text(75, 0.4, f'$t = {opt_p[0]:3.1e}N^{{ {opt_p[1]:4.2f} }}$', fontsize=10, color='red')
plt.text(200, 200, r'Measured', fontsize=10, color='black')

fname = 'temp_performance.svg'
plt.savefig(fname, format='svg', transparent=True, bbox_inches='tight')
print("wrote:", fname)

#
# # ## Example KMC Runs
# # Small examples of running the algorithm for KMC, using both SG lignin and C lignin.
#
# # ### SG Lignin
# # Run an example starting with 2 monomers and an S to G ratio of 1
# num_monos = 2
# sg_ratio = 1
# pct_s = sg_ratio / (1 + sg_ratio)
#
# monomer_draw = np.random.rand(num_monos)
# initial_monomers = create_initial_monomers(pct_s, monomer_draw)
# initial_events = create_initial_events(initial_monomers, rxn_rates)
# initial_state = create_initial_state(initial_events, initial_monomers)
# initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE))
#
# result = run_kmc(rxn_rates, initial_state, initial_events, n_max=10, t_max=1, sg_ratio=sg_ratio)
# nodes = result[MONO_LIST]
# adj = result[ADJ_MATRIX]
# block = generate_mol(adj, nodes)
# mol = MolFromMolBlock(block)
# Compute2DCoords(mol)
# MolToImage(mol, size=(950, 250))
# fname = "temp_ligpoly_sg.png"
# MolToFile(mol, fname, size=(950, 300))
#
# # ### C lignin
# # Run an example with 2 monomers but using the C lignin energies
#
# ini_num_monos = 2
#
# initial_monomers = [Monomer(C, i) for i in range(ini_num_monos)]
# initial_events = [Event(OX, [i], rxn_rates[OX][C][MONOMER]) for i in range(len(initial_monomers))]
# initial_state = create_initial_state(initial_events, initial_monomers)
# # `events` may be a set or a list
# events = {initial_events[i] for i in range(len(initial_monomers))}
# events.add(Event(GROW, [], rate=DEF_ADD_RATE))
#
# result = run_kmc(rxn_rates, initial_state, sorted(events), n_max=10, t_max=1)
#
# nodes = result[MONO_LIST]
# adj = result[ADJ_MATRIX]
# block = generate_mol(adj, nodes)
# mol = MolFromMolBlock(block)
# Compute2DCoords(mol)
# MolToImage(mol, size=(950, 250))
# fname = "temp_ligpoly_c.png"
# MolToFile(mol, fname, size=(950, 300))
#
# # ## Sensitivity Analyses Examples
# # The meat of the results and discussion for our paper lay in the predictions of how lignin composition should change
# # with different sets of parameters used for lignification. These calculations were performed on desktop hardware over
# # about a week's period, but for the sake of explanation, shorter examples are used here. We investigated the impact
# # of S to G ratio and addition rate primarily.
#
# # ### SG Batch Sensitivity
# # The first analysis performed is the dependence of monomer yields and bond contents on SG ratio, where we selected
# # multiple SG ratios between 0.1 and 10 and ran the simulations for these scenarios.
# sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
# num_repeats = 5
# num_monos = 200
#
# fun = par.delayed(run_kmc)
# sg_result_list = []
#
# for sg_ratio in sg_opts:
#     # Set the percentage of S
#     pct_s = sg_ratio / (1 + sg_ratio)
#
#     # Make choices about what kinds of monomers there are and create them
#     monomer_draw = np.random.rand(num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#
#     # Initialize the monomers, events, and state
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#
#     results = par.Parallel(n_jobs=4)([fun(rxn_rates, initial_state, initial_events, n_max=num_monos, t_max=1)
#                                       for _ in range(num_repeats)])
#
#     sg_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for S to G ratio {:5.2f} on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}'.format(sg_ratio, t.tm_year, t.tm_mon, t.tm_mday,
#                                                             t.tm_hour, t.tm_min, t.tm_sec))
#
# # We now aggregate the results that we obtained in the form of adjacency matrices and monomer lists and times to
# # meaningful values of monomer yields and bond contents.
#
# # noinspection DuplicatedCode
# num_sg_opts = len(sg_opts)
# analysis = []
#
# for i in range(num_sg_opts):
#     sg_results = sg_result_list[i]
#     cur_adjs = [sg_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                            for j in range(num_repeats)))
#     t = time.localtime()
#     print('Finished analysis for S:G of {:5.2f} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(sg_opts[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # Now as an example, we will look at the distribution of monomer yields from RCF (assuming C-O bonds (beta-O4,
# # alpha-O4, and 5-O4 bonds), and only C-O bonds are broken in this process), and then summarize that distribution with
# # the sample mean and standard deviation, which will be used for all of the subsequent analyses. To visualize these
# # results, we first need to extract the information from the analysis data structure, and normalize the data to the
# # total number of monomers or total number of bonds.
# # noinspection DuplicatedCode
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_sg_opts)]
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_sg_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_sg_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
# # With this aggregation complete, we now plot the histograms of beta04 bond yields for 3 cases of SG ratios: 0.1, 1, and 10.
# # With so few repeats, these histograms look very poor, but with greater repeats there is a much clearer trend and
# # approximation of the normal distribution - which justifies the use of sample norm and standard deviation as summary statistics.
# plt.figure(figsize=(3.5, 3.5))
# plt.hist(monomer_yields[0], bins=np.linspace(0, 1.0, 21), density=1, facecolor='black', alpha=0.5,
#          edgecolor='black', label='S/G=0.1')
# plt.hist(monomer_yields[5], bins=np.linspace(0, 1.0, 21), density=1, facecolor='red', alpha=0.5,
#          edgecolor='black', label='S/G=1')
# plt.hist(monomer_yields[10], bins=np.linspace(0, 1.0, 21), density=1, facecolor='blue', alpha=0.5,
#          edgecolor='black', label='S/G=10')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top','right','bottom','left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True,
#                       right=True, width=1.5, length=6)
# plt.xlabel('Monomer Yield (%)', fontsize=14)
# plt.legend(fontsize=14, loc='best')
# plt.savefig('temp_ligpoly_mono_yields.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.close()
#
# # As mentioned above, with more repeats these distributions approach the normal distribution. In addition, if we want
# # to add more sensitivity results to this plot, it starts to look messier and more crowded. Therefore, it becomes much
# # easier to use the mean and variance of our samples as summary statistics, and just plot these values (using the standard deviation as the error bar on every point).
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.savefig('temp_ligpoly_mono_yields_sg.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.close()
#
# # The monomer yields plot above is easy, because there is just one trend that we look at. For this instance, that
# # trend is fairly messy, again because there are fewer repeats and fewer monomers in each simulation, but the idea is still the same.
# #
# # Let us now examine the bond contents that were predicted from this simulation.
# colors = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.6, 0), (0.6, 0, 0.6), (1, 0.549, 0),
#           (0, 0.6, 0.6), (1, 0.8, 0), (0.6078, 0.2980, 0), (0.6, 0, 0), (0, 0, 0.6)]
# plt.figure(figsize=(3.5,3.5))
# plt.errorbar(sg_opts, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(sg_opts, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(sg_opts, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(sg_opts, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(sg_opts, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(sg_opts, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(sg_opts, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
# plt.savefig('temp_ligpoly_bond_types_sg.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.close()
#
#
# # ### Addition rate sensitivity
# # We will now perform the same task on different monomer addition rates.
# add_rates = np.logspace(4, 14, 21)
#
# fun = par.delayed(run_kmc)
# add_result_list = []
#
# # Set the percentage of S
# sg_ratio = 1
# pct_s = sg_ratio / (1 + sg_ratio)
#
# # Set the initial and maximum number of monomers to be modeled.
# ini_num_monos = 4
# max_monos = 200
#
# for add_rate in add_rates:
#     # Initialize the monomers, events, and state
#     monomer_draw = np.random.rand(ini_num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#     initial_events.append(Event(GROW, [], rate=add_rate))
#
#     results = par.Parallel(n_jobs = 4)([fun(rxn_rates, initial_state, initial_events, n_max=max_monos,
#                                             sg_ratio=sg_ratio) for _ in range(num_repeats)])
#
#     add_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for addition rate {:.2e} monomers/s on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}.'.format(add_rate, t.tm_year, t.tm_mon, t.tm_mday,
#                                                              t.tm_hour, t.tm_min, t.tm_sec))
#
# num_opts = len(add_rates)
# analysis = []
#
# for i in range(num_opts):
#     opt_results = add_result_list[i]
#     cur_adjs = [opt_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                                        for j in range(num_repeats)))
#
#     t = time.localtime()
#     print('Finished analysis for monomer addition rate {:5.2e} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(add_rates[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_opts)]
#
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_opts)]
#
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(add_rates, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('Addition Rate (monomers/s)', fontsize=14)
# plt.savefig('temp_ligpoly_monos_add_rates.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.close()
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(add_rates, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(add_rates, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(add_rates, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(add_rates, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(add_rates, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(add_rates, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(add_rates, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('Addition Rate (monomers/s)', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
# plt.savefig('temp_ligpoly_bonds_add_rates.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.close()
#
#
# # ### Continuous monomer addition, SG ratio sensitivity
# # We will now perform the same task on different SG ratios with a slow addition rate.
#
# sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
# num_repeats = 5
# fun = par.delayed(run_kmc)
# sg_result_list = []
#
# for sg_ratio in sg_opts:
#     # Set the percentage of S
#     pct_s = sg_ratio / (1 + sg_ratio)
#
#     # Make choices about what kinds of monomers there are and create them
#     ini_num_monos = 5
#     max_monos = 200
#     monomer_draw = np.random.rand(ini_num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#
#     # Initialize the monomers, events, and state
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#     initial_events.append(Event(GROW, [], rate=1e4))
#
#     results = par.Parallel(n_jobs=4)([fun(rxn_rates, initial_state, initial_events, n_max=max_monos, t_max=1,
#                                           sg_ratio=sg_ratio) for _ in range(num_repeats)])
#
#     sg_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for S to G ratio {:5.2f} on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}'.format(sg_ratio, t.tm_year, t.tm_mon, t.tm_mday,
#                                                             t.tm_hour, t.tm_min, t.tm_sec))
#
# num_sg_opts = len(sg_opts)
# analysis = []
#
# for i in range(num_sg_opts):
#     sg_results = sg_result_list[i]
#     cur_adjs = [sg_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                            for j in range(num_repeats)))
#     t = time.localtime()
#     print('Finished analysis for S:G of {:5.2f} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(sg_opts[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_sg_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_sg_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# fname = 'temp_ligpoly_monos_sg_opts.svg'
# plt.savefig(fname, format='svg', transparent=True, bbox_inches='tight')
# print("wrote:", fname)
# plt.close()
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(sg_opts, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(sg_opts, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(sg_opts, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(sg_opts, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(sg_opts, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(sg_opts, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
# fname = 'temp_ligpoly_bonds_sg_opts.svg'
# plt.savefig(fname, format='svg', transparent=True, bbox_inches='tight')
# print("wrote:", fname)
# plt.close()
#
#
# # ### Continuous monomer addition, SG ratio sensitivity at 1e7 monomers/s
# # We will now perform the same task on different SG ratios with three addition rates
#
# sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
# num_repeats = 5
#
# fun = par.delayed(run_kmc)
# sg_result_list = []
#
# for sg_ratio in sg_opts:
#     # Set the percentage of S
#     pct_s = sg_ratio / (1 + sg_ratio)
#
#     # Make choices about what kinds of monomers there are and create them
#     ini_num_monos = 5
#     max_monos = 200
#     monomer_draw = np.random.rand(ini_num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#
#     # Initialize the monomers, events, and state
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#     initial_events.append(Event(GROW, [], rate=1e7))
#
#     results = par.Parallel(n_jobs=4)([fun(rxn_rates, initial_state, initial_events, n_max=max_monos, t_max=1,
#                                           sg_ratio=sg_ratio) for _ in range(num_repeats)])
#
#     sg_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for SG ratio {:5.2f} on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}.'.format(sg_ratio, t.tm_year, t.tm_mon, t.tm_mday,
#                                                              t.tm_hour, t.tm_min, t.tm_sec))
#
# num_sg_opts = len(sg_opts)
# analysis = []
#
# for i in range(num_sg_opts):
#     sg_results = sg_result_list[i]
#     cur_adjs = [sg_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                            for j in range(num_repeats)))
#     t = time.localtime()
#     print('Finished analysis for S:G of {:5.2f} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(sg_opts[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # In[ ]:
#
#
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_sg_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_sg_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(sg_opts, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(sg_opts, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(sg_opts, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(sg_opts, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(sg_opts, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(sg_opts, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
#
#
# # ### Continuous monomer addition, SG ratio sensitivity 1e9 monomers/second
# #
# # Continuing analysis of different SG ratios at different addition rates
#
# # In[ ]:
#
#
# sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
# num_repeats = 5
#
# fun = par.delayed(run_kmc)
# sg_result_list = []
#
# for sg_ratio in sg_opts:
#     # Set the percentage of S
#     pct_s = sg_ratio / (1 + sg_ratio)
#
#     # Make choices about what kinds of monomers there are and create them
#     ini_num_monos = 5
#     max_monos = 200
#     monomer_draw = np.random.rand(ini_num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#
#     # Initialize the monomers, events, and state
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#     initial_events.append(Event(GROW, [], rate=1e9))
#
#     results = par.Parallel(n_jobs=4)([fun(rxn_rates, initial_state, initial_events, n_max=max_monos, t_max=1,
#                                           sg_ratio=sg_ratio) for _ in range(num_repeats)])
#
#     sg_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for SG ratio {:5.2f} on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}.'.format(sg_ratio, t.tm_year, t.tm_mon, t.tm_mday,
#                                                              t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # In[ ]:
#
#
# num_sg_opts = len(sg_opts)
# analysis = []
#
# for i in range(num_sg_opts):
#     sg_results = sg_result_list[i]
#     cur_adjs = [sg_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                            for j in range(num_repeats)))
#     t = time.localtime()
#     print('Finished analysis for S:G of {:5.2f} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(sg_opts[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # In[ ]:
#
#
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_sg_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_sg_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(sg_opts, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(sg_opts, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(sg_opts, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(sg_opts, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(sg_opts, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(sg_opts, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
#
#
# # ### Continuous monomer addition, SG ratio sensitivity 1e11 monomers/second
# # Continuing analysis of different SG ratios at different addition rates
#
# # In[ ]:
#
#
# sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
# num_repeats = 5
#
# fun = par.delayed(run_kmc)
# sg_result_list = []
#
# for sg_ratio in sg_opts:
#     # Set the percentage of S
#     pct_s = sg_ratio / (1 + sg_ratio)
#
#     # Make choices about what kinds of monomers there are and create them
#     ini_num_monos = 5
#     max_monos = 200
#     monomer_draw = np.random.rand(ini_num_monos)
#     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
#
#     # Initialize the monomers, events, and state
#     initial_events = create_initial_events(initial_monomers, rxn_rates)
#     initial_state = create_initial_state(initial_events, initial_monomers)
#     initial_events.append(Event(GROW, [], rate=1e11))
#
#
#
#     results = par.Parallel(n_jobs=4)([fun(rxn_rates, initial_state, initial_events, n_max=max_monos, t_max=1,
#                                           sg_ratio=sg_ratio) for _ in range(num_repeats)])
#
#     sg_result_list.append(results)
#     t = time.localtime()
#     print('Completed sensitivity iteration for SG ratio {:5.2f} on '
#           '{}-{:02d}-{:02d} at {:02d}:{:02d}:{:02d}.'.format(sg_ratio, t.tm_year, t.tm_mon, t.tm_mday,
#                                                              t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # In[ ]:
#
#
# num_sg_opts = len(sg_opts)
# analysis = []
#
# for i in range(num_sg_opts):
#     sg_results = sg_result_list[i]
#     cur_adjs = [sg_results[j][ADJ_MATRIX] for j in range(num_repeats)]
#     analysis.append(par.Parallel(n_jobs=4)(par.delayed(analyze_adj_matrix)(cur_adjs[j])
#                                            for j in range(num_repeats)))
#     t = time.localtime()
#     print('Finished analysis for S:G of {:5.2f} on {}-{:02d}-{:02d} at '
#           '{:02d}:{:02d}:{:02d}'.format(sg_opts[i], t.tm_year, t.tm_mon, t.tm_mday,
#                                         t.tm_hour, t.tm_min, t.tm_sec))
#
#
# # In[ ]:
#
#
# num_monos = [[sum([analysis[j][i][CHAIN_LEN][k] * k for k in analysis[j][i][CHAIN_LEN]])
#               for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# monomer_yields = [[analysis[j][i][RCF_YIELDS][1]/num_monos[j][i]
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# betaO4_content = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
#                    for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta5_content = [[analysis[j][i][BONDS][B5]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# betabeta_content = [[analysis[j][i][BONDS][BB]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
# beta1_content = [[analysis[j][i][BONDS][B1]/sum(analysis[j][i][BONDS].values())
#                   for i in range(num_repeats)] for j in range(num_sg_opts)]
# c5o4_content = [[analysis[j][i][BONDS][C5O4]/sum(analysis[j][i][BONDS].values())
#                  for i in range(num_repeats)] for j in range(num_sg_opts)]
# alphaO4_content = [[analysis[j][i][BONDS][AO4]/sum(analysis[j][i][BONDS].values())
#                     for i in range(num_repeats)] for j in range(num_sg_opts)]
# fivefive_content = [[analysis[j][i][BONDS][C5C5]/sum(analysis[j][i][BONDS].values())
#                      for i in range(num_repeats)] for j in range(num_sg_opts)]
#
# av_mon_yield = [np.mean(percent) for percent in monomer_yields]
# std_mon_yield = [np.sqrt(np.var(percent)) for percent in monomer_yields]
#
# av_bo4_content = [np.mean(percent) for percent in betaO4_content]
# std_bo4_content = [np.sqrt(np.var(percent)) for percent in betaO4_content]
#
# av_b5_content = [np.mean(percent) for percent in beta5_content]
# std_b5_content = [np.sqrt(np.var(percent)) for percent in beta5_content]
#
# av_bb_content = [np.mean(percent) for percent in betabeta_content]
# std_bb_content = [np.sqrt(np.var(percent)) for percent in betabeta_content]
#
# av_b1_content = [np.mean(percent) for percent in beta1_content]
# std_b1_content = [np.sqrt(np.var(percent)) for percent in beta1_content]
#
# av_5o4_content = [np.mean(percent) for percent in c5o4_content]
# std_5o4_content = [np.sqrt(np.var(percent)) for percent in c5o4_content]
#
# av_ao4_content = [np.mean(percent) for percent in alphaO4_content]
# std_ao4_content = [np.sqrt(np.var(percent)) for percent in alphaO4_content]
#
# av_55_content = [np.mean(percent) for percent in fivefive_content]
# std_55_content = [np.sqrt(np.var(percent)) for percent in fivefive_content]
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_mon_yield, yerr=std_mon_yield, linestyle='none', marker='.', markersize=10,
#              markerfacecolor='black', markeredgecolor='black', capsize=3, ecolor='black')
# plt.xscale('log')
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Monomer Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(3.5, 3.5))
# plt.errorbar(sg_opts, av_bo4_content, yerr=std_bo4_content, linestyle='none', marker='.',
#              markersize=10, markerfacecolor=colors[0], markeredgecolor=colors[0], label=BO4,
#              capsize=3,ecolor=colors[0])
# plt.errorbar(sg_opts, av_bb_content, yerr=std_bb_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[1], markeredgecolor=colors[1], label=BB, capsize=3, ecolor=colors[1])
# plt.errorbar(sg_opts, av_b5_content, yerr=std_b5_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[2], markeredgecolor=colors[2], label=B5, capsize=3, ecolor=colors[2])
# plt.errorbar(sg_opts, av_b1_content, yerr=std_b1_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[3], markeredgecolor=colors[3], label=B1, capsize=3, ecolor=colors[3])
# plt.errorbar(sg_opts, av_5o4_content, yerr=std_5o4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[4], markeredgecolor=colors[4], label=C5O4, capsize=3, ecolor=colors[4])
# plt.errorbar(sg_opts, av_ao4_content, yerr=std_ao4_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[5], markeredgecolor=colors[5], label=AO4, capsize=3, ecolor=colors[5])
# plt.errorbar(sg_opts, av_55_content, yerr=std_55_content, linestyle='none', marker='.', markersize=10,
#              markerfacecolor=colors[6], markeredgecolor=colors[6], label=C5C5, capsize=3, ecolor=colors[6])
# plt.xscale('log')
#
# [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
# plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1.5, length=6)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
#                       width=1, length=4)
# plt.ylabel('Bond Type Yield (%)', fontsize=14)
# plt.xlabel('SG Ratio', fontsize=14)
# plt.ylim([0.0, 1.0])
# plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(1.2, 1.05), frameon=False)
#
#
# # As previously noted, this package can also be run from the command line. See the
# # [README on github](https://github.com/michaelorella/lignin-kmc).

#!/usr/bin/env python3
import logging
import os
import unittest

import joblib as par
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from common_wrangler.common import silent_remove, diff_lines
from rdkit.Chem import MolFromMolBlock
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import MolToFile
from ligninkmc.analysis import (get_bond_type_v_time_dict)
from ligninkmc.create_lignin import (create_initial_monomers,
                                     create_initial_events, create_initial_state)
from ligninkmc.event import Event
from ligninkmc.kmc_common import (C5O4, C5C5, B5, BB, BO4, AO4, B1, GROW, TIME, MONO_LIST, ADJ_MATRIX)
from ligninkmc.kmc_functions import run_kmc
from ligninkmc.visualization import generate_graph_representation
from ligninkmc.visualization import generate_mol, gen_psfgen
from test_lignin_kmc_parts import (create_sample_kmc_result, create_sample_kmc_result_c_lignin, GOOD_RXN_RATES,
                                   get_avg_bo4_bonds)

__author__ = 'hmayes'


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'run_kmc')

# Output files #
PNG_10MER = os.path.join(SUB_DATA_DIR, 'test_10mer.png')
PNG_C_LIGNIN = os.path.join(SUB_DATA_DIR, 'test_c_lignin.png')

TCL_FNAME = "psfgen.tcl"
TCL_FILE_LOC = os.path.join(SUB_DATA_DIR, TCL_FNAME)
GOOD_TCL_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen.tcl")
GOOD_TCL_C_LIGNIN_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_c_lignin.tcl")
GOOD_TCL_SHORT_SIM_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_short_sim.tcl")
GOOD_TCL_NO_GROW_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_no_grow.tcl")


class TestVisualization(unittest.TestCase):
    def testMakePNG(self):
        try:
            silent_remove(PNG_10MER)
            result = create_sample_kmc_result()
            nodes = result[MONO_LIST]
            adj = result[ADJ_MATRIX]
            block = generate_mol(adj, nodes)
            mol = MolFromMolBlock(block)
            Compute2DCoords(mol)
            MolToFile(mol, PNG_10MER, size=(1300, 300))
            self.assertTrue(os.path.isfile(PNG_10MER))
        finally:
            silent_remove(PNG_10MER, disable=DISABLE_REMOVE)

    def testMakePSFGEN(self):
        try:
            silent_remove(TCL_FILE_LOC)
            result = create_sample_kmc_result()
            gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L",
                       toppar_dir='toppar', out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_OUT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testMakePSFGENCLignin(self):
        # Only added one line to coverage... oh well!
        try:
            silent_remove(PNG_C_LIGNIN)
            silent_remove(TCL_FILE_LOC)
            result = create_sample_kmc_result_c_lignin()
            gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", toppar_dir=None,
                       out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_C_LIGNIN_OUT))

            nodes = result[MONO_LIST]
            adj = result[ADJ_MATRIX]
            block = generate_mol(adj, nodes)
            mol = MolFromMolBlock(block)
            Compute2DCoords(mol)
            MolToFile(mol, PNG_C_LIGNIN, size=(1300, 300))
            self.assertTrue(os.path.isfile(PNG_C_LIGNIN))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            silent_remove(PNG_C_LIGNIN, disable=DISABLE_REMOVE)
            pass

    def testNoGrowth(self):
        # Here, all the monomers are available at the beginning of teh simulation
        try:
            sg_ratio = 2.5
            pct_s = sg_ratio / (1 + sg_ratio)
            num_monos = 200
            np.random.seed(10)
            monomer_draw = np.random.rand(num_monos)
            initial_monomers = create_initial_monomers(pct_s, monomer_draw)
            initial_events = create_initial_events(monomer_draw, pct_s, GOOD_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            # since GROW is not added to events, no additional monomers will be added
            result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=2, random_seed=10,
                             sg_ratio=sg_ratio)
            self.assertTrue(len(result[MONO_LIST]) == num_monos)
            gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testMultiProc(self):
        # Note: this test did not increase coverage. Added to help debug notebook; does not need to be
        #    part of test suite
        # Checking how much the joblib parallelization helped: with 200 monos and 4 sg_options, n_jobs=4:
        #     with run_multi: Ran 1 test in 30.875s
        #     without run_multi: Ran 1 test in 85.104s
        run_multi = True
        if run_multi:
            fun = par.delayed(run_kmc)
            num_jobs = 4
        else:
            fun = None
            num_jobs = None
        sg_opts = [0.1, 2.33, 10]
        num_sg_opts = len(sg_opts)
        num_repeats = 4
        num_monos = 40

        sg_result_list = []

        # will add to random seed in the iterations to insure using a different seed for each repeat
        random_seed = 10
        for sg_ratio in sg_opts:
            random_seed += int(sg_ratio)
            # Set the percentage of S
            pct_s = sg_ratio / (1 + sg_ratio)

            # Make choices about what kinds of monomers there are and create them
            # make the seed sg_ratio so doesn't use the same seed for each iteration
            np.random.seed(random_seed)
            monomer_draw = np.random.rand(num_monos)
            initial_monomers = create_initial_monomers(pct_s, monomer_draw)

            # Initialize the monomers, events, and state
            initial_events = create_initial_events(monomer_draw, pct_s, GOOD_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)

            if run_multi:
                results = par.Parallel(n_jobs=num_jobs)([fun(GOOD_RXN_RATES, initial_state, initial_events,
                                                             n_max=num_monos, t_max=1, random_seed=(random_seed + i))
                                                         for i in range(num_repeats)])
            else:
                results = [run_kmc(GOOD_RXN_RATES, initial_state, initial_events, n_max=num_monos, t_max=1,
                                   random_seed=(random_seed + i)) for i in range(num_repeats)]
            sg_result_list.append(results)

        av_bo4_bonds, std_bo4_bonds = get_avg_bo4_bonds(num_sg_opts, sg_result_list, num_repeats, num_jobs)

        good_av_bo4 = [0.369551282051282, 0.6275655354602723, 0.7924836601307189]
        good_std_bo4 = [0.04336646245367748, 0.04148252892416798, 0.03911342876988829]
        self.assertTrue(np.allclose(av_bo4_bonds, good_av_bo4))
        self.assertTrue(np.allclose(std_bo4_bonds, good_std_bo4))

    def testIniRates(self):
        # Note: this test did not increase coverage. Added to help debug notebook; does not need to be
        #    part of test suite
        run_multi = True
        if run_multi:
            fun = par.delayed(run_kmc)
            num_jobs = 4
        else:
            fun = None
            num_jobs = None
        # Set the percentage of S
        sg_ratio = 1.1
        pct_s = sg_ratio / (1 + sg_ratio)

        ini_monos = 2
        max_monos = 32
        num_repeats = 4

        # FYI: np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)[source]
        num_rates = 3
        add_rates = np.logspace(4, 12, num_rates)
        add_rates_result_list = []

        # will add to random seed in the iterations to insure using a different seed for each repeat
        random_seed = 10

        for add_rate in add_rates:
            # Make choices about what kinds of monomers there are and create them
            np.random.seed(random_seed)
            monomer_draw = np.random.rand(ini_monos)
            initial_monomers = create_initial_monomers(pct_s, monomer_draw)

            # Initialize events and state, then add ability to grow
            initial_events = create_initial_events(monomer_draw, pct_s, GOOD_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            initial_events.append(Event(GROW, [], rate=add_rate, bond=sg_ratio))

            if run_multi:
                results = par.Parallel(n_jobs=num_jobs)([fun(GOOD_RXN_RATES, initial_state, initial_events,
                                                             n_max=max_monos, t_max=1, sg_ratio=pct_s,
                                                             random_seed=(random_seed + i))
                                                         for i in range(num_repeats)])
            else:
                results = [run_kmc(GOOD_RXN_RATES, initial_state, initial_events, n_max=max_monos, t_max=1,
                                   sg_ratio=pct_s, random_seed=(random_seed + i)) for i in range(num_repeats)]
            add_rates_result_list.append(results)

        av_bo4_bonds, std_bo4_bonds = get_avg_bo4_bonds(num_rates, add_rates_result_list, num_repeats, num_jobs)

        good_av_bo4 = [0.3519924098671727, 0.15933528836754643, 0.43202764976958524]
        good_std_bo4 = [0.17148021343411038, 0.1168981315424912, 0.25275090383382787]
        self.assertTrue(np.allclose(av_bo4_bonds, good_av_bo4))
        self.assertTrue(np.allclose(std_bo4_bonds, good_std_bo4))

    def testDynamics(self):
        # Tests procedures in the Dynamics.ipynb
        sg_ratio = 1
        pct_s = sg_ratio / (1 + sg_ratio)
        num_monos = 40
        np.random.seed(10)
        monomer_draw = np.random.rand(num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)
        initial_events = create_initial_events(monomer_draw, pct_s, GOOD_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        # since GROW is not added to events, no additional monomers will be added (sg_ratio is thus not needed)
        result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=20, random_seed=10, dynamics=True)
        # With dynamics, the MONO_LIST will be a list of lists:
        #    the inner list is the usual MONO_LIST, but here is it saved for every time step
        expected_num_t_steps = 140
        self.assertTrue(len(result[MONO_LIST]) == expected_num_t_steps)
        self.assertTrue(len(result[MONO_LIST][-1]) == num_monos)

        # Setting up to print: want dict[key: [], ...] where the inner list is values by timestep
        #                      instead of list of timesteps with [[key: val, ...], ... ]
        t_steps = result[TIME]
        adj_list = result[ADJ_MATRIX]
        good_num_timesteps = 140
        self.assertEqual(len(t_steps), good_num_timesteps)

        bond_type_dict, olig_len_dict, sum_list = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=10)

        # test results by checking sums
        good_bond_type_sum_dict = {BO4: 1111, B1: 0, BB: 358, B5: 705, C5C5: 0, AO4: 0, C5O4: 112}
        bond_type_sum_dict = {}
        for bond_type, val_list in bond_type_dict.items():
            self.assertEqual(len(val_list), good_num_timesteps)
            bond_type_sum_dict[bond_type] = sum(val_list)
        self.assertEqual(bond_type_sum_dict, good_bond_type_sum_dict)

        good_olig_len_sum_dict = {1: 2984, 2: 168, 3: 24, 4: 88, 5: 20, 6: 72, 7: 42, 8: 80, 9: 810, 10: 40, 11: 121,
                                  12: 504, 13: 39, 14: 28, 15: 75, 16: 96, 17: 0, 18: 36, 19: 247, 20: 0, 21: 126}
        olig_len_sum_dict = {}
        for olig_len, val_list in olig_len_dict.items():
            self.assertEqual(len(val_list), good_num_timesteps)
            olig_len_sum_dict[olig_len] = sum(val_list)
        self.assertEqual(olig_len_sum_dict, good_olig_len_sum_dict)

        good_sum_sum_list = 1312
        self.assertEqual(sum(sum_list), good_sum_sum_list)

    def testGraphRepresentation(self):
        result = create_sample_kmc_result(max_time=1., num_initial_monos=3, max_monos=10, sg_ratio=1.)
        graph_rep = generate_graph_representation(result[ADJ_MATRIX], result[MONO_LIST])
        pos = nx.nx_agraph.graphviz_layout(graph_rep)
        nodes = graph_rep.nodes()
        nodecolors = [graph_rep.node[n]['t'] for n in nodes]
        edges = graph_rep.edges()
        edgecolors = [graph_rep[u][v]['l'] for u, v in edges]
        ncolors = ['skyblue', 'yellowgreen', 'r']
        labels = ['graph_rep', 'S', 'C']

        for node_color in set(nodecolors):
            node_list = list(np.nonzero(np.array(nodecolors) == node_color)[0])
            nx.draw_networkx_nodes(graph_rep, pos=pos, nodelist=node_list, node_color=ncolors[node_color],
                                   label=labels[node_color], with_labels=True)
        # edgemap = plt.get_cmap('Set1')
        # nodemap = plt.get_cmap('tab10')
        # nx.draw_networkx(graph_rep, pos=pos, edges=edges, node_color=nodecolors, edge_color=edgecolors,
        #               cmap=nodemap, vmin=0, vmax=2, edge_cmap=edgemap, edge_vmin=0, edge_vmax=8, with_labels=True)
        edge_colors = ['k', 'steelblue', 'dodgerblue', 'lime', 'aqua', 'darkorange', 'orange', 'wheat', 'red']
        edge_labels = ['$\\beta$O4', '$\\beta$5', '55', '$\\alpha$O4', '4O5', '$\\beta$1', '$\\beta$1\'',
                       '$\\beta$1\'\'', '$\\beta\\beta$']
        edges = list(edges)
        for edge_color in set(edgecolors):
            edge_list = []
            for edge in (np.nonzero(np.array(edgecolors) == edge_color)[0]):
                edge_list.append(edges[edge])
            plt.plot([],[],color=edge_colors[edge_color],label=edge_labels[edge_color])
            nx.draw_networkx_edges(graph_rep, pos=pos, edgelist=edge_list, edge_color=edge_colors[edge_color],
                                   label=edge_labels[edge_color])
        plt.legend(fontsize=14, frameon=False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("lignindemo.png")
        plt.savefig("lignindemo.svg")

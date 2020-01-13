#!/usr/bin/env python3
import logging
import unittest
from collections import OrderedDict
import joblib as par
import numpy as np
from scipy.sparse import dok_matrix
from ligninkmc import Monomer, Event, run
from ligninkmc.Analysis import (breakBond, countBonds, countYields)
from ligninkmc.KineticMonteCarlo import (G, S, C, MONOMER, OLIGOMER,
                                         C5O4, C5C5, B5, BB, BO4, AO4, B1, OX, Q, B1_ALT, GROW, INT_TO_TYPE_DICT,
                                         MAX_NUM_DECIMAL, ADJ_MATRIX)

# Constants
AFFECTED = 'affected'
MON_MON = (MONOMER, MONOMER)
MON_OLI = (MONOMER, OLIGOMER)
OLI_MON = (OLIGOMER, MONOMER)
OLI_OLI = (OLIGOMER, OLIGOMER)

TIME = 'time'
MONO_LIST = 'monomers'

BONDS = 'Bonds'
CHAIN_LEN = 'Chain Lengths'
RCF_BONDS = 'RCF Bonds'
RCF_YIELDS = 'RCF Yields'

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)


# Data #
SHORT_TIME = 0.00001
ADJ_ZEROS = dok_matrix([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

MONO_DRAW_3 = [0.48772, 0.15174, 0.7886]
MONO_DRAW_20 = [0.48772, 0.15174, 0.7886, 0.48772, 0.15174, 0.7886, 0.48772, 0.15174, 0.7886, 0.48772, 0.15174, 0.7886,
                0.48772, 0.15174, 0.7886, 0.48772, 0.15174, 0.7886, 0.48772, 0.15174]

ADJ2 = dok_matrix([[0, 4, 0, 0, 0],
                   [8, 0, 0, 0, 0],
                   [0, 0, 0, 8, 0],
                   [0, 0, 5, 0, 0],
                   [0, 0, 0, 0, 0]])

ADJ3 = dok_matrix([[0, 4, 8, 0, 0],
                   [8, 0, 0, 0, 0],
                   [5, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

DEF_RXN_RATES = {C5O4: {(G, G): {MON_MON: 38335.5972148372, MON_OLI: 123.419593715543, OLI_MON: 123.419593715543,
                                 OLI_OLI: 3698609451.84164},
                        (S, G): {MON_MON: 63606.8417529500, MON_OLI: 123.419593715543, OLI_MON: 123.419593715543,
                                 OLI_OLI: 3698609451.84164},
                        (C, C): {MON_MON: 11762.4692901771, MON_OLI: 11762.4692901771, OLI_MON: 11762.4692901771,
                                 OLI_OLI: 11762.4692901771}},
                 C5C5: {(G, G): {MON_MON: 4272.63018912086, MON_OLI: 22.8233180720356, OLI_MON: 22.8233180720356,
                                 OLI_OLI: 10182201166.0217},
                        (C, C): {MON_MON: 105537.166803781, MON_OLI: 105537.166803781, OLI_MON: 105537.166803781,
                                 OLI_OLI: 105537.166803781}},
                 B5: {(G, G): {MON_MON: 577740233.381881, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315},
                      (G, S): {MON_MON: 577740233.381881, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315},
                      (C, C): {MON_MON: 251507997491.634, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315}},
                 BB: {(G, G): {MON_MON: 958592907.607318, MON_OLI: 958592907.607318, OLI_MON: 958592907.607318,
                               OLI_OLI: 958592907.607318},
                      (S, G): {MON_MON: 106838377.218107, MON_OLI: 106838377.218107, OLI_MON: 106838377.218107,
                               OLI_OLI: 106838377.218107},
                      (G, S): {MON_MON: 106838377.218107, MON_OLI: 106838377.218107, OLI_MON: 106838377.218107,
                               OLI_OLI: 106838377.218107},
                      (S, S): {MON_MON: 958592907.607318, MON_OLI: 958592907.607318, OLI_MON: 958592907.607318,
                               OLI_OLI: 958592907.607318},
                      (C, C): {MON_MON: 32781102.2219828, MON_OLI: 32781102.2219828, OLI_MON: 32781102.2219828,
                               OLI_OLI: 32781102.2219828}},
                 BO4: {(G, G): {MON_MON: 149736731.431189, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (S, G): {MON_MON: 1327129.87498242, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (G, S): {MON_MON: 1860006.62719604, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (S, S): {MON_MON: 407201.805441432, MON_OLI: 147913.051594236, OLI_MON: 147913.051594236,
                                OLI_OLI: 147913.051594236},
                       (C, C): {MON_MON: 1590507825.87210, MON_OLI: 692396712512.577, OLI_MON: 692396712512.577,
                                OLI_OLI: 692396712512.577}},
                 AO4: {(G, G): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (S, G): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (G, S): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (S, S): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (C, C): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265}},
                 B1: {(G, G): {MON_OLI: 570703.795464849, OLI_MON: 570703.795464849, OLI_OLI: 570703.795464849},
                      (S, G): {MON_OLI: 16485.4030071542, OLI_MON: 16485.4030071542, OLI_OLI: 16485.4030071542},
                      (G, S): {MON_OLI: 89146.6234207596, OLI_MON: 89146.6234207596, OLI_OLI: 89146.6234207596},
                      (S, S): {MON_OLI: 11762.4692901771, OLI_MON: 11762.4692901771, OLI_OLI: 11762.4692901771},
                      (C, C): {MON_OLI: 570703.795464849, OLI_MON: 570703.795464849, OLI_OLI: 570703.795464849}},
                 OX: {G: {MONOMER: 1360057059567.54, OLIGOMER: 149736731.431189},
                      S: {MONOMER: 2256621533195.09, OLIGOMER: 151582896154.443},
                      C: {MONOMER: 1360057059567.54, OLIGOMER: 1360057059567.54}},
                 Q: {G: {MONOMER: 45383.9995564285, OLIGOMER: 45383.9995564285},
                     S: {MONOMER: 16485.4030071542, OLIGOMER: 16485.4030071542},
                     C: {MONOMER: 45383.9995564285, OLIGOMER: 45383.9995564285}}
                 }


def get_avg_num_bonds(bond_type, num_opts, result_list, num_repeats):
    analysis = []
    for i in range(num_opts):
        opt_results = result_list[i]
        cur_adjs = [opt_results[j][ADJ_MATRIX] for j in range(num_repeats)]
        analysis.append([analyze_adj_matrix(cur_adjs[j]) for j in range(num_repeats)])

    num_bonds = [[analysis[j][i][BONDS][bond_type]/sum(analysis[j][i][BONDS].values())
                  for i in range(num_repeats)] for j in range(num_opts)]
    av_num_bonds = [np.mean(bond_pcts) for bond_pcts in num_bonds]
    std_num_bonds = [np.sqrt(np.var(bond_pcts)) for bond_pcts in num_bonds]
    return av_num_bonds, std_num_bonds


def get_avg_num_bonds_single_option(bond_type, result_list, num_repeats):
    analysis = []
    for j in range(num_repeats):
        run_results = result_list[j]
        cur_adjs = run_results[ADJ_MATRIX]
        analysis.append(analyze_adj_matrix(cur_adjs))

    num_bonds = [analysis[j][BONDS][bond_type]/sum(analysis[j][BONDS].values())
                 for j in range(num_repeats)]
    av_num_bonds = np.mean(num_bonds)
    std_num_bonds = np.sqrt(np.var(num_bonds))
    return av_num_bonds, std_num_bonds


def analyze_adj_matrix(adjacency):
    """
    Performs the analysis for a single simulation to extract the relevant macroscopic properties, such as both the
    simulated frequency of different oligomer sizes and the number of each different type of bond before and after in
    silico RCF. The specific code to handle each of these properties is written in the count_bonds(adj) and
    count_oligomer_yields(adj) specifically.

    :param adjacency: scipy dok_matrix  -- the adjacency matrix for the lignin polymer that has been simulated
    :return: A dictionary of results: Chain Lengths, RCF Yields, Bonds, and RCF Bonds
    """

    # Remove any excess b1 bonds from the matrix, e.g. bonds that should be
    # broken during synthesis
    adjacency = breakBond(adjacency, B1_ALT)

    # Examine the initial polymers before any bonds are broken
    counts = countYields(adjacency)
    bond_distributions = countBonds(adjacency)

    # Simulate the RCF process at complete conversion by breaking all of the
    # alkyl C-O bonds that were formed during the reaction
    rcf_adj = breakBond(breakBond(breakBond(adjacency, BO4), AO4), C5O4)

    # Now count the bonds and yields remaining
    rcf_counts = countYields(rcf_adj)
    rcf_bonds = countBonds(rcf_adj)

    return {BONDS: bond_distributions, CHAIN_LEN: counts, RCF_YIELDS: rcf_counts, RCF_BONDS: rcf_bonds}


def create_initial_monomers(pct_s, monomer_draw):
    """
    Make a monomer list (length of monomer_draw) based on the types determined by the monomer_draw list and pct_s
    :param pct_s: float ([0:1]), fraction of  monomers that should be type "S"
    :param monomer_draw: a list of floats ([0:1)) to determine if the monomer should be type "G" (val < pct_s) or
                         "S", otherwise
    :return: list of Monomer objects of specified type
    """
    return [Monomer(INT_TO_TYPE_DICT[int(mono_type_draw < pct_s)], i) for i, mono_type_draw in enumerate(monomer_draw)]


def create_initial_state(initial_events, initial_monomers):
    return {i: {MONOMER: initial_monomers[i], AFFECTED: {initial_events[i]}} for i in range(len(initial_monomers))}


def create_initial_events(initial_monomers, rxn_rates):
    """
    # Create event_dict that will oxidize every monomer
    :param initial_monomers: a list of Monomer objects
    :param rxn_rates: dict of dict of dicts of reaction rates in 1/s
    :return: a list of oxidation Event objects to initialize the state by allowing oxidation of every monomer
    """
    return [Event(OX, [mon.identity], rxn_rates[OX][mon.type][MONOMER]) for mon in initial_monomers]


def create_sample_kmc_result(max_time=1., num_initial_monos=3, max_monos=10, sg_ratio=0.75, seed=10):
    # The set lists are to minimize randomness in testing (adding while debugging source of randomness in some tests;
    #     leaving because it doesn't hurt a thing; also leaving option to make a monomer_draw of arbitrary length
    #     using a seed, but rounding those numbers because the machine precision differences in floats was the bug
    np.random.seed(seed)
    if num_initial_monos == 3:
        monomer_draw = MONO_DRAW_3
    elif num_initial_monos == 20:
        monomer_draw = [0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665, 0.19806286,
                        0.76053071, 0.16911084, 0.08833981, 0.68535982, 0.95339335, 0.00394827, 0.51219226,
                        0.81262096, 0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578]
    else:
        monomer_draw = np.around(np.random.rand(num_initial_monos), MAX_NUM_DECIMAL)

    # these are tested separately elsewhere
    initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
    initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
    initial_state = OrderedDict(create_initial_state(initial_events, initial_monomers))
    initial_events.append(Event(GROW, [], rate=1e4))

    result = run(rates=DEF_RXN_RATES, initialState=initial_state, initialEvents=initial_events,
                 nMax=max_monos, tFinal=max_time, sg_ratio=sg_ratio, random_seed=10)
    return result


# Tests #


# noinspection DuplicatedCode
class TestAnalyzeKMCParts(unittest.TestCase):
    def testCountYieldsAllMonomers(self):
        good_olig_len_dict = {1: 5}
        olig_len_dict = countYields(ADJ_ZEROS)
        self.assertTrue(olig_len_dict == good_olig_len_dict)

    def testCountYields1(self):
        # passed!
        adj_1 = dok_matrix([[0, 4, 0, 0, 0],
                            [8, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        good_olig_len_dict = {1: 3, 2: 1}
        olig_len_dict = countYields(adj_1)
        self.assertTrue(olig_len_dict == good_olig_len_dict)

    def testCountYields2(self):
        # passed!
        olig_len_dict = countYields(ADJ2)
        good_olig_len_dict = {1: 1, 2: 2}
        self.assertTrue(olig_len_dict == good_olig_len_dict)

    def testCountYields3(self):
        olig_len_dict = countYields(ADJ3)
        good_olig_len_dict = {1: 2, 3: 1}
        self.assertTrue(olig_len_dict == good_olig_len_dict)

    def testCountYields4(self):
        adj = dok_matrix((10, 10), dtype=np.float32)
        adj_dict = {(0, 1): 8.0, (1, 0): 8.0, (1, 2): 4.0, (2, 1): 8.0, (2, 3): 4.0, (3, 2): 8.0, (3, 4): 5.0,
                    (4, 3): 8.0, (5, 4): 8.0, (4, 5): 5.0, (5, 6): 4.0, (6, 5): 8.0, (7, 8): 8.0, (8, 7): 8.0,
                    (0, 8): 4.0, (8, 0): 5.0, (8, 9): 4.0, (9, 8): 8.0}
        for key, val in adj_dict.items():
            adj[key] = val
        olig_len_dict = countYields(adj)
        print("Olig_len_dict: ", olig_len_dict)
        # good_olig_len_dict = {10: 1}
        # self.assertTrue(olig_len_dict == good_olig_len_dict)


    def testCountBonds(self):
        good_bond_dict = {BO4: 2, B1: 0, BB: 1, B5: 1, C5C5: 0, AO4: 0, C5O4: 0}
        adj_a = dok_matrix([[0, 8, 0, 0, 0],
                            [4, 0, 8, 0, 0],
                            [0, 5, 0, 8, 0],
                            [0, 0, 8, 0, 4],
                            [0, 0, 0, 8, 0]])
        adj_bonds = countBonds(adj_a)
        self.assertTrue(adj_bonds == good_bond_dict)

    def testBreakBO4Bonds(self):
        good_broken_adj = np.asarray([[0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 8, 0],
                                      [0, 0, 8, 0, 0],
                                      [0, 0, 0, 0, 0]])
        a = dok_matrix((5, 5))
        a[1, 0] = 4
        a[0, 1] = 8
        a[2, 3] = 8
        a[3, 2] = 8
        broken_adj = breakBond(a, BO4).toarray()
        self.assertTrue(np.array_equal(broken_adj, good_broken_adj))

    def testBreakB1_ALTBonds(self):
        good_broken_adj = np.asarray([[0, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 8, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]])
        a = dok_matrix([[0, 4, 0, 0, 0],
                        [8, 0, 1, 0, 0],
                        [0, 8, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
        broken_adj = breakBond(a, B1_ALT).toarray()
        self.assertTrue(np.array_equal(broken_adj, good_broken_adj))


# noinspection DuplicatedCode
class TestVisualization(unittest.TestCase):
    def testCompareVersions(self):
        monomer_types = [[G, S, G, G, S, S, S, G, S, S, G, G, S, G, G, G, G, S, G, G, G, S, G, S, S, S, G, S, S, G, G],
                         [S, S, G, G, S, G, S, G, G, G, G, S, S, S, S, S, G, S, S, S, G, G, S, G, S, G, S, S, G, S, S],
                         [S, S, S, S, G, S, S, G, G, S, G, S, G, G, G, G, S, S, S, S, S, S, S, G, S, S, G, S, G, S, G]]
        num_repeats = len(monomer_types)
        sg_result_list = []

        # will add to random seed in the iterations to insure using a different seed for each repeat
        random_seed = 10
        for i in range(num_repeats):
            # Initialize the monomers, event_dict, and state
            initial_monomers = [Monomer(mono_type, m) for m, mono_type in enumerate(monomer_types[i])]
            num_monos = len(initial_monomers)
            initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            results = run(rates=DEF_RXN_RATES, initialState=initial_state, initialEvents=initial_events,
                          nMax=num_monos, tFinal=1, random_seed=random_seed + i)
            sg_result_list.append(results)

        av_bo4_bonds, std_bo4_bonds = get_avg_num_bonds_single_option(BO4, sg_result_list, num_repeats)
        print("Average fraction BO4 bonds: {:.3f}".format(av_bo4_bonds))
        print("Std dev fraction BO4 bonds: {:.3f}".format(std_bo4_bonds))
        self.assertLess(av_bo4_bonds, .2)

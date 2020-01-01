#!/usr/bin/env python3
import logging
import os
import unittest
from collections import OrderedDict

import joblib as par
import numpy as np
from rdkit.Chem import MolFromMolBlock
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import MolToFile
from scipy.sparse import dok_matrix
from common_wrangler.common import (InvalidDataError, capture_stdout, silent_remove, diff_lines, capture_stderr)
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

# Constants #
DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'run_kmc')

# Output files #
TEST_PNG = os.path.join(SUB_DATA_DIR, 'test.png')
C_LIGNIN_MOL_OUT = os.path.join(SUB_DATA_DIR, 'c_lignin_molfile.txt')
GOOD_C_LIGNIN_MOL_OUT = os.path.join(SUB_DATA_DIR, 'c_lignin_molfile_good.txt')

TCL_FNAME = "psfgen.tcl"
TCL_FILE_LOC = os.path.join(SUB_DATA_DIR, TCL_FNAME)
GOOD_TCL_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen.tcl")
GOOD_TCL_C_LIGNIN_OUT = os.path.join(SUB_DATA_DIR, "good_c_lignin.tcl")
GOOD_TCL_SHORT_SIM_OUT = os.path.join(SUB_DATA_DIR, "good_short_sim.tcl")
GOOD_TCL_NO_GROW_OUT = os.path.join(SUB_DATA_DIR, "good_no_grow.tcl")
GOOD_TCL_SHORT = os.path.join(SUB_DATA_DIR, "good_short.tcl")

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
    result = run_kmc(DEF_RXN_RATES, initial_state, initial_events, n_max=max_monos, t_max=max_time,
                     random_seed=10, sg_ratio=sg_ratio)
    return result


def create_sample_kmc_result_c_lignin(num_monos=2, max_monos=12, seed=10):
    initial_monomers = [Monomer(C, i) for i in range(num_monos)]
    # noinspection PyTypeChecker
    initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
    initial_state = create_initial_state(initial_events, initial_monomers)
    initial_events.append(Event(GROW, [], rate=1e4))
    result = run_kmc(DEF_RXN_RATES, initial_state, sorted(initial_events), n_max=max_monos, t_max=2, random_seed=seed)
    return result


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


# Tests #

class TestCalcRates(unittest.TestCase):
    """
    Tests calculation of rate coefficients by the Eyring equation.
    """
    def testCalcRatesFromKcalMol(self):
        rxn_rates = calc_rates(DEF_TEMP, ea_kcal_mol_dict=DEF_E_BARRIER_KCAL_MOL)
        self.assertTrue(len(rxn_rates) == len(DEF_RXN_RATES))
        rxn_type, substrate, substrate_type = None, None, None  # to make IDE happy
        try:
            for rxn_type in DEF_RXN_RATES:
                for substrate in DEF_RXN_RATES[rxn_type]:
                    for substrate_type in DEF_RXN_RATES[rxn_type][substrate]:
                        self.assertAlmostEqual(DEF_RXN_RATES[rxn_type][substrate][substrate_type],
                                               rxn_rates[rxn_type][substrate][substrate_type])
        except (TypeError, IndexError) as e:
            print(f'{e}\nError when looking at rxn_type: {rxn_type} substrate: {substrate}    '
                  f'substrate_type:    {substrate_type}')


class TestMonomers(unittest.TestCase):
    def testCreateConiferyl(self):
        mon = Monomer(G, 0)  # Makes a guaiacol type monomer with ID = 0
        self.assertTrue(mon.open == {8, 4, 5})
        self.assertTrue(str(mon) == '0: coniferyl alcohol is connected to {0} and active at position 0')

    def testCreateSyringol(self):
        mon = Monomer(S, 2)  # Makes a syringol type monomer with ID = 2
        self.assertTrue(mon.open == {4, 8})
        self.assertTrue(mon.connectedTo == {2})
        self.assertTrue(str(mon) == '2: sinapyl alcohol is connected to {2} and active at position 0')
        self.assertTrue(repr(mon) == '2: sinapyl alcohol \n')

    def testHUnit(self):
        # todo: update once H is added
        try:
            mon = Monomer(H, 2)
            # type type 3 is not currently implemented
            self.assertFalse(mon)  # should not be reached
        except InvalidDataError as e:
            self.assertTrue("only the following" in e.args[0])

    def testUnknownUnit(self):
        try:
            mon = Monomer("@", 2)
            # not a real type
            self.assertFalse(mon)  # should not be reached
        except InvalidDataError as e:
            self.assertTrue("only the following" in e.args[0])

    def testHash(self):
        mon1 = Monomer(S, 5)
        mon2 = Monomer(S, 5)
        check_set = {mon1, mon2}
        self.assertTrue(len(check_set) == 1)


class TestEvent(unittest.TestCase):
    def testIDRepr(self):
        rxn = OX
        # noinspection PyTypeChecker
        event1 = Event(rxn, [2], DEF_RXN_RATES[rxn][G][MONOMER])
        self.assertTrue(str(event1) == "Performing oxidation on index 2")

    def testIDReprBond(self):
        rxn = BO4
        # noinspection PyTypeChecker
        event1 = Event(rxn, [1, 2], DEF_RXN_RATES[rxn][(G, S)][MON_OLI], (4, 8))
        good_str = "Forming bo4 bond between indices [1, 2] (adjacency_matrix update (4, 8))"
        self.assertTrue(str(event1) == good_str)
        self.assertTrue(repr(event1) == good_str)

    def testEventIDHash(self):
        monomer_a = Monomer(S, 4)
        monomer_b = Monomer(S, 4)
        events_a = create_initial_events([monomer_a], DEF_RXN_RATES)
        events_b = create_initial_events([monomer_b], DEF_RXN_RATES)
        self.assertTrue(events_a == events_b)
        check_set = {events_a[0], events_b[0]}
        self.assertTrue(len(check_set) == 1)


class TestCreateInitialMonomers(unittest.TestCase):
    def testCreate3Monomers(self):
        initial_monomers = create_initial_monomers(0.75, [0.48772, 0.15174, 0.7886])
        self.assertTrue(len(initial_monomers) == 3)
        self.assertTrue(initial_monomers[0].type == S)
        self.assertTrue(initial_monomers[1].type == S)
        self.assertTrue(initial_monomers[2].type == G)
        self.assertTrue(initial_monomers[1] < initial_monomers[2])
        self.assertFalse(initial_monomers[0] == initial_monomers[1])


class TestState(unittest.TestCase):
    def testCreateInitialState(self):
        sg_ratio = 0.75
        monomer_draw = [0.48772, 0.15174, 0.7886]
        initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        self.assertTrue(len(initial_state) == 3)
        self.assertTrue(str(initial_monomers[0]) == str(initial_state[0][MONOMER]))


class TestRunKMC(unittest.TestCase):
    def testMissingRequiredSGRatio(self):
        # set up variable to allow running run_kmc without specifying sg_ratio
        initial_sg_ratio = 0.75
        num_initial_monos = 3
        monomer_draw = np.around(np.random.rand(num_initial_monos), MAX_NUM_DECIMAL)
        # these are tested separately
        initial_monomers = create_initial_monomers(initial_sg_ratio, monomer_draw)
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        events = {initial_events[i] for i in range(num_initial_monos)}
        events.add(Event(GROW, [], rate=1e4))
        try:
            run_kmc(DEF_RXN_RATES, initial_state, sorted(events), n_max=20, t_max=1, random_seed=10)
            self.assertFalse("Should not arrive here; An error should have be raised")
        except InvalidDataError as e:
            self.assertTrue("A numeric sg_ratio" in e.args[0])

    def testSampleRunKMC(self):
        result = create_sample_kmc_result()
        self.assertTrue(len(result[TIME]) == 38)
        self.assertAlmostEqual(result[TIME][-1], 0.0022254602430780875)
        self.assertTrue(len(result[MONO_LIST]) == 10)
        self.assertTrue(str(result[MONO_LIST][-1]) == '9: sinapyl alcohol is connected to {8, 9, 7} and active at '
                                                      'position 4')
        good_dok_keys = [(2, 0), (0, 2), (0, 1), (1, 0), (1, 3), (3, 1), (4, 3), (3, 4), (5, 4), (4, 5), (5, 6),
                         (6, 5), (8, 7), (7, 8), (9, 7), (7, 9), ]
        good_dok_vals = [5.0, 8.0, 4.0, 8.0, 4.0, 8.0, 8.0, 5.0, 8.0, 5.0, 4.0, 8.0, 5.0, 8.0, 8.0, 4.0]

        self.assertTrue(list(result[ADJ_MATRIX].keys()) == good_dok_keys)
        self.assertTrue(list(result[ADJ_MATRIX].values()) == good_dok_vals)

    def testSampleRunKMCCLignin(self):
        result = create_sample_kmc_result_c_lignin()
        self.assertTrue(len(result[TIME]) == 45)
        self.assertAlmostEqual(result[TIME][-1], 0.00227415851740560983)
        self.assertTrue(len(result[MONO_LIST]) == 12)
        self.assertTrue(str(result[MONO_LIST][-1]) == '11: caffeoyl alcohol is connected to {0, 1, 2, 3, 4, 5, 6, 7, '
                                                      '8, 9, 10, 11} and active at position 4')
        good_dok_keys = [(0, 1), (1, 0), (2, 1), (1, 2), (2, 3), (3, 2), (4, 3), (3, 4), (4, 5), (5, 4), (5, 6),
                         (6, 5), (6, 7), (7, 6), (7, 8), (8, 7), (9, 8), (8, 9), (10, 9), (9, 10), (11, 10), (10, 11)]
        good_dok_vals = [5.0, 8.0, 8.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 8.0, 4.0,
                         8.0, 4.0, 8.0, 4.0]
        self.assertTrue(list(result[ADJ_MATRIX].keys()) == good_dok_keys)
        self.assertTrue(list(result[ADJ_MATRIX].values()) == good_dok_vals)


class TestAnalyzeKMCParts(unittest.TestCase):
    def testFindOneFragment(self):
        a = dok_matrix((2, 2))
        frags, branches = find_fragments(a)
        good_frags = [{0}, {1}]
        self.assertEqual(frags, good_frags)
        good_branches = [0, 0]
        self.assertEqual(branches, good_branches)

    def testFindTwoFragments(self):
        a_array = [[0., 1., 1., 0., 0.],
                   [1., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 0., 0., 1., 0.]]
        a = dok_matrix(a_array)
        frags, branches = find_fragments(a)
        good_frags = [{0, 1, 2}, {3, 4}]
        self.assertEqual(frags, good_frags)
        good_branches = [0, 0]
        self.assertEqual(branches, good_branches)

    def testFindThreeFragments(self):
        # does not increase coverage, but that's okay
        a = dok_matrix((5, 5))
        a[0, 4] = 1
        a[4, 0] = 1
        frags, branches = find_fragments(a)
        good_frags = [{0, 4}, {1}, {2}, {3}]
        self.assertEqual(frags, good_frags)
        good_branches = [0, 0, 0, 0]
        self.assertEqual(branches, good_branches)

    def testFragmentSize1(self):
        frags = [{0}, {1}]
        result = fragment_size(frags)
        good_result = {0: 1, 1: 1}
        self.assertEqual(result, good_result)

    def testFragmentSize2(self):
        # Does not increase coverage; keep anyway
        frags = [{0, 4, 2}, {1, 3}]
        result = fragment_size(frags)
        good_result = {0: 3, 2: 3, 4: 3, 1: 2, 3: 2}
        self.assertEqual(result, good_result)

    def testFragmentSize3(self):
        frags = [{0, 1, 2, 3, 4}]
        result = fragment_size(frags)
        good_result = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}
        self.assertEqual(result, good_result)

    def testCountYieldsAllMonomers(self):
        good_olig_len_dict = {1: 5}
        good_olig_monos_dict = {1: 5}
        good_olig_branch_dict = {1: 0}
        good_olig_branch_coeff_dict = {1: 0}
        olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(ADJ_ZEROS)
        self.assertTrue(olig_len_dict == good_olig_len_dict)
        self.assertTrue(olig_monos_dict == good_olig_monos_dict)
        self.assertTrue(olig_branch_dict == good_olig_branch_dict)
        self.assertTrue(olig_branch_coeff_dict == good_olig_branch_coeff_dict)

    def testCountYields1(self):
        adj_1 = dok_matrix([[0, 4, 0, 0, 0],
                            [8, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        good_olig_len_dict = {1: 3, 2: 1}
        good_olig_monos_dict = {1: 3, 2: 2}
        good_olig_branch_dict = {1: 0, 2: 0}
        good_olig_branch_coeff_dict = {1: 0, 2: 0}
        olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adj_1)
        self.assertTrue(olig_len_dict == good_olig_len_dict)
        self.assertTrue(olig_monos_dict == good_olig_monos_dict)
        self.assertTrue(olig_branch_dict == good_olig_branch_dict)
        self.assertTrue(olig_branch_coeff_dict == good_olig_branch_coeff_dict)

    def testCountYields2(self):
        olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(ADJ2)
        good_olig_len_dict = {1: 1, 2: 2}
        good_olig_monos_dict = {1: 1, 2: 4}
        good_olig_branch_dict = {1: 0, 2: 0}
        good_olig_branch_coeff_dict = {1: 0, 2: 0}
        self.assertTrue(olig_len_dict == good_olig_len_dict)
        self.assertTrue(olig_monos_dict == good_olig_monos_dict)
        self.assertTrue(olig_branch_dict == good_olig_branch_dict)
        self.assertTrue(olig_branch_coeff_dict == good_olig_branch_coeff_dict)

    def testCountYields3(self):
        olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(ADJ3)
        good_olig_len_dict = {1: 2, 3: 1}
        good_olig_monos_dict = {1: 2, 3: 3}
        good_olig_branch_dict = {1: 0, 3: 0}
        good_olig_branch_coeff_dict = {1: 0, 3: 0}
        self.assertTrue(olig_len_dict == good_olig_len_dict)
        self.assertTrue(olig_monos_dict == good_olig_monos_dict)
        self.assertTrue(olig_branch_dict == good_olig_branch_dict)
        self.assertTrue(olig_branch_coeff_dict == good_olig_branch_coeff_dict)

    def testCountYields4(self):
        adj = dok_matrix((10, 10), dtype=np.float32)
        adj_dict = {(0, 1): 8.0, (1, 0): 8.0, (1, 2): 4.0, (2, 1): 8.0, (2, 3): 4.0, (3, 2): 8.0, (3, 4): 5.0,
                    (4, 3): 8.0, (5, 4): 8.0, (4, 5): 5.0, (5, 6): 4.0, (6, 5): 8.0, (7, 8): 8.0, (8, 7): 8.0,
                    (0, 8): 4.0, (8, 0): 5.0, (8, 9): 4.0, (9, 8): 8.0}
        for key, val in adj_dict.items():
            adj[key] = val
        olig_len_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adj)
        good_olig_len_dict = {10: 1}
        good_olig_monos_dict = {10: 10}
        good_olig_branch_dict = {10: 1}
        good_olig_branch_coeff_dict = {10: 0.1}
        self.assertTrue(olig_len_dict == good_olig_len_dict)
        self.assertTrue(olig_monos_dict == good_olig_monos_dict)
        self.assertTrue(olig_branch_dict == good_olig_branch_dict)
        self.assertTrue(olig_branch_coeff_dict == good_olig_branch_coeff_dict)

    def testCountBonds(self):
        good_bond_dict = {BO4: 2, B1: 0, BB: 1, B5: 1, C5C5: 0, AO4: 0, C5O4: 0}
        adj_a = dok_matrix([[0, 8, 0, 0, 0],
                            [4, 0, 8, 0, 0],
                            [0, 5, 0, 8, 0],
                            [0, 0, 8, 0, 4],
                            [0, 0, 0, 8, 0]])
        adj_bonds = count_bonds(adj_a)
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
        broken_adj = break_bond_type(a, BO4).toarray()
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
        broken_adj = break_bond_type(a, B1_ALT).toarray()
        self.assertTrue(np.array_equal(broken_adj, good_broken_adj))

    def testDegree0(self):
        # Testing all monomers
        adj = ADJ_ZEROS
        mono_degree = list(degree(adj))
        good_mono_degree = []
        self.assertTrue(mono_degree == good_mono_degree)
        branch_coeff = overall_branching_coefficient(adj)
        self.assertTrue(branch_coeff == 0)

    def testBranchCoeff1(self):
        # testing dimers and a monomer
        adj = ADJ2
        mono_degree = list(degree(adj))
        good_mono_degree = [1, 1, 1, 1]
        self.assertTrue(mono_degree == good_mono_degree)
        branch_coeff = overall_branching_coefficient(adj)
        self.assertTrue(branch_coeff == 0)

    def testBranchCoeff2(self):
        # testing timer and dimers
        adj = ADJ3
        branch_degree = list(degree(adj))
        good_branch_degree = [2, 1, 1]
        self.assertTrue(branch_degree == good_branch_degree)
        branch_coeff = overall_branching_coefficient(adj)
        self.assertTrue(branch_coeff == 0)

    def testBranchCoeff3(self):
        adj = dok_matrix((10, 10), dtype=np.float32)
        adj_dict = {(0, 1): 8.0, (1, 0): 8.0, (1, 2): 4.0, (2, 1): 8.0, (2, 3): 4.0, (3, 2): 8.0, (3, 4): 5.0,
                    (4, 3): 8.0, (5, 4): 8.0, (4, 5): 5.0, (5, 6): 4.0, (6, 5): 8.0, (7, 8): 8.0, (8, 7): 8.0,
                    (0, 8): 4.0, (8, 0): 5.0, (8, 9): 4.0, (9, 8): 8.0}
        for key, val in adj_dict.items():
            adj[key] = val
        branch_degree = list(degree(adj))
        good_branch_degree = [2, 2, 2, 2, 2, 2, 1, 1, 3, 1]
        self.assertTrue(branch_degree == good_branch_degree)
        branch_coeff = overall_branching_coefficient(adj)
        good_branch_coeff = 0.1
        self.assertAlmostEqual(branch_coeff, good_branch_coeff)

    def testBranchCoeff4(self):
        # this has 3 fragments: 2 dimers and a pentamer, which is branched
        adj = dok_matrix((9, 9), dtype=np.float32)
        adj_dict = {(4, 8): 8.0, (8, 4): 8.0, (7, 3): 8.0, (3, 7): 8.0, (0, 2): 5.0, (2, 0): 8.0, (4, 2): 4.0,
                    (2, 4): 5.0, (5, 6): 8.0, (6, 5): 5.0, (1, 2): 8.0, (2, 1): 4.0}
        for key, val in adj_dict.items():
            adj[key] = val
        branch_degree = list(degree(adj))
        good_branch_degree = [1, 1, 3, 1, 2, 1, 1, 1, 1]
        self.assertTrue(branch_degree == good_branch_degree)
        branch_coeff = overall_branching_coefficient(adj)
        good_branch_coeff = 1/9
        self.assertAlmostEqual(branch_coeff, good_branch_coeff)
        # Uncomment below to visually check output
        # mol = MolFromMolBlock(block)
        # Compute2DCoords(mol)
        # MolToFile(mol, TEST_PNG, size=(2000, 1000))


class TestAnalyzeKMCSummary(unittest.TestCase):
    def testKMCResultSummary(self):
        result = create_sample_kmc_result()
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        self.assertTrue(summary[CHAIN_LEN] == {3: 1, 7: 1})
        self.assertTrue(summary[BONDS] == {BO4: 4, BB: 0, B5: 4, C5C5: 0, C5O4: 0, AO4: 0, B1: 0})
        self.assertTrue(summary[RCF_YIELDS] == {1: 3, 2: 2, 3: 1})
        self.assertTrue(summary[RCF_BONDS] == {BO4: 0, BB: 0, B5: 4, C5C5: 0, C5O4: 0, AO4: 0, B1: 0})

    def testKMCResultSummaryDescription(self):
        result = create_sample_kmc_result()
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n       1 trimer(s) (chain length 3)\n" \
                             "       1 oligomer(s) of chain length 7, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    4     BB:    0" \
                            "     B5:    4     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n       3 monomer(s) (chain length " \
                                 "1)\n       2 dimer(s) (chain length 2)\n       1 trimer(s) (chain length 3)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     BB:    0    " \
                                " B5:    4     B1:    0    5O4:    0    AO4:    0     55:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testKMCShortSimResultSummaryDescription(self):
        result = create_sample_kmc_result(max_time=SHORT_TIME)
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 3 monomers, which formed:\n" \
                             "       1 trimer(s) (chain length 3)"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    1     BB:    0" \
                            "     B5:    1     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_olig_summary = "Breaking C-O bonds to simulate RCF results in:\n       1 monomer(s) (chain " \
                                "length 1)\n       1 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     BB:    0    " \
                                " B5:    1     B1:    0    5O4:    0    AO4:    0     55:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_olig_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testKMCShortSimManyMonosResultSummaryDescription(self):
        result = create_sample_kmc_result(max_time=SHORT_TIME, num_initial_monos=20, max_monos=40)
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 21 monomers, which formed:\n       " \
                             "1 monomer(s) (chain length 1)\n       8 dimer(s) (chain length 2)\n       " \
                             "1 oligomer(s) of chain length 4, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    0     BB:    6" \
                            "     B5:    4     B1:    0    5O4:    1    AO4:    0     55:    0"
        good_rcf_olig_summary = "Breaking C-O bonds to simulate RCF results in:\n       1 monomer(s) (chain length 1)" \
                                "\n      10 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     BB:    6    " \
                                " B5:    4     B1:    0    5O4:    0    AO4:    0     55:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_olig_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testBO4OligOlig(self):
        # TODO: Use this test to see an instance of beta-o-4 bond formation between oligomers
        # This catches creating this type of linkage, for better understanding linkages that can be created. When done
        # investigating, delete this test and the logic in the code that highlights when this linkage type is created.

        # minimize random calls by providing set list of monomer types
        initial_mono_type_list = [S, S, G, S, S, S, G, S, S, S, G, S, S, G, S, G, S, G, G, S, S, S, S, S, S, S, S,
                                  S, S, S, G, S, G, S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, G, S]
        num_monos = 24
        random_num = 21
        initial_monomers = [Monomer(mono_type, i) for i, mono_type in
                            enumerate(initial_mono_type_list[0:num_monos])]
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        # since GROW is not added to event_dict, no additional monomers will be added
        with capture_stdout(run_kmc, DEF_RXN_RATES, initial_state, sorted(initial_events), t_max=2,
                            random_seed=random_num) as output:
            self.assertTrue("bo4 reaction between oligomers with 16 and 17" in output)
            self.assertTrue("bo4 reaction between oligomers with 14 and 17" in output)


class TestVisualization(unittest.TestCase):
    def testMakePNG(self):
        # smoke test only--that it doesn't fail, not that it looks correct (that's outside the scope of this package)
        # The choices shown resulted (at last check) in 3 fragments, one of which has a branch
        try:
            silent_remove(TEST_PNG)
            result = create_sample_kmc_result(num_initial_monos=9, max_monos=9, seed=4, max_time=SHORT_TIME)
            summary = analyze_adj_matrix(result[ADJ_MATRIX])
            adj_analysis_to_stdout(summary)
            nodes = result[MONO_LIST]
            adj = result[ADJ_MATRIX]
            block = generate_mol(adj, nodes)
            mol = MolFromMolBlock(block)
            Compute2DCoords(mol)
            MolToFile(mol, TEST_PNG, size=(2000, 1200))
            self.assertTrue(os.path.isfile(TEST_PNG))
        finally:
            silent_remove(TEST_PNG, disable=DISABLE_REMOVE)
            pass

    def testMakeTCL(self):
        try:
            silent_remove(TCL_FILE_LOC)
            result = create_sample_kmc_result()
            gen_tcl(result[ADJ_MATRIX], result[MONO_LIST], tcl_fname=TCL_FNAME, chain_id="L",
                    toppar_dir='toppar', out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_OUT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testMakeTCLCLignin(self):
        # Only adds 3 lines to coverage... oh well! At least it's quick.
        try:
            seed = 1
            monos = 7
            silent_remove(TCL_FILE_LOC)
            result = create_sample_kmc_result_c_lignin(num_monos=monos, max_monos=monos*2, seed=seed)
            good_last_time = 0.006328460451357003
            self.assertAlmostEqual(result[TIME][-1], good_last_time)
            gen_tcl(result[ADJ_MATRIX], result[MONO_LIST], tcl_fname=TCL_FNAME, chain_id="L", toppar_dir=None,
                    out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_C_LIGNIN_OUT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testGenMolCLignin(self):
        # Only testing parts of this package; printing block to ease checking output
        try:
            result = create_sample_kmc_result_c_lignin(num_monos=12, max_monos=14)
            block = generate_mol(result[ADJ_MATRIX], result[MONO_LIST])
            with open(C_LIGNIN_MOL_OUT, "w") as f:
                f.write(block)
            self.assertFalse(diff_lines(C_LIGNIN_MOL_OUT, GOOD_C_LIGNIN_MOL_OUT))
            # # Uncomment below to visually check output
            # mol = MolFromMolBlock(block)
            # Compute2DCoords(mol)
            # MolToFile(mol, TEST_PNG, size=(2000, 1000))
        finally:
            # silent_remove(TEST_PNG, disable=DISABLE_REMOVE)
            silent_remove(C_LIGNIN_MOL_OUT, disable=DISABLE_REMOVE)
            pass

    def testB1BondGenMol(self):
        # TODO: Update test as the B1 bond problem is resolved. The variable in this test cause the broken part of the
        #       program to be visited. Currently, the test passes when the expected error message is returned.
        #       When fixed, this test can be updated to pass when expected results are returned.
        # Here, all the monomers are available at the beginning of the simulation; set type list for reproducibility
        full_mono_type_list = [S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, G, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, S, S, S, G, S, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, G, S, S, S, G, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S,
                               S, S, S, S, S, S, G, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, ]
        seed = 41
        num_monos = 92
        try:
            mono_type_list = full_mono_type_list[0: num_monos]
            initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(mono_type_list)]
            initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            result = run_kmc(DEF_RXN_RATES, initial_state, initial_events, t_max=2, random_seed=seed)

            nodes = result[MONO_LIST]
            adj = result[ADJ_MATRIX]
            block = generate_mol(adj, nodes)

            # Here, trying to catch bug in B1 bond representation. Test will be updated when bug is fixed.
            self.assertFalse("I thought I'd fail!")
            # After bug is fixed, add checks for correct generate_mol output
            # Below not needed for testing functionality; for showing image to visually check
            silent_remove(TEST_PNG)
            mol = MolFromMolBlock(block)
            Compute2DCoords(mol)
            MolToFile(mol, TEST_PNG, size=(2000, 1200))
            self.assertTrue(os.path.isfile(TEST_PNG))
            # If desired, also check generated psfgen (may not help coverage... to be seen...)
            gen_tcl(result[ADJ_MATRIX], result[MONO_LIST], tcl_fname=TCL_FNAME, chain_id="L", out_dir=SUB_DATA_DIR)
            # If kept, create and check new "good" file
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
        except InvalidDataError as e:
            print(seed, num_monos)
            print(e.args[0])
            self.assertTrue("This program cannot currently" in e.args[0])
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            silent_remove(TEST_PNG, disable=DISABLE_REMOVE)
            pass

    def testDynamics(self):
        # Tests procedures in the Dynamics.ipynb
        # minimize number of random calls during testing (here, set monomer type distribution)
        monomer_type_list = [G, S, G, G, S, S, S, G, S, S, G, G, S, G, G, G, G, S, G, G, G, S, S, G, S, S, G, G, ]
        num_monos = len(monomer_type_list)
        initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(monomer_type_list)]
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        # since GROW is not added to event_dict, no additional monomers will be added (sg_ratio is thus not needed)
        result = run_kmc(DEF_RXN_RATES, initial_state, sorted(initial_events), random_seed=10, dynamics=True)
        # With dynamics, the MONO_LIST will be a list of monomer lists:
        #    the inner list is the usual MONO_LIST, but here is it saved for every time step
        t_steps = result[TIME]
        expected_num_t_steps = 86
        self.assertEqual(len(t_steps), expected_num_t_steps)
        self.assertTrue(len(result[MONO_LIST]) == expected_num_t_steps)
        self.assertTrue(len(result[MONO_LIST][-1]) == num_monos)
        # want dict[key: [], ...] where the inner list is values by timestep
        #                      instead of list of timesteps with [[key: val, ...], ... ]
        adj_list = result[ADJ_MATRIX]
        (bond_type_dict, olig_len_dict, sum_list, olig_count_dict,
         sum_count_list) = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=10)

        # test results by checking sums
        good_bond_type_sum_dict = {BO4: 95, B1: 0, BB: 274, B5: 292, C5C5: 0, AO4: 0, C5O4: 179}
        bond_type_sum_dict = {}
        for bond_type, val_list in bond_type_dict.items():
            self.assertEqual(len(val_list), expected_num_t_steps)
            bond_type_sum_dict[bond_type] = sum(val_list)
        self.assertEqual(bond_type_sum_dict, good_bond_type_sum_dict)

        good_olig_len_sum_dict = {1: 1133, 2: 686, 3: 84, 4: 72, 5: 0, 6: 48, 7: 0, 8: 72, 9: 0, 10: 100, 11: 187,
                                  12: 0, 13: 26}
        olig_len_sum_dict = {}
        for olig_len, val_list in olig_len_dict.items():
            self.assertEqual(len(val_list), expected_num_t_steps)
            olig_len_sum_dict[olig_len] = sum(val_list)
        self.assertEqual(olig_len_sum_dict, good_olig_len_sum_dict)

        sum_sums = int(sum(sum_list))
        good_sum_sum_list = 313
        self.assertEqual(sum_sums, good_sum_sum_list)

    def testIniRates(self):
        # Note: this test did not increase coverage. Added to help debug notebook.
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

        # minimize random calls
        monomer_type_list = [S, G]
        initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(monomer_type_list)]
        max_monos = 32
        num_repeats = 4
        initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
        # FYI: np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)[source]
        num_rates = 3
        add_rates = np.logspace(4, 12, num_rates)
        add_rates_result_list = []

        # will add to random seed in the iterations to insure using a different seed for each repeat
        random_seed = 2

        for add_rate in add_rates:
            initial_state = create_initial_state(initial_events, initial_monomers)
            initial_events.append(Event(GROW, [], rate=add_rate))
            if run_multi:
                results = par.Parallel(n_jobs=num_jobs)([fun(DEF_RXN_RATES, initial_state, initial_events,
                                                             n_max=max_monos, t_max=1, sg_ratio=pct_s,
                                                             random_seed=(random_seed + i))
                                                         for i in range(num_repeats)])
            else:
                results = [run_kmc(DEF_RXN_RATES, initial_state, initial_events, n_max=max_monos, t_max=1,
                                   sg_ratio=pct_s, random_seed=(random_seed + i)) for i in range(num_repeats)]
            add_rates_result_list.append(results)

        av_bo4_bonds, std_bo4_bonds = get_avg_num_bonds(BO4, num_rates, add_rates_result_list, num_repeats)

        good_av_bo4 = [0.49193548387096775, 0.1822323949687687, 0.08509100150779311]
        good_std_bo4 = [0.026746974115769352, 0.024804184817725252, 0.041353038248119714]
        self.assertTrue(np.allclose(av_bo4_bonds, good_av_bo4))
        self.assertTrue(np.allclose(std_bo4_bonds, good_std_bo4))

    def testNoGrowth(self):
        # Here, all the monomers are available at the beginning of the simulation
        # Increases coverage of gen_tcl
        try:
            # minimize random calls by providing set list of monomer types
            initial_mono_type_list = [S, S, G, S, S, S, G, S, S, S, G, S, S, G, S, G, S, G, G, S, S, S, S, S, S, S, S,
                                      S, S, S, G, S, G, S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, G, S,
                                      S, S, S, S, S, S, S, S, G, S, S, S, S, S, S, S, G, S, S, S, S, S, S, G, G, S, S,
                                      S, S, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, G, S, S, S, S, G, S, S, G, S,
                                      G, S, S, S, S, S, S, S, S, S, S, S, S, G, S, S, S, S, G, S, S, S, S, S, S, S, S,
                                      S, S, S, G, S, S, S, S, S, S, G, S, G, S, S, S, S, S, S, S, S, S, G, S, S, S, S]
            num_monos = 67
            random_num = 202
            initial_monomers = [Monomer(mono_type, i) for i, mono_type in
                                enumerate(initial_mono_type_list[0:num_monos])]
            initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            # since GROW is not added to event_dict, no additional monomers will be added
            result = run_kmc(DEF_RXN_RATES, initial_state, sorted(initial_events), t_max=2,
                             random_seed=random_num)
            # quick tests for run_kmc differences
            self.assertTrue(len(result[TIME]) == 186)
            self.assertAlmostEqual(result[TIME][-1], 0.005550643939956779)
            self.assertTrue(len(result[MONO_LIST]) == num_monos)
            # the function we want to test here is below
            gen_tcl(result[ADJ_MATRIX], result[MONO_LIST], tcl_fname=TCL_FNAME, chain_id="L",
                    out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testTCLTruncateSegname(self):
        # Tests providing a chain_id that is longer than one character
        try:
            # easier to run_kmc to create monomer_list than recreate it here (adj easier) so doing so
            # minimize random calls by providing set list of monomer types
            initial_mono_type_list = [S, S, G, S, S, S, G, S]
            num_monos = len(initial_mono_type_list)
            initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(initial_mono_type_list)]
            initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            # since GROW is not added to event_dict, no additional monomers will be added
            result = run_kmc(DEF_RXN_RATES, initial_state, sorted(initial_events), t_max=2, random_seed=8)
            # quick tests to make sure run_kmc gives expected results (not what we want to test here)
            self.assertAlmostEqual(result[TIME][-1], 0.000766574526703574)
            self.assertTrue(len(result[MONO_LIST]) == num_monos)
            # the function we want to test here is below
            with capture_stderr(gen_tcl, result[ADJ_MATRIX], result[MONO_LIST], chain_id="lignin",
                                out_dir=SUB_DATA_DIR) as output:
                self.assertTrue("should be one character" in output)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_SHORT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testTCLGenEmptySegname(self):
        # tcl_fname="psfgen.tcl", psf_fname='lignin', chain_id="L", toppar_dir="toppar/"
        # Here, all the monomers are available at the beginning of the simulation
        # Increases coverage of gen_tcl
        try:
            # easier to run_kmc to create monomer_list than recreate it here (adj easier) so doing so
            # minimize random calls by providing set list of monomer types
            initial_mono_type_list = [S, S, G, S, S, S, G, S]
            num_monos = len(initial_mono_type_list)
            initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(initial_mono_type_list)]
            initial_events = create_initial_events(initial_monomers, DEF_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            # since GROW is not added to event_dict, no additional monomers will be added
            result = run_kmc(DEF_RXN_RATES, initial_state, sorted(initial_events), t_max=2, random_seed=8)
            # quick tests to make sure run_kmc gives expected results (not what we want to test here)
            self.assertAlmostEqual(result[TIME][-1], 0.000766574526703574)
            self.assertTrue(len(result[MONO_LIST]) == num_monos)
            # the function we want to test here is below
            with capture_stderr(gen_tcl, result[ADJ_MATRIX], result[MONO_LIST], chain_id=" ",
                                out_dir=SUB_DATA_DIR) as output:
                self.assertTrue("should be one character" in output)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_SHORT))
        finally:
            silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
            pass

    def testCheckBO4Fraction(self):
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
            results = run_kmc(DEF_RXN_RATES, initial_state, initial_events,
                              n_max=num_monos, t_max=2, random_seed=random_seed)
            sg_result_list.append(results)

        av_bo4_bonds, std_bo4_bonds = get_avg_num_bonds_single_option(BO4, sg_result_list, num_repeats)
        print("Average fraction BO4 bonds: {:.3f}".format(av_bo4_bonds))
        print("Std dev fraction BO4 bonds: {:.3f}".format(std_bo4_bonds))
        self.assertLess(av_bo4_bonds, .2)
        self.assertTrue(np.allclose(av_bo4_bonds, 0.11142697881828317))
        self.assertTrue(np.allclose(std_bo4_bonds, 0.0444745498074339))

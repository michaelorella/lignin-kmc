#!/usr/bin/env python3
import os
import unittest
import numpy as np
import joblib as par
import logging
from collections import OrderedDict
from scipy.sparse import dok_matrix
from common_wrangler.common import (InvalidDataError, capture_stdout, silent_remove, diff_lines)
from rdkit.Chem import MolFromMolBlock
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import MolToFile
from ligninkmc.create_lignin import (calc_rates, DEF_TEMP, create_initial_monomers,
                                     create_initial_events, create_initial_state, DEF_ADD_RATE,
                                     analyze_adj_matrix, count_bonds, count_oligomer_yields, calc_monos_per_olig,
                                     break_bond_type, adj_analysis_to_stdout, find_fragments, fragment_size,
                                     get_bond_type_v_time_dict)
from ligninkmc.kmc_common import (Event, Monomer, C5O4, OX, Q, C5C5, B5, BB, BO4, AO4, B1,
                                  MON_MON, MON_OLI, OLI_OLI, OLI_MON, MONOMER, OLIGOMER, GROW, TIME, MONO_LIST,
                                  ADJ_MATRIX, CHAIN_LEN, BONDS, RCF_YIELDS, RCF_BONDS, B1_ALT, DEF_E_A_KCAL_MOL,
                                  MAX_NUM_DECIMAL)
from ligninkmc.kmc_functions import run_kmc
from ligninkmc.visualization import (generate_mol, gen_psfgen)

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
PNG_B1 = os.path.join(SUB_DATA_DIR, 'test_b1.png')

TCL_FNAME = "psfgen.tcl"
TCL_FILE_LOC = os.path.join(SUB_DATA_DIR, TCL_FNAME)
GOOD_TCL_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen.tcl")
GOOD_TCL_C_LIGNIN_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_c_lignin.tcl")
GOOD_TCL_SHORT_SIM_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_short_sim.tcl")
GOOD_TCL_NO_GROW_OUT = os.path.join(SUB_DATA_DIR, "good_psfgen_no_grow.tcl")

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

GOOD_RXN_RATES = {C5O4: {(0, 0): {MON_MON: 38335.597214837195, MON_OLI: 123.41959371554347, OLI_MON: 123.41959371554347,
                                  OLI_OLI: 3698609451.841636},
                         (1, 0): {MON_MON: 63606.84175294998, MON_OLI: 123.41959371554347, OLI_MON: 123.41959371554347,
                                  OLI_OLI: 3698609451.841636},
                         (2, 2): {MON_MON: 11762.469290177061, MON_OLI: 11762.469290177061,
                                  OLI_MON: 11762.469290177061, OLI_OLI: 11762.469290177061}},
                  C5C5: {(0, 0): {MON_MON: 4272.630189120858, MON_OLI: 22.82331807203557, OLI_MON: 22.82331807203557,
                                  OLI_OLI: 10182201166.021704},
                         (2, 2): {MON_MON: 105537.16680378099, MON_OLI: 105537.16680378099,
                                  OLI_MON: 105537.16680378099, (OLIGOMER, OLIGOMER): 105537.16680378099}},
                  B5: {(0, 0): {MON_MON: 577740233.3818815, MON_OLI: 348201801.4313151, OLI_MON: 348201801.4313151,
                                OLI_OLI: 348201801.4313151},
                       (0, 1): {MON_MON: 577740233.3818815, MON_OLI: 348201801.4313151, OLI_MON: 348201801.4313151,
                                OLI_OLI: 348201801.4313151},
                       (2, 2): {MON_MON: 251507997491.63364, MON_OLI: 348201801.4313151,
                                OLI_MON: 348201801.4313151, OLI_OLI: 348201801.4313151}},
                  BB: {(0, 0): {MON_MON: 958592907.6073179, MON_OLI: 958592907.6073179, OLI_MON: 958592907.6073179,
                                OLI_OLI: 958592907.6073179},
                       (1, 0): {MON_MON: 106838377.21810664, MON_OLI: 106838377.21810664, OLI_MON: 106838377.21810664,
                                OLI_OLI: 106838377.21810664},
                       (1, 1): {MON_MON: 958592907.6073179, MON_OLI: 958592907.6073179, OLI_MON: 958592907.6073179,
                                OLI_OLI: 958592907.6073179},
                       (0, 1): {MON_MON: 106838377.21810664, MON_OLI: 106838377.21810664, OLI_MON: 106838377.21810664,
                                OLI_OLI: 106838377.21810664},
                       (2, 2): {MON_MON: 32781102.221982773, MON_OLI: 32781102.221982773, OLI_MON: 32781102.221982773,
                                OLI_OLI: 32781102.221982773}},
                  BO4: {(0, 0): {MON_MON: 149736731.43118873, MON_OLI: 177267402.79460046, OLI_MON: 177267402.79460046,
                                 OLI_OLI: 177267402.79460046},
                        (1, 0): {MON_MON: 1327129.8749824178, MON_OLI: 177267402.79460046, OLI_MON: 177267402.79460046,
                                 OLI_OLI: 177267402.79460046},
                        (0, 1): {MON_MON: 1860006.627196039, MON_OLI: 177267402.79460046, OLI_MON: 177267402.79460046,
                                 OLI_OLI: 177267402.79460046},
                        (1, 1): {MON_MON: 407201.805441432, MON_OLI: 147913.05159423634, OLI_MON: 147913.05159423634,
                                 OLI_OLI: 147913.05159423634},
                        (2, 2): {MON_MON: 1590507825.8720958, MON_OLI: 692396712512.5765, OLI_MON: 692396712512.5765,
                                 OLI_OLI: 692396712512.5765}},
                  AO4: {(0, 0): {MON_MON: 0.004169189173972648, MON_OLI: 0.004169189173972648,
                                 OLI_MON: 0.004169189173972648, OLI_OLI: 0.004169189173972648},
                        (1, 0): {MON_MON: 0.004169189173972648, MON_OLI: 0.004169189173972648,
                                 OLI_MON: 0.004169189173972648, OLI_OLI: 0.004169189173972648},
                        (0, 1): {MON_MON: 0.004169189173972648, MON_OLI: 0.004169189173972648,
                                 OLI_MON: 0.004169189173972648, OLI_OLI: 0.004169189173972648},
                        (1, 1): {MON_MON: 0.004169189173972648, MON_OLI: 0.004169189173972648,
                                 OLI_MON: 0.004169189173972648, OLI_OLI: 0.004169189173972648},
                        (2, 2): {MON_MON: 0.004169189173972648, MON_OLI: 0.004169189173972648,
                                 OLI_MON: 0.004169189173972648, OLI_OLI: 0.004169189173972648}},
                  B1: {(0, 0): {MON_OLI: 570703.7954648494, OLI_MON: 570703.7954648494, OLI_OLI: 570703.7954648494},
                       (1, 0): {MON_OLI: 16485.40300715421, OLI_MON: 16485.40300715421, OLI_OLI: 16485.40300715421},
                       (0, 1): {MON_OLI: 89146.62342075957, OLI_MON: 89146.62342075957, OLI_OLI: 89146.62342075957},
                       (1, 1): {MON_OLI: 11762.469290177061, OLI_MON: 11762.469290177061, OLI_OLI: 11762.469290177061},
                       (2, 2): {MON_OLI: 570703.7954648494, OLI_MON: 570703.7954648494, OLI_OLI: 570703.7954648494}},
                  OX: {0: {MONOMER: 1360057059567.5383, OLIGOMER: 149736731.43118873},
                       1: {MONOMER: 2256621533195.0864, OLIGOMER: 151582896154.44305},
                       2: {MONOMER: 1360057059567.5383, OLIGOMER: 1360057059567.5383}},
                  Q: {0: {MONOMER: 45383.99955642849, OLIGOMER: 45383.99955642849},
                      1: {MONOMER: 16485.40300715421, OLIGOMER: 16485.40300715421},
                      2: {MONOMER: 45383.99955642849, OLIGOMER: 45383.99955642849}}}

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


def create_sample_kmc_result(max_time=1., num_initial_monos=3, max_monos=10, sg_ratio=0.75):
    # The set lists are to minimize randomness in testing (adding while debugging source of randomness in some tests;
    #     leaving because it doesn't hurt a thing; also leaving option to make a monomer_draw of arbitrary length
    #     using a seed, but rounding those numbers because the machine precision differences in floats was the bug
    np.random.seed(10)
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
    initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    initial_state = OrderedDict(create_initial_state(initial_events, initial_monomers))
    initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE))
    #            # make random seed and sort event_dict for testing reliability
    result = run_kmc(GOOD_RXN_RATES, initial_state, initial_events, n_max=max_monos, t_max=max_time,
                     random_seed=10, sg_ratio=sg_ratio)
    return result


def create_sample_kmc_result_c_lignin():
    num_monos = 2
    initial_monomers = [Monomer(2, i) for i in range(num_monos)]
    # noinspection PyTypeChecker
    initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    initial_state = create_initial_state(initial_events, initial_monomers)
    initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE))
    result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), n_max=12, t_max=2, random_seed=10)
    return result


def get_avg_bo4_bonds(num_opts, result_list, num_repeats):
    analysis = []
    for i in range(num_opts):
        opt_results = result_list[i]
        cur_adjs = [opt_results[j][ADJ_MATRIX] for j in range(num_repeats)]
        analysis.append([analyze_adj_matrix(cur_adjs[j]) for j in range(num_repeats)])

    bo4_bonds = [[analysis[j][i][BONDS][BO4]/sum(analysis[j][i][BONDS].values())
                  for i in range(num_repeats)] for j in range(num_opts)]
    av_bo4_bonds = [np.mean(bond_pcts) for bond_pcts in bo4_bonds]
    std_bo4_bonds = [np.sqrt(np.var(bond_pcts)) for bond_pcts in bo4_bonds]
    return av_bo4_bonds, std_bo4_bonds


# Tests #

class TestCalcRates(unittest.TestCase):
    """
    Tests calculation of rate coefficients by the Eyring equation.
    """
    def test_calc_rates_from_kcal_mol(self):
        rxn_rates = calc_rates(DEF_TEMP, ea_kcal_mol_dict=DEF_E_A_KCAL_MOL)
        # check size, then values via nested loops instead of dealing with almost equal dicts
        self.assertTrue(len(rxn_rates) == len(GOOD_RXN_RATES))
        rxn_type, substrate, substrate_type = None, None, None  # to make IDE happy
        try:
            for rxn_type in GOOD_RXN_RATES:
                for substrate in GOOD_RXN_RATES[rxn_type]:
                    for substrate_type in GOOD_RXN_RATES[rxn_type][substrate]:
                        self.assertAlmostEqual(GOOD_RXN_RATES[rxn_type][substrate][substrate_type],
                                               rxn_rates[rxn_type][substrate][substrate_type])
        except (TypeError, IndexError) as e:
            print(f'{e}\nError when looking at rxn_type: {rxn_type} substrate: {substrate}    '
                  f'substrate_type:    {substrate_type}')


class TestMonomers(unittest.TestCase):
    def testCreateConiferyl(self):
        mon = Monomer(0, 0)  # Makes a guaiacol type monomer with ID = 0
        self.assertTrue(mon.open == {8, 4, 5})
        self.assertTrue(str(mon) == '0: coniferyl alcohol is connected to {0} and active at position 0')

    def testCreateSyringol(self):
        mon = Monomer(1, 2)  # Makes a syringol type monomer with ID = 2
        self.assertTrue(mon.open == {4, 8})
        self.assertTrue(mon.connectedTo == {2})
        self.assertTrue(str(mon) == '2: sinapyl alcohol is connected to {2} and active at position 0')
        self.assertTrue(repr(mon) == '2: sinapyl alcohol \n')

    def testUnknownUnit(self):
        try:
            mon = Monomer(3, 2)
            # type type 3 is not currently implemented
            self.assertFalse(mon)  # should not be reached
        except InvalidDataError as e:
            self.assertTrue("only the following" in e.args[0])

    def testHash(self):
        mon1 = Monomer(1, 5)
        mon2 = Monomer(1, 5)
        check_set = {mon1, mon2}
        self.assertTrue(len(check_set) == 1)


class TestEvent(unittest.TestCase):
    def testIDRepr(self):
        rxn = OX
        # noinspection PyTypeChecker
        event1 = Event(rxn, [2], GOOD_RXN_RATES[rxn][0][MONOMER])
        self.assertTrue(str(event1) == "Performing oxidation on index 2")

    def testIDReprBond(self):
        rxn = BO4
        # noinspection PyTypeChecker
        event1 = Event(rxn, [1, 2], GOOD_RXN_RATES[rxn][(0, 1)][MON_OLI], (4, 5))
        good_str = "Forming bo4 bond between indices [1, 2] (adjacency_matrix update (4, 5))"
        self.assertTrue(str(event1) == good_str)
        self.assertTrue(repr(event1) == good_str)

    def testEventIDHash(self):
        monomer_a = Monomer(1, 4)
        monomer_b = Monomer(1, 4)
        events_a = create_initial_events([monomer_a], GOOD_RXN_RATES)
        events_b = create_initial_events([monomer_b], GOOD_RXN_RATES)
        self.assertTrue(events_a == events_b)
        check_set = {events_a[0], events_b[0]}
        self.assertTrue(len(check_set) == 1)


class TestCreateInitialMonomers(unittest.TestCase):
    def testInvalidSGRatio(self):
        try:
            create_initial_monomers(None, [0.48772, 0.15174, 0.7886])
            self.assertFalse("Should not arrive here; An error should have be raised")
        except InvalidDataError as e:
            self.assertTrue("None" in e.args[0])

    def testCreate3Monomers(self):
        initial_monomers = create_initial_monomers(0.75, [0.48772, 0.15174, 0.7886])
        self.assertTrue(len(initial_monomers) == 3)
        self.assertTrue(initial_monomers[0].type == 1)
        self.assertTrue(initial_monomers[1].type == 1)
        self.assertTrue(initial_monomers[2].type == 0)
        self.assertTrue(initial_monomers[1] < initial_monomers[2])
        self.assertFalse(initial_monomers[0] == initial_monomers[1])


class TestState(unittest.TestCase):
    def testCreateInitialState(self):
        sg_ratio = 0.75
        monomer_draw = [0.48772, 0.15174, 0.7886]
        initial_monomers = create_initial_monomers(sg_ratio, monomer_draw)
        initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
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
        initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        events = {initial_events[i] for i in range(num_initial_monos)}
        events.add(Event(GROW, [], rate=DEF_ADD_RATE))
        try:
            run_kmc(GOOD_RXN_RATES, initial_state, sorted(events), n_max=20, t_max=1, random_seed=10)
            self.assertFalse("Should not arrive here; An error should have be raised")
        except InvalidDataError as e:
            self.assertTrue("A numeric sg_ratio" in e.args[0])

    def testSampleRunKMC(self):
        result = create_sample_kmc_result()
        self.assertTrue(len(result[TIME]) == 43)
        self.assertAlmostEqual(result[TIME][-1], 0.00019793717541788304)
        self.assertTrue(len(result[MONO_LIST]) == 10)
        self.assertTrue(str(result[MONO_LIST][-1]) == '9: coniferyl alcohol is connected to '
                                                      '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} and active at position -1')
        good_dok_keys = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (4, 5), (5, 4), (6, 7), (7, 6), (8, 9),
                         (9, 8), (9, 2), (2, 9), (3, 5), (5, 3), (7, 9), (9, 7)]
        good_dok_vals = [8.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 5.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0]
        self.assertTrue(list(result[ADJ_MATRIX].keys()) == good_dok_keys)
        self.assertTrue(list(result[ADJ_MATRIX].values()) == good_dok_vals)

    def testSampleRunKMCCLignin(self):
        result = create_sample_kmc_result_c_lignin()
        self.assertTrue(len(result[TIME]) == 45)
        self.assertAlmostEqual(result[TIME][-1], 0.0005193025082191715)
        self.assertTrue(len(result[MONO_LIST]) == 12)
        self.assertTrue(str(result[MONO_LIST][-1]) == '11: caffeoyl alcohol is connected to '
                                                      '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} and active at position 4')
        good_dok_keys = [(1, 0), (0, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6),
                         (6, 5), (6, 7), (7, 6), (7, 8), (8, 7), (8, 9), (9, 8), (10, 9), (9, 10), (10, 11), (11, 10)]
        good_dok_vals = [8.0, 5.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0,
                         8.0, 4.0, 4.0, 8.0]
        self.assertTrue(list(result[ADJ_MATRIX].keys()) == good_dok_keys)
        self.assertTrue(list(result[ADJ_MATRIX].values()) == good_dok_vals)


class TestAnalyzeKMC(unittest.TestCase):
    def testFindOneFragment(self):
        a = dok_matrix((2, 2))
        result = find_fragments(a)
        good_result = [{0}, {1}]
        self.assertEqual(result, good_result)

    def testFindTwoFragments(self):
        a_array = [[0., 1., 1., 0., 0.],
                   [1., 0., 0., 0., 0.],
                   [1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 0., 0., 1., 0.]]
        a = dok_matrix(a_array)
        result = find_fragments(a)
        good_result = [{0, 1, 2}, {3, 4}]
        self.assertEqual(result, good_result)

    def testFindThreeFragments(self):
        # does not increase coverage, but that's okay
        a = dok_matrix((5, 5))
        a[0, 4] = 1
        a[4, 0] = 1
        result = find_fragments(a)
        good_result = [{0, 4}, {1}, {2}, {3}]
        self.assertEqual(result, good_result)

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
        # Does not increase coverage; keep anyway
        frags = [{0, 1, 2, 3, 4}]
        result = fragment_size(frags)
        good_result = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}
        self.assertEqual(result, good_result)

    def testCountYieldsAllMonomers(self):
        good_adj_zeros_dict = {1: 5}
        adj_yields_dict = dict(count_oligomer_yields(ADJ_ZEROS))
        self.assertTrue(adj_yields_dict == good_adj_zeros_dict)

    def testCountYields1(self):
        good_yield_dict = {2: 1, 1: 3}
        adj_1 = dok_matrix([[0, 4, 0, 0, 0],
                            [8, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        adj_yields_dict = dict(count_oligomer_yields(adj_1))
        self.assertTrue(adj_yields_dict == good_yield_dict)

    def testCountYields2(self):
        good_yield_dict = {1: 1, 2: 2}
        adj_yields_dict = dict(count_oligomer_yields(ADJ2))
        self.assertTrue(adj_yields_dict == good_yield_dict)

    def testCountYields3(self):
        good_yield_dict = {3: 1, 1: 2}
        adj_yields_dict = dict(count_oligomer_yields(ADJ3))
        self.assertTrue(adj_yields_dict == good_yield_dict)

    def testCalcMonosPerOlig2(self):
        good_adj_dict = {1: 1, 2: 4}
        olig_monos_dict = dict(calc_monos_per_olig(ADJ2))
        self.assertTrue(olig_monos_dict == good_adj_dict)

    def testCalcMonosPerOlig3(self):
        good_adj_dict = {3: 3, 1: 2}
        olig_monos_dict = dict(calc_monos_per_olig(ADJ3))
        self.assertTrue(olig_monos_dict == good_adj_dict)

    def testCountBonds(self):
        good_bond_dict = {BO4: 2, B1: 0, BB: 1, B5: 1, C5C5: 0, AO4: 0, C5O4: 0}
        adj_a = dok_matrix([[0, 8, 0, 0, 0],
                            [4, 0, 8, 0, 0],
                            [0, 5, 0, 8, 0],
                            [0, 0, 8, 0, 4],
                            [0, 0, 0, 8, 0]])
        adj_bonds = dict(count_bonds(adj_a))
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

    def testKMCResultSummary(self):
        result = create_sample_kmc_result()
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        self.assertTrue(summary[CHAIN_LEN] == {10: 1})
        self.assertTrue(summary[BONDS] == {C5C5: 2, C5O4: 1, AO4: 0, BO4: 4, B1: 0, BB: 1, B5: 1})
        self.assertTrue(summary[RCF_YIELDS] == {1: 3, 2: 2, 3: 1})
        self.assertTrue(summary[RCF_BONDS] == {C5C5: 2, C5O4: 0, AO4: 0, BO4: 0, B1: 0, BB: 1, B5: 1})

    def testKMCResultSummaryDescription(self):
        result = create_sample_kmc_result()
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n       1 oligomer(s) of chain length 10"
        good_bond_summary = "composed of the following bond types and number:\n     55:    2    5O4:    1    " \
                            "AO4:    0     B1:    0     B5:    1     BB:    1    BO4:    4"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n       3 monomer(s) (chain length " \
                                 "1)\n       2 dimer(s) (chain length 2)\n       1 trimer(s) (chain length 3)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n     55:    2    5O4:    0    " \
                                "AO4:    0     B1:    0     B5:    1     BB:    1    BO4:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testKMCShortSimResultSummaryDescription(self):
        result = create_sample_kmc_result(max_time=SHORT_TIME)
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 4 monomers, which formed:\n" \
                             "       1 monomer(s) (chain length 1)\n       1 trimer(s) (chain length 3)"
        good_bond_summary = "composed of the following bond types and number:\n     55:    0    5O4:    0    " \
                            "AO4:    0     B1:    0     B5:    0     BB:    1    BO4:    1"
        good_rcf_olig_summary = "Breaking C-O bonds to simulate RCF results in:\n       2 monomer(s) (chain " \
                                "length 1)\n       1 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n     55:    0    5O4:    0    " \
                                "AO4:    0     B1:    0     B5:    0     BB:    1    BO4:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_olig_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testKMCShortSimManyMonosResultSummaryDescription(self):
        result = create_sample_kmc_result(max_time=SHORT_TIME, num_initial_monos=20, max_monos=40)
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        # adj_analysis_to_stdout(summary)
        good_chain_summary = "Lignin KMC created 21 monomers, which formed:\n       3 monomer(s) (chain length 1)\n" \
                             "       3 dimer(s) (chain length 2)\n       2 oligomer(s) of chain length 6"
        good_bond_summary = "composed of the following bond types and number:\n     55:    0    5O4:    2    " \
                            "AO4:    0     B1:    0     B5:    2     BB:    6    BO4:    3"
        good_rcf_olig_summary = "Breaking C-O bonds to simulate RCF results in:\n       6 monomer(s) (chain length 1)" \
                                "\n       6 dimer(s) (chain length 2)\n       1 trimer(s) (chain length 3)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n     55:    0    5O4:    0    " \
                                "AO4:    0     B1:    0     B5:    2     BB:    6    BO4:    0"
        with capture_stdout(adj_analysis_to_stdout, summary) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_olig_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)


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

    # def testMakePSFGENCLignin(self):
    #     # Only added one line to coverage... oh well!
    #     try:
    #         silent_remove(PNG_C_LIGNIN)
    #         silent_remove(TCL_FILE_LOC)
    #         result = create_sample_kmc_result_c_lignin()
    #         gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", toppar_dir=None,
    #                    out_dir=SUB_DATA_DIR)
    #         self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_C_LIGNIN_OUT))
    #
    #         nodes = result[MONO_LIST]
    #         adj = result[ADJ_MATRIX]
    #         block = generate_mol(adj, nodes)
    #         mol = MolFromMolBlock(block)
    #         Compute2DCoords(mol)
    #         MolToFile(mol, PNG_C_LIGNIN, size=(1300, 300))
    #         self.assertTrue(os.path.isfile(PNG_C_LIGNIN))
    #     finally:
    #         silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         silent_remove(PNG_C_LIGNIN, disable=DISABLE_REMOVE)
    #         pass

    # def testFishingForB1Bond(self):
    #     try:
    #         for sg_ratio in [0.1, 1., 5., 10.]:
    #             ini_num_monos = 200
    #             max_num_monos = 400
    #         pct_s = sg_ratio / (1 + sg_ratio)
    #         # num_monos = 200
    #         np.random.seed(1)
    #         monomer_draw = np.around(np.random.rand(ini_num_monos), MAX_NUM_DECIMAL)
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE, bond=sg_ratio))
    #         result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=0.02, random_seed=1,
    #                          sg_ratio=sg_ratio, n_max=max_num_monos)
    #         adj_result = analyze_adj_matrix(result[ADJ_MATRIX])
    #         if adj_result[BONDS][B1] > 0:
    #             print(f"Woot! sg{sg_ratio}")
    #         # self.assertTrue(len(result[MONO_LIST]) == num_monos)
    #         # gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
    #         # self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
    #     finally:
    #         # silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         pass

    def testB1BondGenMol(self):
        # Here, all the monomers are available at the beginning of the simulation
        try:
            sg_ratio = 10.
            pct_s = sg_ratio / (1 + sg_ratio)
            np.random.seed(1)
            monomer_draw = [0.417022004702574, 0.7203244934421581, 0.00011437481734488664, 0.30233257263183977,
                            0.14675589081711304, 0.0923385947687978, 0.1862602113776709, 0.34556072704304774,
                            0.39676747423066994, 0.538816734003357, 0.4191945144032948, 0.6852195003967595,
                            0.20445224973151743, 0.8781174363909454, 0.027387593197926163, 0.6704675101784022,
                            0.41730480236712697, 0.5586898284457517, 0.14038693859523377, 0.1981014890848788,
                            0.8007445686755367, 0.9682615757193975, 0.31342417815924284, 0.6923226156693141,
                            0.8763891522960383, 0.8946066635038473, 0.08504421136977791, 0.03905478323288236,
                            0.1698304195645689, 0.8781425034294131, 0.0983468338330501, 0.42110762500505217,
                            0.9578895301505019, 0.5331652849730171, 0.6918771139504734, 0.31551563100606295,
                            0.6865009276815837, 0.8346256718973729, 0.018288277344191806, 0.7501443149449675,
                            0.9888610889064947, 0.7481656543798394, 0.2804439920644052, 0.7892793284514885,
                            0.10322600657764203, 0.44789352617590517, 0.9085955030930956, 0.2936141483736795,
                            0.28777533858634874, 0.13002857211827767, 0.019366957870297075, 0.678835532939891,
                            0.21162811600005904, 0.2655466593722262, 0.4915731592803383, 0.053362545117080384,
                            0.5741176054920131, 0.14672857490581015, 0.5893055369032842, 0.6997583600209312,
                            0.10233442882782584, 0.4140559878195683, 0.6944001577277451, 0.41417926952690265,
                            0.04995345894608716, 0.5358964059155116, 0.6637946452197888, 0.5148891120583086,
                            0.9445947559908133, 0.5865550405019929, 0.9034019152878835, 0.13747470414623753,
                            0.13927634725075855, 0.8073912887095238, 0.3976768369855336, 0.16535419711693278,
                            0.9275085803960339, 0.34776585974550656, 0.7508121031361555, 0.7259979853504515,
                            0.8833060912058098, 0.6236722070556089, 0.7509424340273372, 0.34889834197784253,
                            0.2699278917650261, 0.8958862181960668, 0.4280911898712949, 0.9648400471483856,
                            0.6634414978184481, 0.6216957202091218, 0.11474597295337519, 0.9494892587070712,
                            0.4499121334799405, 0.5783896143871318, 0.40813680276128117, 0.2370269802430277,
                            0.9033795205622538, 0.5736794866722859, 0.00287032703115897, 0.6171449136207239,
                            0.32664490177209615, 0.5270581022576093, 0.8859420993107745, 0.35726976000249977,
                            0.9085351509197992, 0.6233601157918027, 0.015821242846556283, 0.9294372337437613,
                            0.690896917516924, 0.9973228504514805, 0.17234050834532855, 0.13713574962887776,
                            0.9325954630371636, 0.6968181614899002, 0.06600017272206249, 0.7554630526024664,
                            0.7538761884612464, 0.9230245355464833, 0.7115247586284718, 0.1242709619721647,
                            0.01988013383979559, 0.026210986877719278, 0.028306488020794607, 0.2462110676030459,
                            0.860027948682888, 0.5388310643416528, 0.5528219786857659, 0.8420308923596057,
                            0.12417331511991114, 0.2791836790111395, 0.5857592714582879, 0.9695957483196745,
                            0.56103021925571, 0.01864728937294302, 0.8006326726806163, 0.23297427384102043,
                            0.8071051956187791, 0.38786064406417176, 0.8635418545594287, 0.7471216427371846,
                            0.5562402339904189, 0.13645522566068502, 0.05991768951221166, 0.12134345574073735,
                            0.044551878544761725, 0.1074941291060929, 0.2257093386078547, 0.7129889803826767,
                            0.5597169820541424, 0.012555980159115854, 0.07197427968948678, 0.967276330000272,
                            0.5681004619199421, 0.20329323466099047, 0.2523257445703234, 0.7438258540750929,
                            0.1954294811093188, 0.5813589272732578, 0.9700199890883123, 0.8468288014900353,
                            0.23984775914758616, 0.49376971426872995, 0.6199557183813798, 0.8289808995501787,
                            0.15679139464608427, 0.018576202177409518, 0.07002214371922233, 0.4863451109370318,
                            0.6063294616533303, 0.5688514370864813, 0.31736240932216075, 0.9886161544124489,
                            0.5797452192457969, 0.3801411726235504, 0.5509482191178968, 0.7453344309065021,
                            0.6692328934531846, 0.2649195576628094, 0.06633483442844157, 0.3700841979141063,
                            0.6297175070215645, 0.2101740099148396, 0.7527555537388139, 0.06653648135411494,
                            0.26031509857854096, 0.8047545637433454, 0.19343428262332774, 0.6394608808799401,
                            0.5246703091237337, 0.9248079703993507, 0.263296770487111, 0.06596109068402378,
                            0.7350659632886695, 0.7721780295432468, 0.907815852503524, 0.9319720691968373,
                            0.013951572975597015, 0.2343620861214205, 0.6167783570016576, 0.9490163206876164]
            num_monos = len(monomer_draw)
            initial_monomers = create_initial_monomers(pct_s, monomer_draw)
            initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
            # initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE, bond=sg_ratio))
            result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=0.02, random_seed=1,
                             sg_ratio=sg_ratio)

            silent_remove(PNG_B1)
            nodes = result[MONO_LIST]
            adj = result[ADJ_MATRIX]
            block = generate_mol(adj, nodes)
            self.assertFalse("I thought I'd fail!")
            mol = MolFromMolBlock(block)
            Compute2DCoords(mol)
            MolToFile(mol, PNG_B1, size=(1300, 300))
            self.assertTrue(os.path.isfile(PNG_B1))

            self.assertTrue(len(result[MONO_LIST]) == num_monos)
            gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
            self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
        except InvalidDataError as e:
            print(e.args[0])
            self.assertTrue("This program cannot currently display" in e.args[0])
            silent_remove(PNG_B1, disable=DISABLE_REMOVE)
            pass

    # def testB1BondGenPSF(self):
    #     # Here, all the monomers are available at the beginning of teh simulation
    #     try:
    #         sg_ratio = 10.
    #         pct_s = sg_ratio / (1 + sg_ratio)
    #         num_monos = 200
    #         np.random.seed(1)
    #         monomer_draw = np.around(np.random.rand(num_monos), MAX_NUM_DECIMAL)
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         # initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE, bond=sg_ratio))
    #         result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=0.02, random_seed=1,
    #                          sg_ratio=sg_ratio)
    #         adj_result = analyze_adj_matrix(result[ADJ_MATRIX])
    #         if adj_result[BONDS][B1] > 0:
    #             print(f"Woot! sg={sg_ratio}")
    #         gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
    #         # self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
    #     except InvalidDataError as e:
    #         print(e.args[0])
    #         self.assertTrue("This program cannot currently display" in e.args[0])
    #         silent_remove(PNG_B1, disable=DISABLE_REMOVE)
    #         pass

    # TODO: figure out what these use that the others do not
    # def testDynamics(self):
    #     # Tests procedures in the Dynamics.ipynb
    #     sg_ratio = 1
    #     pct_s = sg_ratio / (1 + sg_ratio)
    #     # minimize number of random calls during testing
    #     monomer_draw = [0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665,
    #                     0.19806286, 0.76053071, 0.16911084, 0.08833981, 0.68535982, 0.95339335,
    #                     0.00394827, 0.51219226, 0.81262096, 0.61252607, 0.72175532, 0.29187607,
    #                     0.91777412, 0.71457578, 0.54254437, 0.14217005, 0.37334076, 0.67413362,
    #                     0.44183317, 0.43401399, 0.61776698, 0.51313824, 0.65039718, 0.60103895,
    #                     0.80522320, 0.52164715, 0.90864888, 0.31923609, 0.09045935, 0.30070006,
    #                     0.11398436, 0.82868133, 0.04689632, 0.62628715]
    #     num_monos = len(monomer_draw)
    #     initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #     initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #     initial_state = create_initial_state(initial_events, initial_monomers)
    #     # since GROW is not added to event_dict, no additional monomers will be added (sg_ratio is thus not needed)
    #     result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=20, random_seed=10, dynamics=True)
    #     # With dynamics, the MONO_LIST will be a list of monomer lists:
    #     #    the inner list is the usual MONO_LIST, but here is it saved for every time step
    #     expected_num_t_steps = 145
    #     self.assertTrue(len(result[MONO_LIST]) == expected_num_t_steps)
    #     self.assertTrue(len(result[MONO_LIST][-1]) == num_monos)
    #
    #     # Setting up to print: want dict[key: [], ...] where the inner list is values by timestep
    #     #                      instead of list of timesteps with [[key: val, ...], ... ]
    #     t_steps = result[TIME]
    #     adj_list = result[ADJ_MATRIX]
    #     self.assertEqual(len(t_steps), expected_num_t_steps)
    #
    #     bond_type_dict, olig_len_dict, sum_list = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=10)
    #
    #     # test results by checking sums
    #     good_bond_type_sum_dict = {BO4: 1334, B1: 0, BB: 378, B5: 582, C5C5: 0, AO4: 0, C5O4: 145}
    #     bond_type_sum_dict = {}
    #     for bond_type, val_list in bond_type_dict.items():
    #         self.assertEqual(len(val_list), expected_num_t_steps)
    #         bond_type_sum_dict[bond_type] = sum(val_list)
    #     self.assertEqual(bond_type_sum_dict, good_bond_type_sum_dict)
    #
    #     good_olig_len_sum_dict = {1: 3039, 2: 172, 3: 24, 4: 116, 5: 25, 6: 72, 7: 91, 8: 80, 9: 819, 10: 70, 11: 88,
    #                               12: 24, 13: 78, 14: 28, 15: 90, 16: 192, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0,
    #                               23: 0, 24: 0, 25: 0, 26: 52, 27: 135, 28: 0, 29: 58, 30: 90, 31: 217, 32: 0, 33: 0,
    #                               34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 240}
    #     olig_len_sum_dict = {}
    #     for olig_len, val_list in olig_len_dict.items():
    #         self.assertEqual(len(val_list), expected_num_t_steps)
    #         olig_len_sum_dict[olig_len] = sum(val_list)
    #     self.assertEqual(olig_len_sum_dict, good_olig_len_sum_dict)
    #
    #     good_sum_sum_list = 1362
    #     self.assertEqual(sum(sum_list), good_sum_sum_list)
    #
    # def testIniRates(self):
    #     # Note: this test did not increase coverage. Added to help debug notebook; does not need to be
    #     #    part of test suite
    #     run_multi = True
    #     if run_multi:
    #         fun = par.delayed(run_kmc)
    #         num_jobs = 4
    #     else:
    #         fun = None
    #         num_jobs = None
    #     # Set the percentage of S
    #     sg_ratio = 1.1
    #     pct_s = sg_ratio / (1 + sg_ratio)
    #
    #     # minimize random calls
    #     monomer_draw = [0.0207519493594015, 0.771320643266746]
    #     max_monos = 32
    #     num_repeats = 4
    #
    #     # FYI: np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)[source]
    #     num_rates = 3
    #     add_rates = np.logspace(4, 12, num_rates)
    #     add_rates_result_list = []
    #
    #     # will add to random seed in the iterations to insure using a different seed for each repeat
    #     random_seed = 10
    #
    #     for add_rate in add_rates:
    #         # Make choices about what kinds of monomers there are and create them
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #
    #         # Initialize event_dict and state, then add ability to grow
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         initial_events.append(Event(GROW, [], rate=add_rate, bond=sg_ratio))
    #
    #         if run_multi:
    #             results = par.Parallel(n_jobs=num_jobs)([fun(GOOD_RXN_RATES, initial_state, initial_events,
    #                                                          n_max=max_monos, t_max=1, sg_ratio=pct_s,
    #                                                          random_seed=(random_seed + i))
    #                                                      for i in range(num_repeats)])
    #         else:
    #             results = [run_kmc(GOOD_RXN_RATES, initial_state, initial_events, n_max=max_monos, t_max=1,
    #                                sg_ratio=pct_s, random_seed=(random_seed + i)) for i in range(num_repeats)]
    #         add_rates_result_list.append(results)
    #
    #     av_bo4_bonds, std_bo4_bonds = get_avg_bo4_bonds(num_rates, add_rates_result_list, num_repeats)
    #
    #     good_av_bo4 = [0.27277039848197343, 0.12903225806451613, 0.5397801123607575]
    #     good_std_bo4 = [0.24424567669648262, 0.1368593770038479, 0.2756883608483714]
    #     self.assertTrue(np.allclose(av_bo4_bonds, good_av_bo4))
    #     self.assertTrue(np.allclose(std_bo4_bonds, good_std_bo4))

    # def testMultiProc(self):
    #     # Note: this test did not increase coverage. Added to help debug notebook; does not need to be
    #     #    part of test suite
    #     # Checking how much the joblib parallelization helped: with 200 monos and 4 sg_options, n_jobs=4:
    #     #     with run_multi: Ran 1 test in 30.875s
    #     #     without run_multi: Ran 1 test in 85.104s
    #     run_multi = True
    #     if run_multi:
    #         fun = par.delayed(run_kmc)
    #         num_jobs = 4
    #     else:
    #         fun = None
    #         num_jobs = None
    #     sg_opts = [0.1, 2.33, 10]
    #     num_sg_opts = len(sg_opts)
    #     num_repeats = 4
    #     # minimize random numbers
    #     monomer_draw = [0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665,
    #                     0.19806286, 0.76053071, 0.16911084, 0.08833981, 0.68535982, 0.95339335,
    #                     0.00394827, 0.51219226, 0.81262096, 0.61252607, 0.72175532, 0.29187607,
    #                     0.91777412, 0.71457578, 0.54254437, 0.14217005, 0.37334076]
    #     num_monos = len(monomer_draw)
    #
    #     sg_result_list = []
    #
    #     # will add to random seed in the iterations to insure using a different seed for each repeat
    #     random_seed = 10
    #     for sg_ratio in sg_opts:
    #         # Set the percentage of S
    #         pct_s = sg_ratio / (1 + sg_ratio)
    #
    #         # Make choices about what kinds of monomers there are and create them
    #         # make the seed sg_ratio so doesn't use the same seed for each iteration
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #
    #         # Initialize the monomers, event_dict, and state
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #
    #         if run_multi:
    #             results = par.Parallel(n_jobs=num_jobs)([fun(GOOD_RXN_RATES, initial_state, initial_events,
    #                                                          n_max=num_monos, t_max=1, random_seed=(random_seed + i))
    #                                                      for i in range(num_repeats)])
    #         else:
    #             results = [run_kmc(GOOD_RXN_RATES, initial_state, initial_events, n_max=num_monos, t_max=1,
    #                                random_seed=(random_seed + i)) for i in range(num_repeats)]
    #         sg_result_list.append(results)
    #
    #     av_bo4_bonds, std_bo4_bonds = get_avg_bo4_bonds(num_sg_opts, sg_result_list, num_repeats)
    #
    #     good_av_bo4 = [0.29545454545454547, 0.4523268398268398, 0.6134599673202614]
    #     good_std_bo4 = [0.08194434716963613, 0.01691248901891003, 0.10612363599574895]
    #     self.assertTrue(np.allclose(av_bo4_bonds, good_av_bo4))
    #     self.assertTrue(np.allclose(std_bo4_bonds, good_std_bo4))
    #
    # def testNoGrowth(self):
    #     # Here, all the monomers are available at the beginning of the simulation
    #     try:
    #         sg_ratio = 2.5
    #         pct_s = sg_ratio / (1 + sg_ratio)
    #         # minimize random calls
    #         monomer_draw = [0.0207519493594015, 0.6336482349262754, 0.7488038825386119, 0.4985070123025904,
    #                         0.22479664553084766, 0.19806286475962398, 0.7605307121989587, 0.16911083656253545,
    #                         0.08833981417401027, 0.6853598183677972, 0.9533933461949365, 0.003948266327914451,
    #                         0.5121922633857766, 0.8126209616521135, 0.6125260668293881, 0.7217553174317995,
    #                         0.29187606817063316, 0.9177741225129434, 0.7145757833976906, 0.5425443680112613,
    #                         0.14217004760152696, 0.3733407600514692, 0.6741336150663453, 0.4418331744229961,
    #                         0.4340139933332937, 0.6177669784693172, 0.5131382425543909, 0.6503971819314672,
    #                         0.6010389534045444, 0.8052231968327465, 0.5216471523936341, 0.9086488808086682,
    #                         0.3192360889885453, 0.09045934927090737, 0.30070005663620336, 0.11398436186354977,
    #                         0.8286813263076767, 0.04689631938924976, 0.6262871483113925, 0.5475861559192435,
    #                         0.8192869956700687, 0.1989475396788123, 0.8568503024577332, 0.3516526394320879,
    #                         0.7546476915298572, 0.2959617068796787, 0.8839364795611863, 0.3255116378322488,
    #                         0.16501589771914849, 0.3925292439465873, 0.0934603745586503, 0.8211056578369285,
    #                         0.15115201964256386, 0.3841144486921996, 0.9442607122388011, 0.9876254749018722,
    #                         0.4563045470947841, 0.8261228438427398, 0.25137413420705934, 0.5973716482308843,
    #                         0.9028317603316274, 0.5345579488018151, 0.5902013629854229, 0.03928176722538734,
    #                         0.3571817586345363, 0.07961309015596418, 0.30545991834281827, 0.330719311982132,
    #                         0.7738302962105958, 0.039959208689977266, 0.42949217843163834, 0.3149268718426883,
    #                         0.6364911430675446, 0.34634715008003303, 0.04309735620499444, 0.879915174517916,
    #                         0.763240587143681, 0.8780966427248583, 0.41750914383926696, 0.6055775643937568,
    #                         0.5134666274082884, 0.5978366479629736, 0.2622156611319503, 0.30087130894070724,
    #                         0.025399782050106068, 0.30306256065103476, 0.24207587540352737, 0.5575781886626442,
    #                         0.5655070198881675, 0.47513224741505056, 0.2927979762895091, 0.06425106069482445,
    #                         0.9788191457576426, 0.33970784363786366, 0.4950486308824543, 0.9770807259226818,
    #                         0.4407738249006665, 0.3182728054789512, 0.5197969858753801, 0.5781364298824675,
    #                         0.8539337505004864, 0.06809727353795003, 0.46453080777933253, 0.7819491186191484,
    #                         0.7186028103822503, 0.5860219800531759, 0.037094413234407875, 0.350656391283133,
    #                         0.563190684492745, 0.29972987242456284, 0.5123341532735493, 0.6734669252847205,
    #                         0.1591937333780935, 0.05047767015399762, 0.33781588706467947, 0.10806377277945256,
    #                         0.17890280857109042, 0.8858270961677057, 0.3653649712141158, 0.21876934917953672,
    #                         0.7524961702186028, 0.10687958439356915, 0.7446032407755606, 0.46978529344049447,
    #                         0.5982556712791092, 0.14762019228529766, 0.18403482209315125, 0.6450721264682419,
    #                         0.048628006263405577, 0.24861250780276944, 0.5424085162280042, 0.2267733432700092,
    #                         0.3814115349046321, 0.9222327869035463, 0.9253568728677768, 0.566749924575,
    #                         0.5334708849890026, 0.014860024633228108, 0.977899263402005, 0.5730289040331858,
    #                         0.791756996276624, 0.5615573602763689, 0.8773352415649347, 0.5841958285306755,
    #                         0.7088498263689552, 0.14853345135645857, 0.4284507389678964, 0.6938900663424117,
    #                         0.10461974452285316, 0.4396052377745905, 0.16620214770453368, 0.5069786292640474,
    #                         0.8190358641362125, 0.09010673472443853, 0.8000687506941452, 0.5651263539578045,
    #                         0.5893477116806074, 0.1981006572162689, 0.4361182553388343, 0.29590376222083736,
    #                         0.03755767594167769, 0.030684840372946276, 0.45310500020123345, 0.7448640769500677,
    #                         0.557295406236397, 0.3851135995654865, 0.1680727975326186, 0.8382613207036929,
    #                         0.5990517974614926, 0.7827148182449711, 0.8485091818947146, 0.6031629758620348,
    #                         0.7810606172902821, 0.6157368760335693, 0.021165191172154985, 0.750464610487491,
    #                         0.1760421332836939, 0.458514206237273, 0.5131227077322451, 0.4840208902868258,
    #                         0.8443857945393476, 0.1748138948221194, 0.0146348751032499, 0.8487640718112321,
    #                         0.7426745772994341, 0.4566975353201722, 0.41689840704937775, 0.11672951094696327,
    #                         0.3386791329893397, 0.09465904074365306, 0.7158308727513142, 0.07708540441603862,
    #                         0.20595025836897252, 0.5737762314512022, 0.293831555203768, 0.6557267427145982,
    #                         0.8035683477432196, 0.35121350299640874, 0.09344037694376706, ]
    #         num_monos = len(monomer_draw)
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         # since GROW is not added to event_dict, no additional monomers will be added
    #         np.random.seed(10)
    #         result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=2, random_seed=10,
    #                          sg_ratio=sg_ratio)
    #         self.assertTrue(len(result[MONO_LIST]) == num_monos)
    #         gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
    #         self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
    #     finally:
    #         silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         pass

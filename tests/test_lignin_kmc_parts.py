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
PNG_B1 = os.path.join(SUB_DATA_DIR, 'test_b1.png')
C_LIGNIN_MOL_OUT = os.path.join(SUB_DATA_DIR, 'c_lignin_molfile.txt')
GOOD_C_LIGNIN_MOL_OUT = os.path.join(SUB_DATA_DIR, 'c_lignin_molfile_good.txt')
C_LIGNIN_PNG = os.path.join(SUB_DATA_DIR, 'c_lignin_molfile.png')

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

GOOD_RXN_RATES = {C5O4: {(0, 0): {MON_MON: 38335.59721483720, MON_OLI: 123.4195937155435, OLI_MON: 123.4195937155435,
                                  OLI_OLI: 3698609451.841636},
                         (1, 0): {MON_MON: 63606.84175294998, MON_OLI: 123.4195937155435, OLI_MON: 123.4195937155435,
                                  OLI_OLI: 3698609451.841636},
                         (2, 2): {MON_MON: 11762.46929017706, MON_OLI: 11762.469290177061,
                                  OLI_MON: 11762.46929017706, OLI_OLI: 11762.469290177061}},
                  C5C5: {(0, 0): {MON_MON: 4272.6301891208580, MON_OLI: 22.82331807203557, OLI_MON: 22.8233180720356,
                                  OLI_OLI: 10182201166.021704},
                         (2, 2): {MON_MON: 105537.166803781, MON_OLI: 105537.166803781,
                                  OLI_MON: 105537.166803781, OLI_OLI: 105537.166803781}},
                  B5: {(0, 0): {MON_MON: 577740233.38188150, MON_OLI: 348201801.4313151, OLI_MON: 348201801.4313151,
                                OLI_OLI: 348201801.43131510},
                       (0, 1): {MON_MON: 577740233.38188150, MON_OLI: 348201801.4313151, OLI_MON: 348201801.4313151,
                                OLI_OLI: 348201801.43131510},
                       (2, 2): {MON_MON: 251507997491.63364, MON_OLI: 348201801.4313151,
                                OLI_MON: 348201801.43131510, OLI_OLI: 348201801.4313151}},
                  BB: {(0, 0): {MON_MON: 958592907.60731790, MON_OLI: 958592907.6073179, OLI_MON: 958592907.6073179,
                                OLI_OLI: 958592907.60731790},
                       (1, 0): {MON_MON: 106838377.21810664, MON_OLI: 106838377.2181066, OLI_MON: 106838377.2181066,
                                OLI_OLI: 106838377.21810664},
                       (1, 1): {MON_MON: 958592907.60731790, MON_OLI: 958592907.6073179, OLI_MON: 958592907.6073179,
                                OLI_OLI: 958592907.60731790},
                       (0, 1): {MON_MON: 106838377.21810664, MON_OLI: 106838377.2181066, OLI_MON: 106838377.2181066,
                                OLI_OLI: 106838377.21810664},
                       (2, 2): {MON_MON: 32781102.221982773, MON_OLI: 32781102.22198277, OLI_MON: 32781102.22198277,
                                OLI_OLI: 32781102.221982773}},
                  BO4: {(0, 0): {MON_MON: 149736731.43118873, MON_OLI: 177267402.7946005, OLI_MON: 177267402.7946005,
                                 OLI_OLI: 177267402.79460046},
                        (1, 0): {MON_MON: 1327129.8749824178, MON_OLI: 177267402.7946005, OLI_MON: 177267402.7946005,
                                 OLI_OLI: 177267402.79460046},
                        (0, 1): {MON_MON: 1860006.6271960390, MON_OLI: 177267402.7946005, OLI_MON: 177267402.7946005,
                                 OLI_OLI: 177267402.79460046},
                        (1, 1): {MON_MON: 407201.80544143200, MON_OLI: 147913.0515942363, OLI_MON: 147913.0515942363,
                                 OLI_OLI: 147913.05159423634},
                        (2, 2): {MON_MON: 1590507825.8720958, MON_OLI: 692396712512.5765, OLI_MON: 692396712512.5765,
                                 OLI_OLI: 692396712512.5765}},
                  AO4: {(0, 0): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                 OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                        (1, 0): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                 OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                        (0, 1): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                 OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                        (1, 1): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                 OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                        (2, 2): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                 OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265}},
                  B1: {(0, 0): {MON_OLI: 570703.7954648494, OLI_MON: 570703.7954648494, OLI_OLI: 570703.7954648494},
                       (1, 0): {MON_OLI: 16485.40300715421, OLI_MON: 16485.40300715421, OLI_OLI: 16485.40300715421},
                       (0, 1): {MON_OLI: 89146.62342075957, OLI_MON: 89146.62342075957, OLI_OLI: 89146.62342075957},
                       (1, 1): {MON_OLI: 11762.46929017706, OLI_MON: 11762.46929017706, OLI_OLI: 11762.46929017706},
                       (2, 2): {MON_OLI: 570703.7954648494, OLI_MON: 570703.7954648494, OLI_OLI: 570703.7954648494}},
                  OX: {0: {MONOMER: 1360057059567.5383, OLIGOMER: 149736731.43118873},
                       1: {MONOMER: 2256621533195.0864, OLIGOMER: 151582896154.44305},
                       2: {MONOMER: 1360057059567.5383, OLIGOMER: 1360057059567.5383}},
                  Q: {0: {MONOMER: 45383.99955642, OLIGOMER: 45383.99955642},
                      1: {MONOMER: 16485.40300715, OLIGOMER: 16485.40300715},
                      2: {MONOMER: 45383.99955642, OLIGOMER: 45383.99955642}}}

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


def create_sample_kmc_result_c_lignin(num_monos=2, max_monos=12):
    initial_monomers = [Monomer(2, i) for i in range(num_monos)]
    # noinspection PyTypeChecker
    initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    initial_state = create_initial_state(initial_events, initial_monomers)
    initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE))
    result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), n_max=max_monos, t_max=2, random_seed=10)
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
    #     # Only adds three lines to coverage... oh well!
    #     try:
    #         silent_remove(TCL_FILE_LOC)
    #         result = create_sample_kmc_result_c_lignin(num_monos=120, max_monos=240)
    #         good_last_time = 0.48400289541505354
    #         self.assertAlmostEqual(result[TIME][-1], good_last_time)
    #         temp = result[ADJ_MATRIX]
    #         temp = result[MONO_LIST]
    #         gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", toppar_dir=None,
    #                    out_dir=SUB_DATA_DIR)
    #     #     self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_C_LIGNIN_OUT))
    #     finally:
    #         # silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         pass

    def testGenMolCLignin(self):
        # Only testing parts of this package; printing block to ease checking output
        try:
            result = create_sample_kmc_result_c_lignin(num_monos=12, max_monos=14)
            block = generate_mol(result[ADJ_MATRIX], result[MONO_LIST])
            with open(C_LIGNIN_MOL_OUT, "w") as f:
                f.write(block)
            self.assertFalse(diff_lines(C_LIGNIN_MOL_OUT, C_LIGNIN_MOL_OUT))
            # Uncomment below to visually check output
            # mol = MolFromMolBlock(block)
            # Compute2DCoords(mol)
            # MolToFile(mol, PNG_B1, size=(2000, 1000))
        finally:
            # silent_remove(C_LIGNIN_PNG, disable=DISABLE_REMOVE)
            silent_remove(C_LIGNIN_MOL_OUT, disable=DISABLE_REMOVE)
            pass

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
            mono_type_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                              1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
            num_monos = len(mono_type_list)
            initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(mono_type_list)]
            initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
            initial_state = create_initial_state(initial_events, initial_monomers)
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
    #     # Only adds one line of coverage
    #     try:
    #         sg_ratio = 10.
    #         pct_s = sg_ratio / (1 + sg_ratio)
    #         num_monos = 200
    #         np.random.seed(1)
    #         monomer_draw = np.around(np.random.rand(num_monos), MAX_NUM_DECIMAL)
    #         initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         initial_events.append(Event(GROW, [], rate=DEF_ADD_RATE, bond=sg_ratio))
    #         result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=0.02, random_seed=1,
    #                          sg_ratio=sg_ratio)
    #         adj_result = analyze_adj_matrix(result[ADJ_MATRIX])
    #         if adj_result[BONDS][B1] > 0:
    #             print(f"Woot! sg={sg_ratio}")
    #         gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
    #         self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
    #     finally:
    #         silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         pass

    def testDynamics(self):
        # Tests procedures in the Dynamics.ipynb
        sg_ratio = 1
        pct_s = sg_ratio / (1 + sg_ratio)
        # minimize number of random calls during testing
        monomer_type_list = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                             0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        num_monos = len(monomer_type_list)
        initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(monomer_type_list)]


        initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
        initial_state = create_initial_state(initial_events, initial_monomers)
        # since GROW is not added to event_dict, no additional monomers will be added (sg_ratio is thus not needed)
        result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=20, random_seed=10, dynamics=True)
        # With dynamics, the MONO_LIST will be a list of monomer lists:
        #    the inner list is the usual MONO_LIST, but here is it saved for every time step
        expected_num_t_steps = 138
        self.assertTrue(len(result[MONO_LIST]) == expected_num_t_steps)
        self.assertTrue(len(result[MONO_LIST][-1]) == num_monos)

        # want dict[key: [], ...] where the inner list is values by timestep
        #                      instead of list of timesteps with [[key: val, ...], ... ]
        t_steps = result[TIME]
        adj_list = result[ADJ_MATRIX]
        self.assertEqual(len(t_steps), expected_num_t_steps)

        bond_type_dict, olig_len_dict, sum_list = get_bond_type_v_time_dict(adj_list, sum_len_larger_than=10)

        # test results by checking sums
        good_bond_type_sum_dict = {BO4: 684, B1: 0, BB: 553, B5: 594, C5C5: 62, AO4: 0, C5O4: 347}
        bond_type_sum_dict = {}
        for bond_type, val_list in bond_type_dict.items():
            self.assertEqual(len(val_list), expected_num_t_steps)
            bond_type_sum_dict[bond_type] = sum(val_list)
        self.assertEqual(bond_type_sum_dict, good_bond_type_sum_dict)

        good_olig_len_sum_dict = {1: 2944, 2: 450, 3: 69, 4: 40, 5: 10, 6: 0, 7: 7, 8: 0, 9: 90, 10: 0, 11: 0,
                                  12: 24, 13: 13, 14: 0, 15: 0, 16: 0, 17: 0, 18: 90, 19: 19, 20: 20, 21: 21, 22: 22,
                                  23: 46, 24: 216, 25: 0, 26: 0, 27: 0, 28: 56, 29: 87, 30: 60, 31: 62, 32: 96, 33: 66,
                                  34: 272, 35: 140, 36: 216, 37: 0, 38: 304, 39: 0, 40: 80}
        olig_len_sum_dict = {}
        for olig_len, val_list in olig_len_dict.items():
            self.assertEqual(len(val_list), expected_num_t_steps)
            olig_len_sum_dict[olig_len] = sum(val_list)
        self.assertEqual(olig_len_sum_dict, good_olig_len_sum_dict)

        good_sum_sum_list = 1910
        self.assertEqual(sum(sum_list), good_sum_sum_list)

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
        type_list = []
        for monomer in initial_monomers:
            type_list.append(monomer.type)
        print(type_list)

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
    #         initial_mono_type_list = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    #                                   1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
    #                                   1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    #                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
    #                                   0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    #                                   1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
    #                                   1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
    #                                   1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    #         num_monos = len(initial_mono_type_list)
    #         initial_monomers = [Monomer(mono_type, i) for i, mono_type in enumerate(initial_mono_type_list)]
    #         initial_events = create_initial_events(initial_monomers, GOOD_RXN_RATES)
    #         initial_state = create_initial_state(initial_events, initial_monomers)
    #         # since GROW is not added to event_dict, no additional monomers will be added
    #         np.random.seed(10)
    #         type_list = []
    #         for monomer in initial_monomers:
    #             type_list.append(monomer.type)
    #         print(type_list)
    #         result = run_kmc(GOOD_RXN_RATES, initial_state, sorted(initial_events), t_max=2, random_seed=10,
    #                          sg_ratio=sg_ratio)
    #         # print(result)
    #         # self.assertTrue(len(result[MONO_LIST]) == num_monos)
    #         # gen_psfgen(result[ADJ_MATRIX], result[MONO_LIST], fname=TCL_FNAME, segname="L", out_dir=SUB_DATA_DIR)
    #         # self.assertFalse(diff_lines(TCL_FILE_LOC, GOOD_TCL_NO_GROW_OUT))
    #         self.assertTrue(False)
    #     finally:
    #         # silent_remove(TCL_FILE_LOC, disable=DISABLE_REMOVE)
    #         pass

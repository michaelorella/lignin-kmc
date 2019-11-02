#!/usr/bin/env python3

import logging
import os
import unittest
import numpy as np
from common_wrangler.common import InvalidDataError
from ligninkmc import Event
from ligninkmc import Monomer
from ligninkmc.kineticMonteCarlo import run_kmc
from ligninkmc.create_lignin import (calc_rates, DEF_TEMP, DEF_E_A_KCAL_MOL, create_initial_monomers,
                                     create_initial_events, create_initial_state, DEF_INI_RATE)
from ligninkmc.kmc_common import (TEMP, E_A_KCAL_MOL, E_A_J_PART, C5O4, OX, Q, C5C5, B5, BB, BO4, AO4, B1,
                                  MON_MON, MON_DIM, DIM_DIM, DIM_MON, MONOMER, DIMER, GROW, TIME, MONO_LIST, ADJ_MATRIX)

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'common')

# Data #


# Output files #
GOOD_LIST_OUT = os.path.join(SUB_DATA_DIR, "good_list.txt")

# Data #
GOOD_RXN_RATES = {C5O4: {(0, 0): {MON_MON: 38335.597214837195, MON_DIM: 123.41959371554347, DIM_MON: 123.41959371554347,
                                  DIM_DIM: 3698609451.841636},
                         (1, 0): {MON_MON: 63606.84175294998, MON_DIM: 123.41959371554347, DIM_MON: 123.41959371554347,
                                  DIM_DIM: 3698609451.841636}},
                  C5C5: {(0, 0): {MON_MON: 4272.630189120858, MON_DIM: 22.82331807203557, DIM_MON: 22.82331807203557,
                                  DIM_DIM: 10182201166.021704}},
                  B5: {(0, 0): {MON_MON: 577740233.3818815, MON_DIM: 348201801.4313151, DIM_MON: 348201801.4313151,
                                DIM_DIM: 348201801.4313151},
                       (0, 1): {MON_MON: 577740233.3818815, MON_DIM: 348201801.4313151, DIM_MON: 348201801.4313151,
                                DIM_DIM: 348201801.4313151}},
                  BB: {(0, 0): {MON_MON: 958592907.6073179, MON_DIM: 958592907.6073179, DIM_MON: 958592907.6073179,
                                DIM_DIM: 958592907.6073179},
                       (1, 0): {MON_MON: 106838377.21810664, MON_DIM: 106838377.21810664, DIM_MON: 106838377.21810664,
                                DIM_DIM: 106838377.21810664},
                       (1, 1): {MON_MON: 958592907.6073179, MON_DIM: 958592907.6073179, DIM_MON: 958592907.6073179,
                                DIM_DIM: 958592907.6073179},
                       (0, 1): {MON_MON: 106838377.21810664, MON_DIM: 106838377.21810664, DIM_MON: 106838377.21810664,
                                DIM_DIM: 106838377.21810664}},
                  BO4: {(0, 0): {MON_MON: 149736731.43118873, MON_DIM: 177267402.79460046, DIM_MON: 177267402.79460046,
                                 DIM_DIM: 177267402.79460046},
                        (1, 0): {MON_MON: 1327129.8749824178, MON_DIM: 177267402.79460046, DIM_MON: 177267402.79460046,
                                 DIM_DIM: 177267402.79460046},
                        (0, 1): {MON_MON: 1860006.627196039, MON_DIM: 177267402.79460046, DIM_MON: 177267402.79460046,
                                 DIM_DIM: 177267402.79460046},
                        (1, 1): {MON_MON: 407201.805441432, MON_DIM: 147913.05159423634, DIM_MON: 147913.05159423634}},
                  AO4: {(0, 0): {MON_MON: 0.004169189173972648, MON_DIM: 0.004169189173972648,
                                 DIM_MON: 0.004169189173972648, DIM_DIM: 0.004169189173972648},
                        (1, 0): {MON_MON: 0.004169189173972648, MON_DIM: 0.004169189173972648,
                                 DIM_MON: 0.004169189173972648, DIM_DIM: 0.004169189173972648},
                        (0, 1): {MON_MON: 0.004169189173972648, MON_DIM: 0.004169189173972648,
                                 DIM_MON: 0.004169189173972648, DIM_DIM: 0.004169189173972648},
                        (1, 1): {MON_MON: 0.004169189173972648, MON_DIM: 0.004169189173972648,
                                 DIM_MON: 0.004169189173972648, DIM_DIM: 0.004169189173972648}},
                  B1: {(0, 0): {MON_DIM: 570703.7954648494, DIM_MON: 570703.7954648494, DIM_DIM: 570703.7954648494},
                       (1, 0): {MON_DIM: 16485.40300715421, DIM_MON: 16485.40300715421, DIM_DIM: 16485.40300715421},
                       (0, 1): {MON_DIM: 89146.62342075957, DIM_MON: 89146.62342075957, DIM_DIM: 89146.62342075957},
                       (1, 1): {MON_DIM: 11762.469290177061, DIM_MON: 11762.469290177061, DIM_DIM: 11762.469290177061}},
                  OX: {0: {MONOMER: 1360057059567.5383, DIMER: 149736731.43118873},
                       1: {MONOMER: 2256621533195.0864, DIMER: 151582896154.44305}},
                  Q: {0: {MONOMER: 45383.99955642849, DIMER: 45383.99955642849},
                      1: {MONOMER: 16485.40300715421, DIMER: 16485.40300715421}}}


# Tests #

class TestCalcRates(unittest.TestCase):
    """
    Tests calculation of rate coefficients by the Eyring equation.
    """
    def test_calc_rates_from_kcal_mol(self):
        config = {TEMP: DEF_TEMP, E_A_KCAL_MOL: DEF_E_A_KCAL_MOL, E_A_J_PART: None}
        result = calc_rates(config)
        rxn_type, substrate, substrate_type = None, None, None  # to make IDE happy
        # nested loops instead of dealing with almost equal dicts
        try:
            for rxn_type in GOOD_RXN_RATES:
                for substrate in GOOD_RXN_RATES[rxn_type]:
                    for substrate_type in GOOD_RXN_RATES[rxn_type][substrate]:
                        # print(GOOD_RXN_RATES[rxn_type][substrate][substrate_type])
                        self.assertAlmostEqual(GOOD_RXN_RATES[rxn_type][substrate][substrate_type],
                                               result[rxn_type][substrate][substrate_type])
        except (TypeError, IndexError) as e:
            print(f'{e}\nError when looking at rxn_type: {rxn_type} substrate: {substrate}    '
                  f'substrate_type:    {substrate_type}')


class TestMonomers(unittest.TestCase):
    def testCreateConiferyl(self):
        mon = Monomer(0, 0)  # Makes a guaiacol unit monomer with ID = 0
        self.assertTrue(mon.open == {8, 4, 5})
        self.assertTrue(str(mon) == '0: coniferyl alcohol is connected to {0} and active at position 0')

    def testCreateSyringol(self):
        mon = Monomer(1, 2)  # Makes a syringol unit monomer with ID = 2
        self.assertTrue(mon.open == {4, 8})
        self.assertTrue(mon.connectedTo == {2})
        self.assertTrue(str(mon) == '2: sinapyl alcohol is connected to {2} and active at position 0')
        self.assertTrue(repr(mon) == '2: sinapyl alcohol \n')

    def testUnknownUnit(self):
        try:
            mon = Monomer(3, 2)  # unit type 3 is not currently implemented
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
        event1 = Event(rxn, [1, 2], GOOD_RXN_RATES[rxn][(0, 1)][MON_DIM], (4, 5))
        good_str = "Forming bo4 bond between indices [1, 2] (adjacency_matrix update (4, 5))"
        self.assertTrue(str(event1) == good_str)
        self.assertTrue(repr(event1) == good_str)

    def testEventIDHash(self):
        events1 = create_initial_events([0.48772], 1, 0.75, GOOD_RXN_RATES)
        events2 = create_initial_events([0.48772], 1, 0.75, GOOD_RXN_RATES)
        self.assertTrue(events1 == events2)
        check_set = {events1[0], events2[0]}
        self.assertTrue(len(check_set) == 1)

    def testInitialEvents(self):
        initial_events = create_initial_events([0.48772, 0.15174, 0.7886], 3, 0.75, GOOD_RXN_RATES)
        self.assertTrue(initial_events[2].index == [2])


class TestCreateInitialMonomers(unittest.TestCase):
    def testCreate3Monomers(self):
        initial_monomers = create_initial_monomers(0.75, 3, [0.48772, 0.15174, 0.7886])
        self.assertTrue(len(initial_monomers) == 3)
        self.assertTrue(initial_monomers[0].type == 1)
        self.assertTrue(initial_monomers[1].type == 1)
        self.assertTrue(initial_monomers[2].type == 0)
        self.assertTrue(initial_monomers[1] < initial_monomers[2])
        self.assertFalse(initial_monomers[0] == initial_monomers[1])


class TestState(unittest.TestCase):
    def testCreateInitialState(self):
        num_monos = 3
        sg_ratio = 0.75
        monomer_draw = [0.48772, 0.15174, 0.7886]
        initial_monomers = create_initial_monomers(sg_ratio, num_monos, monomer_draw)
        initial_events = create_initial_events(monomer_draw, num_monos, sg_ratio, GOOD_RXN_RATES)
        ini_state = create_initial_state(initial_events, initial_monomers, num_monos)
        self.assertTrue(len(ini_state) == 3)
        self.assertTrue(str(initial_monomers[0]) == str(ini_state[0][MONOMER]))


class TestRunKMC(unittest.TestCase):
    def test_adj_matrix(self):
        num_monos = 3
        sg_ratio = 0.75
        monomer_draw = [0.48772, 0.15174, 0.7886]
        # these are tested separately
        initial_monomers = create_initial_monomers(sg_ratio, num_monos, monomer_draw)
        initial_events = create_initial_events(monomer_draw, num_monos, sg_ratio, GOOD_RXN_RATES)
        ini_state = create_initial_state(initial_events, initial_monomers, num_monos)
        # new to test
        events = {initial_events[i] for i in range(num_monos)}
        events.add(Event(GROW, [], rate=DEF_INI_RATE, bond=sg_ratio))
        # make random seed and sort events for testing reliability
        np.random.seed(10)
        result = run_kmc(n_max=10, t_final=1, rates=GOOD_RXN_RATES, initial_state=ini_state,
                         initial_events=sorted(events), random_seed=10)
        self.assertTrue(len(result[TIME]))
        self.assertAlmostEqual(result[TIME][-1], 0.009396540330667606)
        self.assertTrue(len(result[MONO_LIST]) == 10)
        self.assertTrue(str(result[MONO_LIST][-1]) == '9: coniferyl alcohol is connected to '
                                                      '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} and active at position 4')
        good_dok_keys = [(0, 2), (2, 0), (1, 0), (0, 1), (1, 3), (3, 1), (3, 4), (4, 3), (4, 5),
                         (5, 4), (6, 5), (5, 6), (6, 7), (7, 6), (7, 8), (8, 7), (8, 9), (9, 8)]
        good_dok_vals = [8.0, 5.0, 8.0, 4.0, 4.0, 8.0, 5.0, 8.0, 4.0, 8.0, 8.0, 5.0, 5.0, 8.0, 5.0, 8.0, 5.0, 8.0]
        self.assertTrue(list(result[ADJ_MATRIX].keys()) == good_dok_keys)
        self.assertTrue(list(result[ADJ_MATRIX].values()) == good_dok_vals)

# class TestRunKMC(unittest.TestCase):
#     def test_adj_matrix(self):
#         initial_monomers = create_initial_monomers(0.5, 3, [0.48772, 0.15174, 0.7886])
#         self.assertTrue(len(initial_monomers) == 3)
#         self.assertTrue(initial_monomers[0].identity == 0)
#         self.assertTrue(initial_monomers[1].identity == 1)
#         # self.assertTrue(initial_monomers[2].identity == 1)

#         ini_events = None
#         ini_state = None
#         residues = run_kmc(n_max=10, t_final=1, rates=GOOD_RXN_RATES, initial_state=ini_state,
#                            initial_events=ini_events)
#         # #  To show the state, we will print the adjacency matrix that has been generated,
#         # #  although this is not the typical output examined.
#         # print(res[ADJ_MATRIX].todense())
#         print(residues)
#         pass

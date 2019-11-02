#!/usr/bin/env python3

import logging
import os
import unittest

from common_wrangler.common import InvalidDataError

from ligninkmc import Monomer
from ligninkmc.kineticMonteCarlo import run_kmc
from ligninkmc.create_lignin import (calc_rates, DEF_TEMP, DEF_E_A_KCAL_MOL, create_initial_monomers)
from ligninkmc.kmc_common import TEMP, E_A_KCAL_MOL, E_A_J_PART, C5O4, OX, Q, C5C5, B5, BB, BO4, AO4, B1, MON_MON, \
    MON_DIM, DIM_DIM, DIM_MON, MONOMER, DIMER

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
            print(e)
            print("Error when looking at rxn_type: {} substrate: {}    substrate_type:    {}"
                  "".format(rxn_type, substrate, substrate_type))


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

# class TestRunKMC(unittest.TestCase):
    # def test_adj_matrix(self):
    #     pct_s = .5
    #     num_monos = 4
    #     np.random.seed(10)
    #     monomer_draw = np.random.rand(num_monos)
    #     initial_monomers = [Monomer(int(s_or_g < pct_s), i) for i, s_or_g in zip(range(num_monos), monomer_draw)]
    #     # #  To show the state, we will print the adjacency matrix that has been generated,
    #     # #  although this is not the typical output examined.
    #     # print(res[ADJ_MATRIX].to_dense())
    #     print(initial_monomers)
    #     pass


class TestCreatInitialMonomers(unittest.TestCase):
    def testCreate3Monomers(self):
        initial_monomers = create_initial_monomers(0.75, 3, [0.48772, 0.15174, 0.7886])
        self.assertTrue(len(initial_monomers) == 3)
        self.assertTrue(initial_monomers[0].type == 1)
        self.assertTrue(initial_monomers[1].type == 1)
        self.assertTrue(initial_monomers[2].type == 0)
        self.assertTrue(initial_monomers[1] < initial_monomers[2])
        self.assertFalse(initial_monomers[0] == initial_monomers[1])

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

#!/usr/bin/env python3

import logging
import os
import unittest
from ligninkmc.plot_bond_formation import main
from common_wrangler.common import capture_stderr, capture_stdout, silent_remove

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
TEST_DIR = os.path.dirname(__file__)
MAIN_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'plots')

# Files #
DEF_BOND_PNG = os.path.join(MAIN_DIR, "bond_v_add_rate_1.png")
DEF_MONO_PNG = os.path.join(MAIN_DIR, "mono_v_olig_1_1.png")

BOND_OPT_1_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1e08.png")
BOND_OPT_2_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1e04.png")
MONO_OPT_1_PNG = os.path.join(SUB_DATA_DIR, "mono_v_olig_0-25_1e08.png")
MONO_OPT_2_PNG = os.path.join(SUB_DATA_DIR, "mono_v_olig_3_1e08.png")
MONO_OPT_3_PNG = os.path.join(SUB_DATA_DIR, "mono_v_olig_0-25_1e04.png")
MONO_OPT_4_PNG = os.path.join(SUB_DATA_DIR, "mono_v_olig_3_1e04.png")

ORELLA_BOND_PNG = os.path.join(SUB_DATA_DIR, "mono_v_olig_1_1.png")
ORELLA_MONO_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1.png")

# Data #

ORELLA_ENERGIES = {'5o4': {(0, 0): {('monomer', 'monomer'): 11.2, ('monomer', 'dimer'): 14.6,
                                    ('dimer', 'monomer'): 14.6, ('dimer', 'dimer'): 4.4},
                           (1, 0): {('monomer', 'monomer'): 10.9, ('monomer', 'dimer'): 14.6,
                                    ('dimer', 'monomer'): 14.6, ('dimer', 'dimer'): 4.4}},
                   '55': {(0, 0): {('monomer', 'monomer'): 12.5, ('monomer', 'dimer'): 15.6,
                                   ('dimer', 'monomer'): 15.6, ('dimer', 'dimer'): 3.8}},
                   'b5': {(0, 0): {('monomer', 'monomer'): 5.5, ('monomer', 'dimer'): 5.8,
                                   ('dimer', 'monomer'): 5.8, ('dimer', 'dimer'): 5.8},
                          (0, 1): {('monomer', 'monomer'): 5.5, ('monomer', 'dimer'): 5.8,
                                   ('dimer', 'monomer'): 5.8, ('dimer', 'dimer'): 5.8}},
                   'bb': {(0, 0): {('monomer', 'monomer'): 5.2, ('monomer', 'dimer'): 5.2,
                                   ('dimer', 'monomer'): 5.2, ('dimer', 'dimer'): 5.2},
                          (1, 0): {('monomer', 'monomer'): 6.5, ('monomer', 'dimer'): 6.5,
                                   ('dimer', 'monomer'): 6.5, ('dimer', 'dimer'): 6.5},
                          (1, 1): {('monomer', 'monomer'): 5.2, ('monomer', 'dimer'): 5.2,
                                   ('dimer', 'monomer'): 5.2, ('dimer', 'dimer'): 5.2}},
                   'bo4': {(0, 0): {('monomer', 'monomer'): 6.3, ('monomer', 'dimer'): 6.2,
                                    ('dimer', 'monomer'): 6.2, ('dimer', 'dimer'): 6.2},
                           (1, 0): {('monomer', 'monomer'): 9.1, ('monomer', 'dimer'): 6.2,
                                    ('dimer', 'monomer'): 6.2, ('dimer', 'dimer'): 6.2},
                           (0, 1): {('monomer', 'monomer'): 8.9, ('monomer', 'dimer'): 6.2,
                                    ('dimer', 'monomer'): 6.2, ('dimer', 'dimer'): 6.2},
                           (1, 1): {('monomer', 'monomer'): 9.8, ('monomer', 'dimer'): 10.4,
                                    ('dimer', 'monomer'): 10.4}},
                   'ao4': {(0, 0): {('monomer', 'monomer'): 20.7, ('monomer', 'dimer'): 20.7,
                                    ('dimer', 'monomer'): 20.7, ('dimer', 'dimer'): 20.7},
                           (1, 0): {('monomer', 'monomer'): 20.7, ('monomer', 'dimer'): 20.7,
                                    ('dimer', 'monomer'): 20.7, ('dimer', 'dimer'): 20.7},
                           (0, 1): {('monomer', 'monomer'): 20.7, ('monomer', 'dimer'): 20.7,
                                    ('dimer', 'monomer'): 20.7, ('dimer', 'dimer'): 20.7},
                           (1, 1): {('monomer', 'monomer'): 20.7, ('monomer', 'dimer'): 20.7,
                                    ('dimer', 'monomer'): 20.7, ('dimer', 'dimer'): 20.7}},
                   'b1': {(0, 0): {('monomer', 'dimer'): 9.6,
                                   ('dimer', 'monomer'): 9.6, ('dimer', 'dimer'): 9.6},
                          (1, 0): {('monomer', 'dimer'): 11.7,
                                   ('dimer', 'monomer'): 11.7, ('dimer', 'dimer'): 11.7},
                          (0, 1): {('monomer', 'dimer'): 10.7,
                                   ('dimer', 'monomer'): 10.7, ('dimer', 'dimer'): 10.7},
                          (1, 1): {('monomer', 'dimer'): 11.9,
                                   ('dimer', 'monomer'): 11.9, ('dimer', 'dimer'): 11.9}},
                   'ox': {0: {'monomer': 0.9, 'dimer': 6.3}, 1: {'monomer': 0.6, 'dimer': 2.2}},
                   'q': {0: {'monomer': 11.1, 'dimer': 11.1}, 1: {'monomer': 11.7, 'dimer': 11.7}}}
ORELLA_ENERGIES['bb'][(0, 1)] = ORELLA_ENERGIES['bb'][(1, 0)]


# Tests #

class TestNoOutput(unittest.TestCase):
    # Most of these are to test for failing nicely when the program encounters invalid input
    def testHelp(self):
        test_input = ['-h']
        # main(test_input)
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    # def testInvalidRandomSeedAlpha(self):
    #     test_input = ["-r", "ghost"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('positive integer value' in output)
    #
    # def testInvalidRandomSeedNegNum(self):
    #     test_input = ["-r", "-1"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('positive integer value' in output)
    #
    def testInvalidRandomSeed0(self):
        test_input = ["-r", "0"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive integer value' in output)
    #
    # def testInvalidRandomSeedTooBig(self):
    #     just_over_max = str(2**32)
    #     test_input = ["-r", just_over_max]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('positive integer value' in output)
    #
    # def testInvalidExtension(self):
    #     test_input = ["-r", "10", "-f", "ghost"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('currently supported types' in output)
    #
    # def testAlphaSGRatio(self):
    #     test_input = ["-r", "10", "-sg", "ghost"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive number' in output)
    #
    # def testNegSGRatio(self):
    #     test_input = ["-r", "10", "-sg", "-0.1"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive number' in output)
    #
    # def testNegSimLen(self):
    #     test_input = ["-r", "10", "-l", "-0.1"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive number' in output)
    #
    # def testNegIniMonos(self):
    #     test_input = ["-r", "10", "-i", "-4"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive integer' in output)

    def testFractionalIniMonos(self):
        test_input = ["-r", "10", "-i", "4.5"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('integer' in output)

    def testZeroSimTime(self):
        test_input = ["-r", "10", "-l", "0"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive' in output)
    #
    # def testNegMaxMonos(self):
    #     test_input = ["-r", "10", "-m", "-8"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive integer' in output)
    #
    # def testFractionalMaxMonos(self):
    #     test_input = ["-r", "10", "-m", "12.1"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive integer' in output)
    #
    # def testFewerMaxThanIniMonos(self):
    #     test_input = ["-r", "10", "-i", "6", "-m", "4"]
    #     # main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('is less than' in output)
    #     with capture_stdout(main, test_input) as output:
    #         self.assertTrue("Lignin KMC created 6 monomers" in output)
    #
    # def testBadImageSize(self):
    #     test_input = ["-r", "10", "-s", "4"]
    #     # main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue("two positive numbers" in output)
    #
    # def testAlphaAddRate(self):
    #     test_input = ["-a", "ghost"]
    #     # main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue("A positive number" in output)
    #
    # def testZeroAddRate(self):
    #     test_input = ["-a", "0"]
    #     # main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue("A positive number" in output)

    def testListWithSpace(self):
        test_input = ["-a", "1.0", "0.1"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("spaces" in output)

    def testNegListVal(self):
        test_input = ["-sg", "0, -0.1"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("positive" in output)

    def testZeroListVal(self):
        test_input = ["-a", "1.0, 0"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("positive" in output)


class TestWarnings(unittest.TestCase):
    def testFewerMaxThanMinMonos(self):
        try:
            test_input = ["-r", "10", "-i", "20", "-m", "10"]
            # main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("initial" in output)
        finally:
            for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testFewerThanMinRepeats(self):
        try:
            test_input = ["-r", "10", "-n", "1", "-m", "10"]
            # main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertTrue("at least 3" in output)
        finally:
            for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass


class TestNormalUse(unittest.TestCase):
    def testSmallNumMonos(self):
        try:
            for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
                silent_remove(fname)
            test_input = ["-r", "10", "-m", "20"]
            main(test_input)
            for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testMultOptions(self):
        expected_pngs = [BOND_OPT_1_PNG, BOND_OPT_2_PNG,
                         MONO_OPT_1_PNG, MONO_OPT_2_PNG, MONO_OPT_3_PNG, MONO_OPT_4_PNG]
        try:
            for fname in expected_pngs:
                silent_remove(fname)
            # add_rates = [1e8, 1e6, 1e4, 1e2, 1]
            # sg_opts = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10]
            test_input = ["-r", "10", "-m", "20", "-a", "1e8, 1e4", "-sg", "0.25, 3", "-d", SUB_DATA_DIR]
            main(test_input)
            for fname in expected_pngs:
                print(fname)
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_pngs:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testOrellaBarriers(self):
        expected_out_files = [ORELLA_BOND_PNG, ORELLA_MONO_PNG]
        try:
            for fname in expected_out_files:
                silent_remove(fname)
            test_input = ["-e", "-m", "10", "-d", SUB_DATA_DIR]
            main(test_input)
            for fname in expected_out_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_out_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

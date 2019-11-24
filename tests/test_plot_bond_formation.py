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
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'plots')

# Files #
DEF_PNG_OUT = os.path.join(SUB_DATA_DIR, "test.png")

# Data #


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

    # def testUnrecognizedArg(self):
    #     test_input = ['-@']
    #     # main(test_input)
    #     if logger.isEnabledFor(logging.DEBUG):
    #         main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue("unrecognized argument" in output)
    #     with capture_stdout(main, test_input) as output:
    #         self.assertTrue("optional arguments" in output)
    #
    # def testBadConfigFile(self):
    #     test_input = ["-c", "ghost.ini"]
    #     # main(test_input)
    #     if logger.isEnabledFor(logging.DEBUG):
    #         main(test_input)
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue("Could not find specified configuration file" in output)
    #     with capture_stdout(main, test_input) as output:
    #         self.assertTrue("optional arguments" in output)
    #
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
    # def testInvalidRandomSeed0(self):
    #     test_input = ["-r", "0"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('positive integer value' in output)
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
    #
    # def testFractionalIniMonos(self):
    #     test_input = ["-r", "10", "-i", "4.5"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive integer' in output)
    #
    # def testAlphaIniMonos(self):
    #     test_input = ["-r", "10", "-i", "ghost"]
    #     with capture_stderr(main, test_input) as output:
    #         self.assertTrue('must be a positive integer' in output)
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


class TestNormalUse(unittest.TestCase):
    def testDefArgs(self):
        # test_input = ["-r", "10"]
        # # main(test_input)
        # good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
        #                      "       1 oligomer(s) of chain length 10, with branching coefficient 0.0"
        # with capture_stdout(main, test_input) as output:
        #     self.assertTrue(good_chain_summary in output)
        silent_remove(DEF_PNG_OUT)

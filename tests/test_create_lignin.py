#!/usr/bin/env python3

import logging
import os
import unittest
from ligninkmc.create_lignin import main
from common_wrangler.common import capture_stderr, capture_stdout

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'run_kmc')

# Files #
SMALL_INI = os.path.join(SUB_DATA_DIR, "small_config.ini")


# Data #


# Tests #

class TestCreateLigninNoOutput(unittest.TestCase):
    def testHelp(self):
        test_input = ['-h']
        # main(test_input)
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testUnrecognizedArg(self):
        test_input = ['-@']
        # main(test_input)
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("unrecognized argument" in output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)

    def testBadConfigFile(self):
        test_input = ["-c", "ghost.ini"]
        # main(test_input)
        if logger.isEnabledFor(logging.DEBUG):
            main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Could not find specified configuration file" in output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("optional arguments" in output)


class TestCreateLigninNormalUse(unittest.TestCase):
    def testDefArgs(self):
        test_input = ["-r", "10"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 trimer(s) (chain length 3)\n       1 oligomer(s) of chain length 7"
        good_bond_summary = "composed of the following bond types and number:\n " \
                            "    55:    0    5O4:    1    AO4:    0     B1:    0     B5:    2     BB:    1    BO4:    4"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       4 monomer(s) (chain length 1)\n" \
                                 "       3 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n     " \
                                "55:    0    5O4:    0    AO4:    0     B1:    0     B5:    2     " \
                                "BB:    1    BO4:    0"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testSmallConfig(self):
        test_input = ["-c", SMALL_INI, "-r", "11"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 oligomer(s) of chain length 10"
        good_bond_summary = "composed of the following bond types and number:\n     " \
                            "55:    0    5O4:    0    AO4:    0     B1:    0     B5:    2     BB:    1    BO4:    6"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       4 monomer(s) (chain length 1)\n" \
                                 "       3 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n     " \
                                "55:    0    5O4:    0    AO4:    0     B1:    0     B5:    2     BB:    1" \
                                "    BO4:    0"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

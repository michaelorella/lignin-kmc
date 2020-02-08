#!/usr/bin/env python3

import logging
import os
import unittest
from ligninkmc.create_lignin import main, OPENING_MSG, DEF_BASENAME
from common_wrangler.common import capture_stderr, capture_stdout, silent_remove, diff_lines

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
TEST_DIR = os.path.dirname(__file__)
MAIN_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'run_kmc')
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
TEMP_DIR = os.path.join("..", TEST_DIR, 'temp_dir/temp_dir')
INNER_TEMP_DIR = os.path.join("..", TEST_DIR, 'temp_dir/')

# Files #
DEF_JSON_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".json")
DEF_PNG_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".png")
DEF_SDF_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".sdf")
DEF_SMI_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".smi")
DEF_SVG_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".svg")
DEF_TCL_OUT = os.path.join(SUB_DATA_DIR, DEF_BASENAME + ".tcl")
GOOD_DEF_JSON_OUT = os.path.join(SUB_DATA_DIR, "lignin-kmc-out_good.json")
GOOD_DEF_TCL_OUT = os.path.join(SUB_DATA_DIR, "lignin-kmc-out_good.tcl")
GOOD_TCL_OPTIONS_OUT = os.path.join(SUB_DATA_DIR, "lignin-kmc-out_options_good.tcl")
SMALL_INI = os.path.join(SUB_DATA_DIR, "small_config.ini")
TEST_SMI_BASENAME = "test_lignin.smi"
TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, TEST_SMI_BASENAME)
GOOD_TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, "good_test_lignin.smi")
TEST_SMI_OUT_TEMP_DIR = os.path.join(TEMP_DIR, TEST_SMI_BASENAME)

BOND_V_STEP_PNG = os.path.join(MAIN_DIR, "bond_dist_v_step_1_1e06.png")
MONO_V_STEP_PNG = os.path.join(MAIN_DIR, "mono_olig_v_step_1_1e06.png")

PLOT_BOND_V_STEP_PNG = os.path.join(PLOT_DIR, "bond_dist_v_step_1_1e06.png")
PLOT_MONO_V_STEP_PNG = os.path.join(PLOT_DIR, "mono_olig_v_step_1_1e06.png")

PLOT_BOND_V_SG6_PNG = os.path.join(PLOT_DIR, "bond_dist_v_sg_1e06.png")
PLOT_BOND_V_SG8_PNG = os.path.join(PLOT_DIR, "bond_dist_v_sg_1e08.png")


# Data #

# testing pieces of json, skipping parts that refer to version; more thorough testing is beyond scope
GOOD_JSON_PARTS = ['"bonds":[{"', '{"bo":2,"atoms":[', '"aromaticAtoms":[0,1,2,3,4,5,',
                   '"aromaticBonds":[0,1,2,3,4,5,', '"atomRings":[[']

# Tests #

class TestCreateLigninNoOutput(unittest.TestCase):
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

    def testInvalidRandomSeedAlpha(self):
        test_input = ["-r", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive integer value' in output)

    def testInvalidRandomSeedNegNum(self):
        test_input = ["-r", "-1"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive integer value' in output)

    def testInvalidRandomSeed0(self):
        test_input = ["-r", "0"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive integer value' in output)

    def testInvalidRandomSeedTooBig(self):
        just_over_max = str(2**32)
        test_input = ["-r", just_over_max]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive integer value' in output)

    def testInvalidExtension(self):
        test_input = ["-r", "10", "-f", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('currently supported types' in output)

    def testAlphaSGRatio(self):
        test_input = ["-r", "10", "-sg", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive floats' in output)

    def testNegSGRatio(self):
        test_input = ["-r", "10", "-sg", "-0.1"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive floats' in output)

    def testNegSimLen(self):
        test_input = ["-r", "10", "-l", "-0.1"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive number' in output)

    def testNegIniMonos(self):
        test_input = ["-r", "10", "-i", "-4"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive integer' in output)

    def testFractionalIniMonos(self):
        test_input = ["-r", "10", "-i", "4.5"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive integer' in output)

    def testZeroSimTime(self):
        test_input = ["-r", "10", "-l", "0"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('positive' in output)

    def testAlphaIniMonos(self):
        test_input = ["-r", "10", "-i", "ghost"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive integer' in output)

    def testNegMaxMonos(self):
        test_input = ["-r", "10", "-m", "-8"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive integer' in output)

    def testFractionalMaxMonos(self):
        test_input = ["-r", "10", "-m", "12.1"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive integer' in output)

    def testFewerMaxThanIniMonos(self):
        test_input = ["-r", "10", "-i", "6", "-m", "4"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue('is less than' in output)
        with capture_stdout(main, test_input) as output:
            self.assertTrue("Lignin KMC created 6 monomers" in output)

    def testBadImageSize(self):
        test_input = ["-r", "10", "-s", "4"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("two positive numbers" in output)

    def testAlphaAddRate(self):
        test_input = ["-a", "ghost"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("positive floats" in output)

    def testZeroAddRate(self):
        test_input = ["-a", "0"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("positive floats" in output)

    def testSizeWithSpace(self):
        test_input = ["-o", DEF_PNG_OUT, "-s", "300", "400"]
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

    def testFewerMaxThanMinMonos(self):
        test_input = ["-r", "10", "-i", "20", "-m", "10", ]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("initial" in output)


class TestCreateLigninNormalUse(unittest.TestCase):
    def testMostlyArgs(self):
        test_input = ["-r", "10", "-b"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:"
        good_bond_summary = "composed of the following bond types and number:"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n"
        good_rcf_bond_summary = "with the following remaining bond types and number:"
        good_smiles = "CO"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(OPENING_MSG in output)
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)
            self.assertTrue(good_smiles in output)
        with capture_stderr(main, test_input) as output:
            self.assertFalse(output)

    def testSaveSmi(self):
        try:
            silent_remove(TEST_SMI_OUT)
            test_input = ["-o", TEST_SMI_BASENAME, "-d", SUB_DATA_DIR]
            main(test_input)
            with open(TEST_SMI_OUT, "r") as f:
                smi_str = f.readlines()
            self.assertTrue("CO" in smi_str[0])
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)
            pass
#
    def testDirInBasename(self):
        # This should ignore the temp_dir; will throw error if it doesn't
        try:
            silent_remove(TEST_SMI_OUT)
            test_input = ["-r", "10", "-o", TEST_SMI_OUT, "-d", TEMP_DIR, "-a", "1.0"]
            # main(test_input)
            with capture_stderr(main, test_input) as output:
                self.assertFalse(output)
            with open(TEST_SMI_OUT, "r") as f:
                smi_str = f.readlines()
            self.assertTrue("CO" in smi_str[0])
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)
            pass

    def testMakeSubDir(self):
        try:
            silent_remove(TEST_SMI_OUT_TEMP_DIR)
            test_input = ["-r", "10", "-d", TEMP_DIR, "-o", TEST_SMI_BASENAME, "-a", "1.0"]
            main(test_input)
            with open(TEST_SMI_OUT_TEMP_DIR, "r") as f:
                smi_str = f.readlines()
            self.assertTrue("CO" in smi_str[0])
        finally:
            silent_remove(TEST_SMI_OUT_TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(TEMP_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            silent_remove(INNER_TEMP_DIR, disable=DISABLE_REMOVE)
            pass

    def testSmallConfig(self):
        test_input = ["-c", SMALL_INI, "-r", "11", "-l", "2.0", "-a", "1e2", "-b", "-m", "12"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 12 monomers, which formed"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    "
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    "
        good_smi_summary = "SMILES representation:"
        with capture_stdout(main, test_input) as output:
            for summary_str in [good_chain_summary, good_bond_summary, good_rcf_chain_summary, good_rcf_bond_summary,
                                good_smi_summary]:
                self.assertTrue(summary_str in output)

    def testSaveJSONPathInOutName(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json", "-o", DEF_JSON_OUT, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()[0]
            for json_part in GOOD_JSON_PARTS:
                print(json_part)
                self.assertTrue(json_part in json_str)
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)
            pass

    def testSaveJSON(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json", "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()[0]
            for json_part in GOOD_JSON_PARTS:
                print(json_part)
                self.assertTrue(json_part in json_str)
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)

    def testSaveJSONTcl(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json tcl", "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()[0]
            self.assertTrue(GOOD_JSON_PARTS[0] in json_str)
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)
            silent_remove(DEF_TCL_OUT, disable=DISABLE_REMOVE)
            pass

    def testSaveJSONCommaPNG(self):
        # Smoke test on png with default image size
        try:
            silent_remove(DEF_PNG_OUT)
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json, png", "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            self.assertTrue(os.path.isfile(DEF_PNG_OUT))
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()[0]
            self.assertTrue(GOOD_JSON_PARTS[0] in json_str)
        finally:
            silent_remove(DEF_PNG_OUT, disable=DISABLE_REMOVE)
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)
            pass

    def testSavePNGSVG(self):
        # Smoke test only (not comparing images)
        try:
            silent_remove(DEF_PNG_OUT)
            silent_remove(DEF_SVG_OUT)
            test_input = ["-r", "10", "-f", "svg", "-o", DEF_BASENAME + '.png', "-d", SUB_DATA_DIR, "-m", "6",
                          "-s", "900 900"]
            main(test_input)
            self.assertTrue(os.path.isfile(DEF_PNG_OUT))
            self.assertTrue(os.path.isfile(DEF_SVG_OUT))
        finally:
            silent_remove(DEF_PNG_OUT, disable=DISABLE_REMOVE)
            silent_remove(DEF_SVG_OUT, disable=DISABLE_REMOVE)

    def testSavePNGSVGNewSize(self):
        # Smoke test only (not comparing images)
        try:
            test_input = ["-r", "10", "-f", "png, svg", "-d", SUB_DATA_DIR, "-s", "(900, 900)"]
            main(test_input)
            self.assertTrue(os.path.isfile(DEF_PNG_OUT))
            self.assertTrue(os.path.isfile(DEF_SVG_OUT))
        finally:
            silent_remove(DEF_PNG_OUT, disable=DISABLE_REMOVE)
            silent_remove(DEF_SVG_OUT, disable=DISABLE_REMOVE)
            pass

    def testAltSGRatio(self):
        test_input = ["-r", "8", "-sg", "2.5", "-a", "1.0"]
        # main(test_input)
        good_smiles = "CO"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testAltIniMaxMonosSimLen(self):
        test_input = ["-r", "10", "-i", "8", "-m", "12", "-l", "0.02", "-a", "1.0"]
        # main(test_input)
        good_smiles = "CO"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testTCLGenOptions(self):
        try:
            test_input = ["-r", "8", "-i", "4", "-m", "4", "-f", "tcl", "-d", SUB_DATA_DIR,
                          "--chain_id", "1", "--psf_fname", "birch", "--toppar_dir", "", "-a", "1.0", "-x"]
            main(test_input)
            self.assertFalse(diff_lines(DEF_TCL_OUT, GOOD_TCL_OPTIONS_OUT))
        finally:
            silent_remove(DEF_TCL_OUT, disable=DISABLE_REMOVE)
            pass


class TestDynamics(unittest.TestCase):
    def testDyn1(self):
        expected_files = [BOND_V_STEP_PNG, MONO_V_STEP_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "10", "-i", "3", "-m", "15", "-dy", "-a", "1e6", "-x"]
            # main(test_input)
            with capture_stdout(main, test_input) as output:
                self.assertTrue("Lignin KMC created 15 monomers, which formed:" in output)
                self.assertEqual(output.count("BO4:"), 1)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testDyn2(self):
        expected_files = [BOND_V_STEP_PNG, MONO_V_STEP_PNG]
        try:
            for file_name in expected_files:
                silent_remove(file_name)
            test_input = ["-r", "10", "-i", "3", "-m", "20", "-dy", "-a", "1e6", "-n", "2", "-x"]
            # main(test_input)
            # testing a piece of output from each of 2 repeats
            with capture_stdout(main, test_input) as output:
                self.assertEqual(output.count("BO4:"), 2)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testDyn4(self):
        # also has multiple sg_ratio; make sure
        expected_files = [BOND_V_STEP_PNG, MONO_V_STEP_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "10", "-i", "3", "-m", "20", "-dy", "-a", "1e6", "-n", "4", "-x"]
            # main(test_input)
            with capture_stdout(main, test_input) as output:
                self.assertEqual(output.count("BO4:"), 4)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testDynPlot1(self):
        # smoke test
        expected_files = [PLOT_BOND_V_SG6_PNG, PLOT_BOND_V_STEP_PNG, PLOT_MONO_V_STEP_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "10", "-i", "6", "-m", "18", "-a", "1e6", "-dy", "-p", "-d", PLOT_DIR, "-x"]
            main(test_input)
            with capture_stdout(main, test_input) as output:
                self.assertTrue("Lignin KMC created 18 monomers, which formed" in output)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            silent_remove(PLOT_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            pass

    def testSGPlot3(self):
        # also has multiple sg_ratio; smoke test only (that files are created, but not testing content
        expected_files = [PLOT_BOND_V_SG6_PNG, PLOT_BOND_V_SG8_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "6", "-i", "8", "-m", "16", "-a", "1e8, 1e6", "-sg", "5,10",
                          "-n", "3", "-p", "-d", PLOT_DIR, "-x"]
            main(test_input)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            silent_remove(TEMP_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            pass

    def testCheckForValenceError(self):
        random_seed = 1
        test_input = ["-i", "5", "-m", "200", "-a", "1", "-sg", "1, 3, 5, 10", "-n", "3", "-r", str(random_seed)]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            if output:
                print("Encountered error:\n", output)
            self.assertFalse(output)

    def testCheckSBonding(self):
        random_seed = 10
        test_input = ["-i", "2", "-m", "64", "-a", "1e2", "-sg", "10000000", "-r", str(random_seed)]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertFalse("Exiting program due to error in producing output" in output)

    # Added as an easy way to run this for its production output
    def testProduction(self):
        # new_out_dir = os.path.join(DATA_DIR, 'new_plots')
        plot_input = ["-x", "-p",
                      # next line: testing
                      "-i", "5", "-m", "100", "-a", "1", "-sg", "1, 10", "-n", "3", "-d", TEMP_DIR,
                      # # alt lines: production
                      # "-i", "2", "-m", "500", "-l", "1e5", "-a", "1e8, 1e6, 1e4, 1e2, 1",
                      # "-sg", "0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10", "-n", "100", "-d", new_out_dir,
                      ]
        try:
            with capture_stderr(main, plot_input) as output:
                self.assertFalse(output)
        finally:
            silent_remove(TEMP_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            pass

    def testLargeN(self):
        plot_input = ["-x",
                      # "-p",
                      # next line: testing
                      # "-i", "5", "-m", "100", "-a", "1", "-sg", "1, 10", "-n", "3", "-d", TEMP_DIR,
                      # larger number for better stats
                      # -i is initial number of monomers, m is maximum
                      # "-i", "5", "-m", "250", "-a", "1",
                      # "-sg", "0.5, 1, 5", "-n", "5000", "-d", TEMP_DIR,
                      "-i", "5", "-m", "250", "-a", "1",
                      "-sg", "0.5, 1, 5", "-n", "3", "-d", TEMP_DIR,
                      # # alt lines: production
                      # "-i", "2", "-m", "500", "-l", "1e5", "-a", "1e8, 1e6, 1e4, 1e2, 1",
                      # "-sg", "0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10", "-n", "100", "-d", new_out_dir,
                      ]
        try:
            with capture_stderr(main, plot_input) as output:
                self.assertFalse(output)
        finally:
            silent_remove(TEMP_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            pass
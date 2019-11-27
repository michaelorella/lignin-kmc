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


# Data #

# testing pieces of json, skipping parts that refer to version; more thorough testing is beyond scope
GOOD_JSON_PARTS = ['"bonds":[{"bo":2,"atoms":[0,1]},{"atoms":[1,2]},{"bo":2,"atoms":[2,3]},{"atoms":[3,4]},'
                   '{"bo":2,"atoms":[4,5]},{"atoms":[5,0]},{"atoms":[0,6]},{"bo":2,"atoms":[6,7],',
                   '{"atoms":[8,9]},{"atoms":[2,10]},{"atoms":[10,11]},{"atoms":[3,12]},{"bo":2,"atoms":[13,14]},'
                   '{"atoms":[14,15]},{"bo":2,"atoms":[15,16]},{"atoms":[16,17]},{"bo":2,"atoms":[17,18]},'
                   '{"atoms":[18,13]},{"atoms":[13,19]},{"atoms":[19,20]},{"atoms":[20,21]},{"atoms":[21,22]},'
                   '{"atoms":[15,23]},{"atoms":[23,24]},{"atoms":[16,25]},{"atoms":[17,26]},{"atoms":[26,27]},'
                   '{"bo":2,"atoms":[28,29]},{"atoms":[29,30]},{"bo":2,"atoms":[30,31]},{"atoms":[31,32]},'
                   '{"bo":2,"atoms":[32,33]},{"atoms":[33,28]},{"atoms":[28,34]},{"atoms":[34,35]},{"atoms":[35,36]},'
                   '{"atoms":[36,37]},{"atoms":[30,38]},{"atoms":[38,39]},{"atoms":[31,40]},{"atoms":[32,41]},'
                   '{"atoms":[41,42]},{"bo":2,"atoms":[43,44]},{"atoms":[44,45]},{"bo":2,"atoms":[45,46]},'
                   '{"atoms":[46,47]},{"bo":2,"atoms":[47,48]},{"atoms":[48,43]},{"atoms":[43,49]},{"atoms":[49,50]},'
                   '{"atoms":[50,51]},{"atoms":[51,52]},{"atoms":[45,53]},{"atoms":[53,54]},{"atoms":[46,55]},'
                   '{"bo":2,"atoms":[56,57]},{"atoms":[57,58]},{"bo":2,"atoms":[58,59]},{"atoms":[59,60]},'
                   '{"bo":2,"atoms":[60,61]},{"atoms":[61,56]},{"atoms":[56,62]},{"atoms":[62,63]},{"atoms":[63,64]},'
                   '{"atoms":[64,65]},{"atoms":[58,66]},{"atoms":[66,67]},{"atoms":[59,68]},{"atoms":[60,69]},'
                   '{"atoms":[69,70]},{"bo":2,"atoms":[71,72]},{"atoms":[72,73]},{"bo":2,"atoms":[73,74]},'
                   '{"atoms":[74,75]},{"bo":2,"atoms":[75,76]},{"atoms":[76,71]},{"atoms":[71,77]},{"atoms":[77,78]},'
                   '{"atoms":[78,79]},{"atoms":[79,80]},{"atoms":[73,81]},{"atoms":[81,82]},{"atoms":[74,83]},'
                   '{"bo":2,"atoms":[84,85]},{"atoms":[85,86]},{"bo":2,"atoms":[86,87]},{"atoms":[87,88]},'
                   '{"bo":2,"atoms":[88,89]},{"atoms":[89,84]},{"atoms":[84,90]},{"atoms":[90,91]},{"atoms":[91,92]},'
                   '{"atoms":[92,93]},{"atoms":[86,94]},{"atoms":[94,95]},{"atoms":[87,96]},{"bo":2,"atoms":[97,98]},'
                   '{"atoms":[98,99]},{"bo":2,"atoms":[99,100]},{"atoms":[100,101]},{"bo":2,"atoms":[101,102]},',
                   '"aromaticBonds":[0,1,2,3,4,5,13,14,15,16,17,18,28,29,30,31,32,33,43,44,45,46,47,48,56,57,58,59,'
                   '60,61,71,72,73,74,75,76,84,85,86,87,88,89,97,98,99,100,101,102,112,113,114,115,116,117,127,128,',
                   '"atomRings":[[0,5,4,3,2,1],[12,3,4,20,19],[14,15,16,17,18,13],[29,30,31,32,33,28],']


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


class TestMoreWarnings(unittest.TestCase):
    def testFewerMaxThanMinMonos(self):
        test_input = ["-r", "10", "-i", "20", "-m", "10", ]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("initial" in output)


class TestCreateLigninNormalUse(unittest.TestCase):
    def testDefArgs(self):
        test_input = ["-r", "10"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 oligomer(s) of chain length 10, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    6 " \
                            "    BB:    0     B5:    3     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       5 monomer(s) (chain length 1)\n       1 dimer(s) (chain length 2)\n" \
                                 "       1 trimer(s) (chain length 3)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     " \
                                "BB:    0     B5:    3     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_smiles = "COc1cc(C(O)C(CO)Oc2c(OC)cc(C(O)C(CO)Oc3c(OC)cc(C4Oc5c(OC)cc(/C=C/CO)cc5C4CO)cc3OC)cc2OC)" \
                      "ccc1OC(CO)C(O)c1cc(OC)c(OC(CO)C(O)c2cc(OC)c3c(c2)C(CO)C(c2cc(OC)c4c(c2)C(CO)C(c2cc(OC)" \
                      "c(OC(CO)C(O)c5cc(OC)c(OC(CO)C(O)c6cc(OC)c([O])c(OC)c6)c(OC)c5)c(OC)c2)O4)O3)c(OC)c1"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(OPENING_MSG in output)
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)
            self.assertTrue(good_smiles in output)

    def testSaveSmi(self):
        try:
            test_input = ["-r", "10", "-o", TEST_SMI_BASENAME, "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)

    def testDirInBasename(self):
        # This should ignore the temp_dir; will throw error if it doesn't
        try:
            test_input = ["-r", "10", "-o", TEST_SMI_OUT, "-d", TEMP_DIR, "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)

    def testMakeSubDir(self):
        try:
            test_input = ["-r", "10", "-d", TEMP_DIR, "-o", TEST_SMI_BASENAME, "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT_TEMP_DIR, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT_TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(INNER_TEMP_DIR, disable=DISABLE_REMOVE)

    def testSmallConfig(self):
        test_input = ["-c", SMALL_INI, "-r", "11", "-l", "1.0", "-a", "1e2"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 oligomer(s) of chain length 10, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    6" \
                            "     BB:    1     B5:    2     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       4 monomer(s) (chain length 1)\n" \
                                 "       3 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0" \
                                "     BB:    1     B5:    2     B1:    0    5O4:    0    AO4:    0     55:    0"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)

    def testSaveJSONPathInOutName(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json", "-o", DEF_JSON_OUT, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            for json_part in GOOD_JSON_PARTS:
                self.assertTrue(json_part in json_str[0])
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)

    def testSaveJSON(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json", "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            self.assertTrue(GOOD_JSON_PARTS[0] in json_str[0])
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)

    def testSaveJSONTcl(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json tcl", "-d", SUB_DATA_DIR, "-a", "1.0"]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            self.assertTrue(GOOD_JSON_PARTS[0] in json_str[0])
            self.assertFalse(diff_lines(DEF_TCL_OUT, GOOD_DEF_TCL_OUT))
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
                json_str = f.readlines()
            self.assertTrue(GOOD_JSON_PARTS[0] in json_str[0])
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
        good_smiles = "COc1cc(C2Oc3c(OC)cc(/C=C/CO)cc3C2CO)ccc1OC(CO)C(O)c1cc(OC)c2c(c1)C(CO)C(c1cc(OC)c(OC(CO)C(O)" \
                      "c3cc(OC)c(OC(CO)C(O)c4cc(OC)c(OC(CO)C(O)c5cc(OC)c(OC(CO)C(O)c6cc(OC)c(OC(CO)C(O)c7cc(OC)" \
                      "c(OC(CO)C(O)c8cc(OC)c([O])c(OC)c8)c(OC)c7)c(OC)c6)c(OC)c5)c(OC)c4)c(OC)c3)c(OC)c1)O2 "
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testAltIniMaxMonosSimLen(self):
        test_input = ["-r", "10", "-i", "8", "-m", "12", "-l", "0.002", "-a", "1.0"]
        # main(test_input)
        good_smiles = "COc1cc(/C=C/CO)ccc1O.COc1cc(C2OCC3C(c4cc(OC)c(OC(CO)C(O)c5cc(OC)c([O])c(OC)c5)c(Oc5c(OC)cc" \
                      "(C6OCC7C(c8cc(OC)c(Oc9cc(C%10Oc%11c(OC)cc(/C=C/CO)cc%11C%10CO)cc(OC)c9[O])c(OC)c8)OCC67)" \
                      "cc5OC)c4)OCC23)ccc1OC(CO)C(O)c1cc(OC)c([O])c(OC)c1"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testPSFGenOptions(self):
        try:
            test_input = ["-r", "8", "-i", "4", "-m", "4", "-f", "tcl", "-d", SUB_DATA_DIR,
                          "--chain_id", "1", "--psf_fname", "birch", "--toppar_dir", "", "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(DEF_TCL_OUT, GOOD_TCL_OPTIONS_OUT))
        finally:
            silent_remove(DEF_TCL_OUT, disable=DISABLE_REMOVE)
            pass


class TestDynamics(unittest.TestCase):
    def testSmallNumMonos(self):
        try:
            # for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
            #     silent_remove(fname)
            test_input = ["-r", "10", "-m", "20", "-dy"]
            main(test_input)
#             for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
#                 self.assertTrue(os.path.isfile(fname))
        finally:
            # for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
            #     silent_remove(fname, disable=DISABLE_REMOVE)
            pass
#
#     def testSmallNumMonosNoDynamics(self):
#         try:
#             for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
#                 silent_remove(fname)
#             test_input = ["-r", "10", "-m", "20"]
#             main(test_input)
#             self.assertTrue(os.path.isfile(DEF_BOND_PNG))
#             self.assertFalse(os.path.isfile(DEF_MONO_PNG))
#         finally:
#             for fname in [DEF_BOND_PNG, DEF_MONO_PNG]:
#                 silent_remove(fname, disable=DISABLE_REMOVE)
#             pass
#
#     def testMultOptions(self):
#         expected_pngs = [BOND_OPT_1_PNG, BOND_OPT_2_PNG,
#                          MONO_OPT_1_PNG, MONO_OPT_2_PNG, MONO_OPT_3_PNG, MONO_OPT_4_PNG]
#         try:
#             for fname in expected_pngs:
#                 silent_remove(fname)
#             test_input = ["-r", "10", "-m", "20", "-a", "1e8, 1e4", "-sg", "0.25, 3", "-d", SUB_DATA_DIR, "-dy"]
#             main(test_input)
#             for fname in expected_pngs:
#                 self.assertTrue(os.path.isfile(fname))
#         finally:
#             for fname in expected_pngs:
#                 silent_remove(fname, disable=DISABLE_REMOVE)
#             pass
#
#     # Do not include the following in test coverage--just a quick way to run this for its production output
#     def testProduction(self):
#         new_out_dir = os.path.join(DATA_DIR, 'new_plots')
#
#         # more efficient to just look at "1e8, 1e6, 1e4" and "1,  3, 5, 10"
#         input_base = ["-i", "5", "-m", "200", "-a", "1e8, 1e6, 1e4, 1e2, 1",
#                       # "-sg", "0.1, 1, 10", "-d", new_out_dir]
#                       "-sg", "0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10", "-d", new_out_dir]
#         input_1 = input_base
#         input_2 = input_base + ["-e"]
#
#         for prod_input in [input_1, input_2]:
#             main(prod_input)
#
#     def testProduction2(self):
#         new_out_dir = os.path.join(DATA_DIR, 'new_plots')
#
#         # Is the S-S oligomer-oligomer bond actually being created????
#
#         # more efficient to just look at "1e8, 1e6, 1e4" and "1,  3, 5, 10"
#         input_base = ["-i", "5", "-m", "200", "-a", "1e8, 1e6",
#                       # "-sg", "0.1, 1, 10", "-d", new_out_dir]
#                       "-sg", "5, 10", "-d", new_out_dir]
#         input_1 = input_base
#         input_2 = input_base + ["-e"]
#
#         for prod_input in [input_1, input_2]:
#             main(prod_input)

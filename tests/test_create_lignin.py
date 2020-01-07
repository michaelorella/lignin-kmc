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
GOOD_JSON_PARTS = ['"bonds":[{"bo":2,"atoms":[0,1]},{"atoms":[1,2]},{"bo":2,"atoms":[2,3]},{"atoms":[3,4]},'
                   '{"bo":2,"atoms":[4,5]},{"atoms":[5,0]},{"atoms":[0,6]},{"atoms":[6,7]},{"atoms":[7,8]},'
                   '{"atoms":[8,9]},{"atoms":[2,10]},{"atoms":[10,11]},{"atoms":[3,12]},{"bo":2,"atoms":[13,14]},'
                   '{"atoms":[14,15]},{"bo":2,"atoms":[15,16]},{"atoms":[16,17]},{"bo":2,"atoms":[17,18]},'
                   '{"atoms":[18,13]},{"atoms":[13,19]},{"atoms":[19,20]},{"atoms":[20,21]},{"atoms":[21,22]},'
                   '{"atoms":[15,23]},{"atoms":[23,24]},{"atoms":[16,25]},{"bo":2,"atoms":[26,27]},{"atoms":[27,28]},'
                   '{"bo":2,"atoms":[28,29]},{"atoms":[29,30]},{"bo":2,"atoms":[30,31]},{"atoms":[31,26]},'
                   '{"atoms":[26,32]},{"atoms":[32,33]},{"atoms":[33,34]},{"atoms":[34,35]},{"atoms":[28,36]},'
                   '{"atoms":[36,37]},{"atoms":[29,38]},{"bo":2,"atoms":[39,40]},{"atoms":[40,41]},'
                   '{"bo":2,"atoms":[41,42]},{"atoms":[42,43]},{"bo":2,"atoms":[43,44]},{"atoms":[44,39]},'
                   '{"atoms":[39,45]},{"atoms":[45,46]},{"atoms":[46,47]},{"atoms":[47,48]},{"atoms":[41,49]},'
                   '{"atoms":[49,50]},{"atoms":[42,51]},{"atoms":[43,52]},{"atoms":[52,53]},{"bo":2,"atoms":[54,55]},'
                   '{"atoms":[55,56]},{"bo":2,"atoms":[56,57]},{"atoms":[57,58]},{"bo":2,"atoms":[58,59]},'
                   '{"atoms":[59,54]},{"atoms":[54,60]},{"atoms":[60,61]},{"atoms":[61,62]},{"atoms":[62,63]},'
                   '{"atoms":[56,64]},{"atoms":[64,65]},{"atoms":[57,66]},{"atoms":[58,67]},{"atoms":[67,68]},'
                   '{"bo":2,"atoms":[69,70]},{"atoms":[70,71]},{"bo":2,"atoms":[71,72]},{"atoms":[72,73]},'
                   '{"bo":2,"atoms":[73,74]},{"atoms":[74,69]},{"atoms":[69,75]},{"atoms":[75,76]},{"atoms":[76,77]},'
                   '{"atoms":[77,78]},{"atoms":[71,79]},{"atoms":[79,80]},{"atoms":[72,81]},{"bo":2,"atoms":[82,83]},'
                   '{"atoms":[83,84]},{"bo":2,"atoms":[84,85]},{"atoms":[85,86]},{"bo":2,"atoms":[86,87]},'
                   '{"atoms":[87,82]},{"atoms":[82,88]},{"atoms":[88,89]},{"atoms":[89,90]},{"atoms":[90,91]},'
                   '{"atoms":[84,92]},{"atoms":[92,93]},{"atoms":[85,94]},{"bo":2,"atoms":[95,96]},{"atoms":[96,97]},'
                   '{"bo":2,"atoms":[97,98]},{"atoms":[98,99]},{"bo":2,"atoms":[99,100]},{"atoms":[100,95]},'
                   '{"atoms":[95,101]},{"atoms":[101,102]},{"atoms":[102,103]},{"atoms":[103,104]},{"atoms":[97,105]},'
                   '{"atoms":[105,106]},{"atoms":[98,107]},{"bo":2,"atoms":[108,109]},{"atoms":[109,110]},'
                   '{"bo":2,"atoms":[110,111]},{"atoms":[111,112]},{"bo":2,"atoms":[112,113]},{"atoms":[113,108]},'
                   '{"atoms":[108,114]},{"atoms":[114,115]},{"atoms":[115,116]},{"atoms":[116,117]},'
                   '{"atoms":[110,118]},{"atoms":[118,119]},{"atoms":[111,120]},{"bo":2,"atoms":[121,122]},'
                   '{"atoms":[122,123]},{"bo":2,"atoms":[123,124]},{"atoms":[124,125]},{"bo":2,"atoms":[125,126]},'
                   '{"atoms":[126,121]},{"atoms":[121,127]},{"atoms":[127,128]},{"atoms":[128,129]},'
                   '{"atoms":[129,130]},{"atoms":[123,131]},{"atoms":[131,132]},{"atoms":[124,133]},'
                   '{"atoms":[125,134]},{"atoms":[134,135]},{"atoms":[20,7]},{"atoms":[19,9]},{"atoms":[6,22]},'
                   '{"atoms":[17,33]},{"atoms":[32,25]},{"atoms":[46,30]},{"atoms":[45,38]},{"atoms":[4,61]},'
                   '{"atoms":[60,12]},{"atoms":[76,66]},{"atoms":[75,136]},{"atoms":[81,89]},{"atoms":[88,137]},'
                   '{"atoms":[102,86]},{"atoms":[101,94]},{"atoms":[115,51]},{"atoms":[114,138]},{"atoms":[99,128]},'
                   '{"atoms":[127,107]}],',
                   '"aromaticAtoms":[0,1,2,3,4,5,13,14,15,16,17,18,26,27,28,29,30,31,39,40,41,42,43,44,54,55,56,57,58,'
                   '59,69,70,71,72,73,74,82,83,84,85,86,87,95,96,97,98,99,100,108,109,110,111,112,113,121,122,123,124,'
                   '125,126]']


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
    def testMostlyDefArgs(self):
        test_input = ["-r", "10", "-b"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 oligomer(s) of chain length 10, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    3 " \
                            "    BB:    1     B5:    5     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       2 monomer(s) (chain length 1)\n       1 trimer(s) (chain length 3)\n" \
                                 "       1 oligomer(s) of chain length 5, with branching coefficient 0.0"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     " \
                                "BB:    1     B5:    5     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_smiles = "COc1cc(C(O)C(CO)Oc2c(OC)cc(C3Oc4c(OC)cc(C5Oc6c(OC)cc(C7OCC8C(c9cc(OC)c%10c(c9)C(CO)C(c9cc(OC)c" \
                      "(OC(CO)C(O)c%11ccc(OC(CO)C(O)c%12cc(OC)c%13c(c%12)C(CO)C(c%12cc(OC)c%14c(c%12)C(CO)C(c%12cc(O" \
                      "C)c([O])c(OC)c%12)O%14)O%13)c(OC)c%11)c(OC)c9)O%10)OCC78)cc6C5CO)cc4C3CO)cc2OC)ccc1[O]"

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
            pass

    def testDirInBasename(self):
        # This should ignore the temp_dir; will throw error if it doesn't
        try:
            test_input = ["-r", "10", "-o", TEST_SMI_OUT, "-d", TEMP_DIR, "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)
            pass

    def testMakeSubDir(self):
        try:
            test_input = ["-r", "10", "-d", TEMP_DIR, "-o", TEST_SMI_BASENAME, "-a", "1.0"]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT_TEMP_DIR, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT_TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(INNER_TEMP_DIR, disable=DISABLE_REMOVE)
            pass

    def testSmallConfig(self):
        test_input = ["-c", SMALL_INI, "-r", "11", "-l", "1.0", "-a", "1e2", "-b"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 oligomer(s) of chain length 10, with branching coefficient 0.0"
        good_bond_summary = "composed of the following bond types and number:\n    BO4:    5" \
                            "     BB:    1     B5:    3     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       2 monomer(s) (chain length 1)\n" \
                                 "       4 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0    " \
                                " BB:    1     B5:    3     B1:    0    5O4:    0    AO4:    0     55:    0"
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
            pass

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
        test_input = ["-r", "6", "-sg", "2.5", "-a", "1.0"]
        # main(test_input)
        good_smiles = " COc1cc(C2Oc3c(OC)cc(C(O)C(CO)Oc4c(OC)cc(C5Oc6c(OC)cc(C(O)C(CO)Oc7c(OC)cc(C(O)C(CO)Oc8c(OC)cc" \
                      "(C9Oc%10c(OC)cc(/C=C/CO)cc%10C9CO)cc8OC)cc7OC)cc6C5CO)cc4OC)cc3C2CO)ccc1OC(CO)C(O)c1cc(OC)c2c" \
                      "(c1)C(CO)C(c1cc(OC)c(OC(CO)C(O)c3cc(OC)c([O])c(OC)c3)c(OC)c1)O2"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testAltIniMaxMonosSimLen(self):
        test_input = ["-r", "10", "-i", "8", "-m", "12", "-l", "0.02", "-a", "1.0"]
        # main(test_input)
        good_smiles = "COc1cc(/C=C/CO)ccc1O.COc1cc(C2OCC3C(c4cc(OC)c(Oc5cc(C6Oc7c(OC)cc(/C=C/CO)cc7C6CO)cc(OC)c5[O])" \
                      "c(Oc5c(OC)cc(C6Oc7c(OC)cc(/C=C/CO)cc7C6CO)cc5OC)c4)OCC23)cc(Oc2c(OC)cc(C3Oc4c(OC)cc(/C=C/CO)c" \
                      "c4C3CO)cc2OC)c1[O]"
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
                self.assertTrue("Lignin KMC created 15 monomers, which formed:\n       "
                                "1 dimer(s) (chain length 2)\n       "
                                "1 oligomer(s) of chain length 13, with branching coefficient 0.077" in output)
                self.assertTrue("BO4:    4     BB:    2     B5:    4     B1:    0    5O4:    3    AO4:    0     "
                                "55:    0" in output)
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
                self.assertTrue("BO4:    7     BB:    2     B5:    5     B1:    0    5O4:    5    AO4:    0     "
                                "55:    0" in output)
                self.assertTrue("BO4:    6     BB:    6     B5:    1     B1:    0    5O4:    5    AO4:    0     "
                                "55:    0" in output)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testDyn4(self):
        # also has multiple sg_ratio
        expected_files = [BOND_V_STEP_PNG, MONO_V_STEP_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "10", "-i", "3", "-m", "20", "-dy", "-a", "1e6", "-n", "4", "-x"]
            # main(test_input)
            with capture_stdout(main, test_input) as output:
                self.assertTrue("BO4:    7     BB:    2     B5:    5     B1:    0    5O4:    5    AO4:    0     "
                                "55:    0" in output)
                self.assertTrue("BO4:    6     BB:    6     B5:    1     B1:    0    5O4:    5    AO4:    0     "
                                "55:    0" in output)
                self.assertTrue("BO4:    6     BB:    2     B5:    5     B1:    0    5O4:    6    AO4:    0     "
                                "55:    0" in output)
                self.assertTrue("BO4:    9     BB:    4     B5:    2     B1:    0    5O4:    3    AO4:    0     "
                                "55:    1" in output)
            for fname in expected_files:
                self.assertTrue(os.path.isfile(fname))
        finally:
            for fname in expected_files:
                silent_remove(fname, disable=DISABLE_REMOVE)
            pass

    def testDynPlot1(self):
        expected_files = [PLOT_BOND_V_SG6_PNG, PLOT_BOND_V_STEP_PNG, PLOT_MONO_V_STEP_PNG]
        try:
            for fname in expected_files:
                silent_remove(fname)
            test_input = ["-r", "10", "-i", "6", "-m", "18", "-a", "1e6", "-dy", "-p", "-d", PLOT_DIR, "-x"]
            # main(test_input)
            with capture_stdout(main, test_input) as output:
                self.assertTrue("1 oligomer(s) of chain length 18, with branching coefficient 0.111" in output)
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

    def testTryOvercomeB1Error(self):
        random_number = 1
        test_input = ["-i", "5", "-m", "200", "-a", "1", "-e",
                      "-sg", "1, 3, 5, 10", "-n", "3", "-r", str(random_number)]
        with capture_stderr(main, test_input) as output:
            self.assertTrue("Will repeat step" in output)
            self.assertTrue("error in producing output" in output)

    # Do not include the following in test coverage--just an easy way to run this for its production output
    def testProduction(self):
        # new_out_dir = os.path.join(DATA_DIR, 'new_plots')
        # more efficient to just look at "1e8, 1e6, 1e4" and "1,  3, 5, 10"
        # for a full list: "-a",  "1e8, 1e6, 1e4, 1e2, 1"
        #                  "-sg", "0.1, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5, 10",
        #                  "-n", "5"m "-d", new_out_dir
        plot_input = ["-i", "5", "-m", "200", "-a", "1", "-x",
                      "-sg", "1, 3, 5, 10", "-n", "3", "-p", "-d", TEMP_DIR]
        try:
            with capture_stderr(main, plot_input) as output:
                self.assertFalse(output)
        finally:
            silent_remove(TEMP_DIR, dir_with_files=True, disable=DISABLE_REMOVE)
            pass

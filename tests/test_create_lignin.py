#!/usr/bin/env python3

import logging
import os
import unittest
from ligninkmc.create_lignin import main, OPENING_MSG, DEF_BASENAME
from common_wrangler.common import capture_stderr, capture_stdout, silent_remove, diff_lines

__author__ = 'hmayes'

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DISABLE_REMOVE = logger.isEnabledFor(logging.DEBUG)

# Constants #
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'test_data')
SUB_DATA_DIR = os.path.join(DATA_DIR, 'run_kmc')
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
SMALL_INI = os.path.join(SUB_DATA_DIR, "small_config.ini")
TEST_SMI_BASENAME = "test_lignin.smi"
TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, TEST_SMI_BASENAME)
GOOD_TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, "good_test_lignin.smi")
TEST_SMI_OUT_TEMP_DIR = os.path.join(TEMP_DIR, TEST_SMI_BASENAME)


# Data #

# testing pieces of json, skipping parts that refer to version; more thorough testing is beyond scope
good_json_parts = ['"bonds":[{"bo":2,"atoms":[0,1]},{"atoms":[1,2]},{"bo":2,"atoms":[2,3]},{"atoms":[3,4]},',
                   '{"atoms":[7,8]},{"atoms":[8,9]},{"atoms":[2,10]},{"atoms":[10,11]},{"atoms":[3,12]},'
                   '{"bo":2,"atoms":[13,14]},{"atoms":[14,15]},{"bo":2,"atoms":[15,16]},{"atoms":[16,17]},'
                   '{"bo":2,"atoms":[17,18]},{"atoms":[18,13]},{"atoms":[13,19]},{"atoms":[19,20]},{"atoms":[20,21]},'
                   '{"atoms":[21,22]},{"atoms":[15,23]},{"atoms":[23,24]},{"atoms":[16,25]},{"atoms":[17,26]},'
                   '{"atoms":[26,27]},{"bo":2,"atoms":[28,29]},{"atoms":[29,30]},{"bo":2,"atoms":[30,31]},'
                   '{"atoms":[31,32]},{"bo":2,"atoms":[32,33]},{"atoms":[33,28]},{"atoms":[28,34]},{"atoms":[34,35]},'
                   '{"atoms":[35,36]},{"atoms":[36,37]},{"atoms":[30,38]},{"atoms":[38,39]},{"atoms":[31,40]},'
                   '{"atoms":[32,41]},{"atoms":[41,42]},{"bo":2,"atoms":[43,44]},{"atoms":[44,45]},',
                   '{"atoms":[43,49]},{"bo":2,"atoms":[49,50],"stereo":"trans","stereoAtoms":[43,51]},'
                   '{"atoms":[50,51]},{"atoms":[51,52]},{"atoms":[45,53]},{"atoms":[53,54]},{"atoms":[46,55]},'
                   '{"bo":2,"atoms":[56,57]},{"atoms":[57,58]},{"bo":2,"atoms":[58,59]},{"atoms":[59,60]},'
                   '{"bo":2,"atoms":[60,61]},{"atoms":[61,56]},{"atoms":[56,62]},{"atoms":[62,63]}',
                   '{"atoms":[64,65]},{"atoms":[58,66]},{"atoms":[66,67]},{"atoms":[59,68]},{"atoms":[60,69]},'
                   '{"atoms":[69,70]},{"bo":2,"atoms":[71,72]},{"atoms":[72,73]},{"bo":2,"atoms":[73,74]},'
                   '{"atoms":[74,75]},{"bo":2,"atoms":[75,76]},{"atoms":[76,71]},{"atoms":[71,77]},{"atoms":[77,78]},'
                   '{"atoms":[78,79]},{"atoms":[79,80]},{"atoms":[73,81]},{"atoms":[81,82]},{"atoms":[74,83]},'
                   '{"atoms":[75,84]},{"atoms":[84,85]},{"bo":2,"atoms":[86,87]},{"atoms":[87,88]},'
                   '{"bo":2,"atoms":[88,89]},{"atoms":[89,90]},{"bo":2,"atoms":[90,91]},{"atoms":[91,86]},'
                   '{"atoms":[86,92]},{"atoms":[92,93]},{"atoms":[93,94]},{"atoms":[94,95]},{"atoms":[88,96]},',
                   '{"atoms":[104,105]},{"bo":2,"atoms":[105,106]},{"atoms":[106,101]},{"atoms":[101,107]},'
                   '{"atoms":[107,108]},{"atoms":[108,109]},{"atoms":[109,110]},{"atoms":[103,111]},'
                   '{"atoms":[111,112]},{"atoms":[104,113]},{"atoms":[105,114]},{"atoms":[114,115]},'
                   '{"bo":2,"atoms":[116,117]},{"atoms":[117,118]},{"bo":2,"atoms":[118,119]},'
                   '{"atoms":[119,120]},{"bo":2,"atoms":[120,121]},{"atoms":[121,116]},{"atoms":[116,122]},'
                   '{"atoms":[122,123]},{"atoms":[123,124]},{"atoms":[124,125]},{"atoms":[118,126]},'
                   '{"atoms":[126,127]},{"atoms":[119,128]},{"bo":2,"atoms":[129,130]},{"atoms":[130,131]}',
                   '"atomRings":[[0,5,4,3,2,1],[12,3,4,20,19],[14,15,16,17,18,13],[29,30,31,32,33,28],'
                   '[72,73,74,75,76,71],[79,78,93,92,80],[87,88,89,90,91,86],[94,93,78,77,95],'
                   '[117,118,119,120,121,116],[130,131,132,133,134,129],[43,48,47,46,45,44],[55,46,47,63,62],'
                   '[57,58,59,60,61,56],[102,103,104,105,106,101]]']


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
            self.assertTrue('must be a positive number' in output)

    def testNegSGRatio(self):
        test_input = ["-r", "10", "-sg", "-0.1"]
        with capture_stderr(main, test_input) as output:
            self.assertTrue('must be a positive number' in output)

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


class TestCreateLigninNormalUse(unittest.TestCase):
    def testDefArgs(self):
        test_input = ["-r", "10"]
        # main(test_input)
        good_chain_summary = "Lignin KMC created 10 monomers, which formed:\n" \
                             "       1 trimer(s) (chain length 3)\n       " \
                             "1 oligomer(s) of chain length 7, with branching coefficient 0.143"
        good_bond_summary = "composed of the following bond types and number:\n " \
                            "   BO4:    4     BB:    1     B5:    2     B1:    0    5O4:    1    AO4:    0     55:    0"
        good_rcf_chain_summary = "Breaking C-O bonds to simulate RCF results in:\n" \
                                 "       4 monomer(s) (chain length 1)\n" \
                                 "       3 dimer(s) (chain length 2)"
        good_rcf_bond_summary = "with the following remaining bond types and number:\n    BO4:    0     " \
                                "BB:    1     B5:    2     B1:    0    5O4:    0    AO4:    0     55:    0"
        good_smiles = "COc1cc(C(O)C(CO)Oc2c(OC)cc(C(O)C(CO)Oc3c(OC)cc(C4OCC5C(c6cc(OC)c([O])c(OC)c6)OCC45)cc3OC)cc2O" \
                      "c2c(OC)cc(C(O)C(CO)Oc3c(OC)cc(C4Oc5c(OC)cc(/C=C/CO)cc5C4CO)cc3OC)cc2OC)cc(OC)c1[O].COc1" \
                      "cc(C(O)C(CO)Oc2c(OC)cc(C3Oc4c(OC)cc(/C=C/CO)cc4C3CO)cc2OC)cc(OC)c1[O]"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(OPENING_MSG in output)
            self.assertTrue(good_chain_summary in output)
            self.assertTrue(good_bond_summary in output)
            self.assertTrue(good_rcf_chain_summary in output)
            self.assertTrue(good_rcf_bond_summary in output)
            self.assertTrue(good_smiles in output)

    def testSaveSmi(self):
        try:
            test_input = ["-r", "10", "-o", TEST_SMI_BASENAME, "-d", SUB_DATA_DIR]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)

    def testDirInBasename(self):
        # This should ignore the temp_dir; will throw error if it doesn't
        try:
            test_input = ["-r", "10", "-o", TEST_SMI_OUT, "-d", TEMP_DIR]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT, disable=DISABLE_REMOVE)

    def testMakeSubDir(self):
        try:
            test_input = ["-r", "10", "-d", TEMP_DIR, "-o", TEST_SMI_BASENAME]
            main(test_input)
            self.assertFalse(diff_lines(TEST_SMI_OUT_TEMP_DIR, GOOD_TEST_SMI_OUT))
        finally:
            silent_remove(TEST_SMI_OUT_TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(TEMP_DIR, disable=DISABLE_REMOVE)
            silent_remove(INNER_TEMP_DIR, disable=DISABLE_REMOVE)

    def testSmallConfig(self):
        test_input = ["-c", SMALL_INI, "-r", "11"]
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
            test_input = ["-r", "10", "-f", "json", "-o", DEF_JSON_OUT]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            for json_part in good_json_parts:
                self.assertTrue(json_part in json_str[0])
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)

    def testSaveJSON(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json", "-d", SUB_DATA_DIR]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            self.assertTrue(good_json_parts[0] in json_str[0])
        finally:
            silent_remove(DEF_JSON_OUT, disable=DISABLE_REMOVE)

    def testSaveJSONTcl(self):
        try:
            silent_remove(DEF_JSON_OUT)
            test_input = ["-r", "10", "-f", "json tcl", "-d", SUB_DATA_DIR]
            main(test_input)
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            self.assertTrue(good_json_parts[0] in json_str[0])
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
            test_input = ["-r", "10", "-f", "json, png", "-d", SUB_DATA_DIR]
            main(test_input)
            self.assertTrue(os.path.isfile(DEF_PNG_OUT))
            with open(DEF_JSON_OUT, "r") as f:
                json_str = f.readlines()
            self.assertTrue(good_json_parts[0] in json_str[0])
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
        test_input = ["-r", "8", "-sg", "2.5"]
        # main(test_input)
        good_smiles = "COc1cc(C(O)C(CO)Oc2c(OC)cc(C3OCC4C(c5cc(OC)c([O])c(OC)c5)OCC34)cc2OC)cc(OC)c1[O].COc1cc" \
                      "(C2OCC3C(c4cc(OC)c([O])c(OC)c4)OCC23)cc(OC)c1[O].COc1cc(C2Oc3c(OC)cc(/C=C/CO)cc3C2CO)ccc1OC" \
                      "(CO)C(O)c1cc(OC)c2c(c1)C(CO)C(c1cc(OC)c(OC(CO)C(O)c3cc(OC)c([O])c(OC)c3)c(OC)c1)O2"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

    def testAltIniMaxMonosSimLen(self):
        test_input = ["-r", "10", "-i", "8", "-m", "12", "-l", "0.0002"]
        # main(test_input)
        good_smiles = "COc1cc(/C=C/CO)ccc1O.COc1cc(C2OCC3C(c4cc(OC)c(OC(CO)C(O)c5cc(OC)c([O])c(OC)c5)c(Oc5c(OC)cc" \
                      "(C6OCC7C(c8cc(OC)c(Oc9cc(C%10Oc%11c(OC)cc(/C=C/CO)cc%11C%10CO)cc(OC)c9[O])c(OC)c8)OCC67)" \
                      "cc5OC)c4)OCC23)ccc1OC(CO)C(O)c1cc(OC)c([O])c(OC)c1"
        with capture_stdout(main, test_input) as output:
            self.assertTrue(good_smiles in output)

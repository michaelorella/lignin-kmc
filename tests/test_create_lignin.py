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
GOOD_TCL_OPTIONS_OUT = os.path.join(SUB_DATA_DIR, "lignin-kmc-out_options_good.tcl")
SMALL_INI = os.path.join(SUB_DATA_DIR, "small_config.ini")
TEST_SMI_BASENAME = "test_lignin.smi"
TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, TEST_SMI_BASENAME)
GOOD_TEST_SMI_OUT = os.path.join(SUB_DATA_DIR, "good_test_lignin.smi")
TEST_SMI_OUT_TEMP_DIR = os.path.join(TEMP_DIR, TEST_SMI_BASENAME)


# Data #

# testing pieces of json, skipping parts that refer to version; more thorough testing is beyond scope

# GOOD_JSON_PARTS = ['"bonds":[{"bo":2,"atoms":[0,1]},{"atoms":[1,2]},{"bo":2,"atoms":[2,3]},{"atoms":[3,4]},'
#                    '{"bo":2,"atoms":[4,5]},{"atoms":[5,0]},{"atoms":[0,6]},{"bo":2,"atoms":[6,7],'
#                    '"stereo":"trans","stereoAtoms":[0,8]},{"atoms":[7,8]},{"atoms":[8,9]},{"atoms":[2,10]},{"atoms":[10,11]},{"atoms":[3,12]},{"bo":2,"atoms":[13,14]},{"atoms":[14,15]},{"bo":2,"atoms":[15,16]},{"atoms":[16,17]},{"bo":2,"atoms":[17,18]},{"atoms":[18,13]},{"atoms":[13,19]},{"atoms":[19,20]},{"atoms":[20,21]},{"atoms":[21,22]},{"atoms":[15,23]},{"atoms":[23,24]},{"atoms":[16,25]},{"atoms":[17,26]},{"atoms":[26,27]},{"bo":2,"atoms":[28,29]},{"atoms":[29,30]},{"bo":2,"atoms":[30,31]},{"atoms":[31,32]},{"bo":2,"atoms":[32,33]},{"atoms":[33,28]},{"atoms":[28,34]},{"atoms":[34,35]},{"atoms":[35,36]},{"atoms":[36,37]},{"atoms":[30,38]},{"atoms":[38,39]},{"atoms":[31,40]},{"atoms":[32,41]},{"atoms":[41,42]},{"bo":2,"atoms":[43,44]},{"atoms":[44,45]},{"bo":2,"atoms":[45,46]},{"atoms":[46,47]},{"bo":2,"atoms":[47,48]},{"atoms":[48,43]},{"atoms":[43,49]},{"atoms":[49,50]},{"atoms":[50,51]},{"atoms":[51,52]},{"atoms":[45,53]},{"atoms":[53,54]},{"atoms":[46,55]},{"bo":2,"atoms":[56,57]},{"atoms":[57,58]},{"bo":2,"atoms":[58,59]},{"atoms":[59,60]},{"bo":2,"atoms":[60,61]},{"atoms":[61,56]},{"atoms":[56,62]},{"atoms":[62,63]},{"atoms":[63,64]},{"atoms":[64,65]},{"atoms":[58,66]},{"atoms":[66,67]},{"atoms":[59,68]},{"atoms":[60,69]},{"atoms":[69,70]},{"bo":2,"atoms":[71,72]},{"atoms":[72,73]},{"bo":2,"atoms":[73,74]},{"atoms":[74,75]},{"bo":2,"atoms":[75,76]},{"atoms":[76,71]},{"atoms":[71,77]},{"atoms":[77,78]},{"atoms":[78,79]},{"atoms":[79,80]},{"atoms":[73,81]},{"atoms":[81,82]},{"atoms":[74,83]},{"bo":2,"atoms":[84,85]},{"atoms":[85,86]},{"bo":2,"atoms":[86,87]},{"atoms":[87,88]},{"bo":2,"atoms":[88,89]},{"atoms":[89,84]},{"atoms":[84,90]},{"atoms":[90,91]},{"atoms":[91,92]},{"atoms":[92,93]},{"atoms":[86,94]},{"atoms":[94,95]},{"atoms":[87,96]},{"bo":2,"atoms":[97,98]},{"atoms":[98,99]},{"bo":2,"atoms":[99,100]},{"atoms":[100,101]},{"bo":2,"atoms":[101,102]},{"atoms":[102,97]},{"atoms":[97,103]},{"atoms":[103,104]},{"atoms":[104,105]},{"atoms":[105,106]},{"atoms":[99,107]},{"atoms":[107,108]},{"atoms":[100,109]},{"atoms":[101,110]},{"atoms":[110,111]},{"bo":2,"atoms":[112,113]},{"atoms":[113,114]},{"bo":2,"atoms":[114,115]},{"atoms":[115,116]},{"bo":2,"atoms":[116,117]},{"atoms":[117,112]},{"atoms":[112,118]},{"atoms":[118,119]},{"atoms":[119,120]},{"atoms":[120,121]},{"atoms":[114,122]},{"atoms":[122,123]},{"atoms":[115,124]},{"atoms":[116,125]},{"atoms":[125,126]},{"bo":2,"atoms":[127,128]},{"atoms":[128,129]},{"bo":2,"atoms":[129,130]},{"atoms":[130,131]},{"bo":2,"atoms":[131,132]},{"atoms":[132,127]},{"atoms":[127,133]},{"atoms":[133,134]},{"atoms":[134,135]},{"atoms":[135,136]},{"atoms":[129,137]},{"atoms":[137,138]},{"atoms":[130,139]},{"atoms":[131,140]},{"atoms":[140,141]},{"atoms":[4,20]},{"atoms":[19,12]},{"atoms":[25,35]},{"atoms":[34,142]},{"atoms":[40,50]},{"atoms":[49,143]},{"atoms":[63,55]},{"atoms":[62,144]},{"atoms":[68,78]},{"atoms":[77,145]},{"atoms":[75,91]},{"atoms":[90,83]},{"atoms":[88,104]},{"atoms":[103,96]},{"atoms":[119,109]},{"atoms":[118,146]},{"atoms":[134,124]},{"atoms":[133,147]}],"conformers":[{"dim":2,"coords":[[-14.7742,14.9415],[-13.2795,14.8159],[-12.6409,13.4586],[-13.497,12.2269],[-14.9917,12.3525],[-15.6303,13.7098],[-15.4128,16.2988],[-14.5567,17.5304],[-15.1953,18.8877],[-14.3391,20.1194],[-11.1461,13.333],[-10.29,14.5647],[-13.1545,10.7666],[-14.5632,8.4948],[-15.9205,7.8562],[-16.0461,6.3615],[-14.8144,5.5054],[-13.4571,6.144],[-13.3315,7.6387],[-14.4376,9.9896],[-15.5731,10.9698],[-17.0335,10.6273],[-17.4671,9.1914],[-17.4034,5.7229],[-17.529,4.2282],[-14.94,4.0106],[-12.2255,5.2878],[-10.8682,5.9264],[-12.6022,0.8036],[-11.245,1.4422],[-10.0133,0.5861],[-10.1389,-0.9085],[-11.4961,-1.5471],[-12.7278,-0.691],[-13.8339,1.6598],[-13.7083,3.1545],[-12.351,3.7931],[-11.1194,2.937],[-8.656,1.2247],[-7.4243,0.3686],[-8.9072,-1.7647],[-11.6217,-3.0419],[-12.979,-3.6805],[-7.9267,-5.6103],[-9.284,-6.2489],[-9.4096,-7.7436],[-8.1779,-8.5997],[-6.8206,-7.9611],[-6.695,-6.4664],[-7.8011,-4.1155],[-9.0328,-3.2594],[-10.3901,-3.898],[-10.5157,-5.3927],[-10.7668,-8.3822],[-10.8924,-9.8769],[-8.3035,-10.0944],[-4.4828,-11.1681],[-4.6084,-12.6628],[-3.3768,-13.519],[-2.0195,-12.8804],[-1.8939,-11.3856],[-3.1256,-10.5295],[-5.7145,-10.312],[-7.0718,-10.9506],[-7.1974,-12.4453],[-5.9657,-13.3014],[-3.5023,-15.0137],[-2.2707,-15.8698],[-0.7878,-13.7365],[-0.5366,-10.747],[-0.411,-9.2523],[3.1583,-13.3154],[4.39,-14.1716],[5.7473,-13.533],[5.8729,-12.0382],[4.6412,-11.1821],[3.2839,-11.8207],[1.8011,-13.954],[0.5694,-13.0979],[0.695,-11.6032],[2.0522,-10.9646],[6.979,-14.3891],[6.8534,-15.8838],[7.0677,-11.1314],[7.4306,-8.4831],[8.9253,-8.6087],[9.7815,-7.3771],[9.1429,-6.0198],[7.6481,-5.8942],[6.792,-7.1259],[6.5745,-9.7148],[5.0748,-9.7462],[4.168,-8.5513],[2.6798,-8.7392],[11.2762,-7.5026],[12.1323,-6.271],[9.7242,-4.637],[8.7144,-2.1621],[7.4827,-1.306],[7.6083,0.1887],[8.9656,0.8273],[10.1972,-0.0288],[10.0716,-1.5235],[8.5888,-3.6568],[7.3057,-4.4338],[5.9229,-3.8525],[4.7281,-4.7593],[6.3766,1.0448],[5.0193,0.4062],[9.0912,2.322],[11.5545,0.6097],[12.7862,-0.2463],[11.9313,5.0939],[13.163,4.2378],[14.5202,4.8764],[14.6458,6.3711],[13.4142,7.2273],[12.0569,6.5887],[10.574,4.4553],[10.4484,2.9606],[11.6801,2.1045],[13.0374,2.7431],[15.7519,4.0203],[17.1092,4.6589],[16.0031,7.0097],[13.5398,8.722],[12.3081,9.5781],[17.6116,10.6378],[16.3799,11.4939],[16.5055,12.9887],[17.8628,13.6273],[19.0944,12.7711],[18.9688,11.2764],[17.486,9.1431],[16.1287,8.5045],[14.897,9.3606],[15.0226,10.8553],[15.2738,13.8448],[15.3994,15.3395],[17.9883,15.122],[20.4517,13.4097],[21.6834,12.5536],[-15.1912,1.0212],[-6.4438,-3.4769],[-5.5889,-8.8172],[1.6755,-15.4488],[9.3423,5.3115],[18.7177,8.2869]]}],"extensions":[{"name":"rdkitRepresentation","formatVersion":1,"toolkitVersion":"2018.03.4","aromaticAtoms":[0,1,2,3,4,5,13,14,15,16,17,18,28,29,30,31,32,33,43,44,45,46,47,48,56,57,58,59,60,61,71,72,73,74,75,76,84,85,86,87,88,89,97,98,99,100,101,102,112,113,114,115,116,117,127,128,129,130,131,132],"aromaticBonds":[0,1,2,3,4,5,13,14,15,16,17,18,28,29,30,31,32,33,43,44,45,46,47,48,56,57,58,59,60,61,71,72,73,74,75,76,84,85,86,87,88,89,97,98,99,100,101,102,112,113,114,115,116,117,127,128,129,130,131,132],"cipRanks":[20,10,71,80,30,2,0,5,43,89,107,35,117,28,18,77,86,77,18,60,7,45,91,113,41,122,113,41,25,16,74,83,74,16,57,66,51,97,110,38,119,110,38,21,12,70,79,9,1,53,63,48,94,106,34,115,24,15,75,84,75,15,55,62,47,93,111,39,120,111,39,22,13,72,81,31,3,54,64,49,95,108,36,116,27,14,73,82,32,4,59,6,44,90,109,37,118,29,19,78,87,78,19,61,8,46,92,114,42,123,114,42,26,17,76,85,76,17,58,67,52,98,112,40,121,112,40,23,11,69,68,69,11,56,65,50,96,105,33,88,105,33,103,99,101,100,104,102],"atomRings":[[0,5,4,3,2,1],[12,3,4,20,19],[14,15,16,17,18,13],[29,30,31,32,33,28],[44,45,46,47,48,43],[57,58,59,60,61,56],[72,73,74,75,76,71],[83,74,75,91,90],[85,86,87,88,89,84],[96,87,88,104,103],[98,99,100,101,102,97],[113,114,115,116,117,112],[128,129,130,131,132,127]]}]}]}']
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

    def testAlphaAddRate(self):
        test_input = ["-a", "ghost"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("A positive number" in output)

    def testZeroAddRate(self):
        test_input = ["-a", "0"]
        # main(test_input)
        with capture_stderr(main, test_input) as output:
            self.assertTrue("A positive number" in output)


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

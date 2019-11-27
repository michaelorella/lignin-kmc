#!/usr/bin/env python3

import logging
import os
import unittest
from ligninkmc.plot_bond_formation import main
from common_wrangler.common import capture_stderr, capture_stdout, silent_remove


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
DEF_MONO_PNG = os.path.join(MAIN_DIR, "mono_olig_v_step_1_1.png")

BOND_OPT_1_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1e08.png")
BOND_OPT_2_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1e04.png")
MONO_OPT_1_PNG = os.path.join(SUB_DATA_DIR, "mono_olig_v_step_0-25_1e08.png")
MONO_OPT_2_PNG = os.path.join(SUB_DATA_DIR, "mono_olig_v_step_3_1e08.png")
MONO_OPT_3_PNG = os.path.join(SUB_DATA_DIR, "mono_olig_v_step_0-25_1e04.png")
MONO_OPT_4_PNG = os.path.join(SUB_DATA_DIR, "mono_olig_v_step_3_1e04.png")

ORELLA_BOND_PNG = os.path.join(SUB_DATA_DIR, "mono_olig_v_step_1_1.png")
ORELLA_MONO_PNG = os.path.join(SUB_DATA_DIR, "bond_v_add_rate_1.png")



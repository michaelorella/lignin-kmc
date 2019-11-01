#!/usr/bin/env python3

import logging
import os
import unittest
from common_wrangler.common import (calc_k)


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


# Tests #

class TestRateCalc(unittest.TestCase):
    """
    Tests calculation of a rate coefficient by the Eyring equation.
    """
    def test_calc_k(self):
        temp = 900.0
        delta_g = 53.7306
        rate_coeff = calc_k(temp, delta_g)
        self.assertEqual(rate_coeff, 1.648326791137026)

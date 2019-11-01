#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launches steps required to build lignin
"""
import argparse
import sys
import numpy as np
from configparser import ConfigParser
from common_wrangler.common import (warning, process_cfg, MAIN_SEC, GOOD_RET, INPUT_ERROR, IO_ERROR, KB, H,
                                    KCAL_MOL_TO_J_PART)
from ligninkmc.kmc_common import (E_A_KCAL_MOL, E_A_J_PART,
                                  TEMP, AO4, B1, B5, BB, BO4, C5O4, C5C5, OX, Q, MONOMER, DIMER)

# Defaults
DEF_TEMP = 298.15  # K

# Default activation energies input in kcal/mol from Gani et al., ACS Sustainable Chem. Eng. 2019, 7, 15, 13270-13277,
#     https://doi.org/10.1021/acssuschemeng.9b02506,
#     as described in Orella et al., ACS Sustainable Chem. Eng. 2019, https://doi.org/10.1021/acssuschemeng.9b03534 
DEF_E_A_KCAL_MOL = {C5O4: {(0, 0): {(MONOMER, MONOMER): 11.2, (MONOMER, DIMER): 14.6, (DIMER, MONOMER): 14.6,
                                    (DIMER, DIMER): 4.4},
                           (1, 0): {(MONOMER, MONOMER): 10.9, (MONOMER, DIMER): 14.6, (DIMER, MONOMER): 14.6,
                                    (DIMER, DIMER): 4.4}},
                    C5C5: {(0, 0): {(MONOMER, MONOMER): 12.5, (MONOMER, DIMER): 15.6, (DIMER, MONOMER): 15.6,
                                    (DIMER, DIMER): 3.8}},
                    B5: {(0, 0): {(MONOMER, MONOMER): 5.5, (MONOMER, DIMER): 5.8, (DIMER, MONOMER): 5.8,
                                  (DIMER, DIMER): 5.8},
                         (0, 1): {(MONOMER, MONOMER): 5.5, (MONOMER, DIMER): 5.8, (DIMER, MONOMER): 5.8,
                                  (DIMER, DIMER): 5.8}},
                    BB: {(0, 0): {(MONOMER, MONOMER): 5.2, (MONOMER, DIMER): 5.2, (DIMER, MONOMER): 5.2,
                                  (DIMER, DIMER): 5.2},
                         (1, 0): {(MONOMER, MONOMER): 6.5, (MONOMER, DIMER): 6.5, (DIMER, MONOMER): 6.5,
                                  (DIMER, DIMER): 6.5},
                         (1, 1): {(MONOMER, MONOMER): 5.2, (MONOMER, DIMER): 5.2, (DIMER, MONOMER): 5.2,
                                  (DIMER, DIMER): 5.2}},
                    BO4: {(0, 0): {(MONOMER, MONOMER): 6.3, (MONOMER, DIMER): 6.2, (DIMER, MONOMER): 6.2,
                                   (DIMER, DIMER): 6.2},
                          (1, 0): {(MONOMER, MONOMER): 9.1, (MONOMER, DIMER): 6.2,
                                   (DIMER, MONOMER): 6.2, (DIMER, DIMER): 6.2},
                          (0, 1): {(MONOMER, MONOMER): 8.9, (MONOMER, DIMER): 6.2,
                                   (DIMER, MONOMER): 6.2, (DIMER, DIMER): 6.2},
                          (1, 1): {(MONOMER, MONOMER): 9.8, (MONOMER, DIMER): 10.4,
                                   (DIMER, MONOMER): 10.4}},
                    AO4: {(0, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (1, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (0, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (1, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7}},
                    B1: {(0, 0): {(MONOMER, DIMER): 9.6, (DIMER, MONOMER): 9.6, (DIMER, DIMER): 9.6},
                         (1, 0): {(MONOMER, DIMER): 11.7, (DIMER, MONOMER): 11.7, (DIMER, DIMER): 11.7},
                         (0, 1): {(MONOMER, DIMER): 10.7, (DIMER, MONOMER): 10.7, (DIMER, DIMER): 10.7},
                         (1, 1): {(MONOMER, DIMER): 11.9, (DIMER, MONOMER): 11.9, (DIMER, DIMER): 11.9}},
                    OX: {0: {MONOMER: 0.9, DIMER: 6.3}, 1: {MONOMER: 0.6, DIMER: 2.2}},
                    Q: {0: {MONOMER: 11.1, DIMER: 11.1}, 1: {MONOMER: 11.7, DIMER: 11.7}}}
DEF_E_A_KCAL_MOL[BB][(0, 1)] = DEF_E_A_KCAL_MOL[BB][(1, 0)]

DEF_CFG_VALS = {TEMP: DEF_TEMP, E_A_KCAL_MOL: DEF_E_A_KCAL_MOL, E_A_J_PART: None,
                }
REQ_KEYS = {}


def read_cfg(f_loc, cfg_proc=process_cfg):
    """
    Reads the given configuration file, returning a dict with the converted values supplemented by default values.

    :param f_loc: The location of the file to read.
    :param cfg_proc: The processor to use for the raw configuration values.  Uses default values when the raw
        value is missing.
    :return: A dict of the processed configuration file's data.
    """
    config = ConfigParser()
    good_files = config.read(f_loc)

    if not good_files:
        return DEF_CFG_VALS

    main_proc = cfg_proc(dict(config.items(MAIN_SEC)), DEF_CFG_VALS, REQ_KEYS)

    return main_proc


def parse_cmdline(argv=None):
    """
    Returns the parsed argument list and return code.
    :param argv: A list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description='Create lignin(s).')
    parser.add_argument("-c", "--config", help="The location of the configuration file in ini format. This file "
                                               "can be used to overwrite default values such as for energies.",
                        default=None, type=read_cfg)
    parser.add_argument("-t", "--temp", help="The temperature (in K) at which lignin biosynthesis should be modeled. "
                                             "The default is {} K.".format(DEF_TEMP), default=DEF_TEMP)

    args = None
    try:
        args = parser.parse_args(argv)
        # if temperature is set on the command line, overwrite from config file or default
        if args.temp != DEF_TEMP:
            # noinspection PyStatementEffect
            args.config[TEMP] == args.temp
    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def calc_rates(cfg):
    """
    Uses temperature and provided activation energy to calculate rates using the Eyring equation
    :param cfg: dict of configuration values, including TEMP, EA_KCAL_MOL, EA_J_MOL for the expected rxn_type types
    :return: rxn_rates: dict in units of 1/s
    """
    # copy config value(s) to convenient names
    temp = cfg[TEMP]

    # want activation energies in J/particle; user can provide them in those units or in kcal/mol
    if cfg[E_A_J_PART] is None:
        cfg[E_A_J_PART] = {
            rxn_type: {substrate: {sub_type: cfg[E_A_KCAL_MOL][rxn_type][substrate][sub_type] * KCAL_MOL_TO_J_PART
                                   for sub_type in cfg[E_A_KCAL_MOL][rxn_type][substrate]}
                       for substrate in cfg[E_A_KCAL_MOL][rxn_type]} for rxn_type in cfg[E_A_KCAL_MOL]}
    # TODO: check if need to change to solution state...
    # the nested loop below was only for debugging
    # for bond_type in cfg[E_A_KCAL_MOL].keys():
    #     for substrate in cfg[E_A_KCAL_MOL][bond_type]:
    #         kcal_mon_typ_dict = cfg[E_A_KCAL_MOL][bond_type][substrate]
    #         joul_mon_typ_dict = cfg[E_A_J_PART][bond_type][substrate]
    #         for linked_species in kcal_mon_typ_dict:
    #             print("bond_type: {:>3}    substrate: {}    linked_species: {:>22}    ea kcal/mol: {:5.2f}    "
    #                   "ea j/particle: {:.2e}".format(bond_type, substrate, str(linked_species),
    #                                                  kcal_mon_typ_dict[linked_species],
    #                                                  joul_mon_typ_dict[linked_species]))
    rxn_rates = {}
    for rxn_type in cfg[E_A_J_PART]:
        rxn_rates[rxn_type] = {}
        for substrate in cfg[E_A_J_PART][rxn_type]:
            rxn_rates[rxn_type][substrate] = {}
            for substrate_type in cfg[E_A_J_PART][rxn_type][substrate]:
                rate = KB * temp / H * np.exp(-cfg[E_A_J_PART][rxn_type][substrate][substrate_type] / KB / temp)
                rxn_rates[rxn_type][substrate][substrate_type] = rate
    return rxn_rates


def main(argv=None):
    """
    Runs the main program.

    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    cfg = args.config
    print(cfg)

    try:
        rxn_rates = calc_rates(cfg)
        print(rxn_rates)

    except IOError as e:
        warning("Problems reading file: {}".format(e))
        return IO_ERROR
    except KeyError as e:
        warning(e)
        return IO_ERROR

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

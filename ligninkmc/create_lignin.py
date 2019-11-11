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
                                    KCAL_MOL_TO_J_PART, InvalidDataError, INVALID_DATA)

from ligninkmc.analysis import adj_analysis_to_stdout, analyze_adj_matrix
from ligninkmc import Event
from ligninkmc import Monomer
from ligninkmc.kmc_functions import run_kmc
from ligninkmc.kmc_common import (E_A_KCAL_MOL, E_A_J_PART, TEMP, INI_MONOS, MAX_MONOS, SIM_TIME, AFFECTED, GROW,
                                  DEF_E_A_KCAL_MOL, OX, MONOMER, DIMER, LIGNIN_SUBUNITS, SG_RATIO,
                                  ADJ_MATRIX, RANDOM_SEED)

# Defaults #

DEF_TEMP = 298.15  # K
DEF_MAX_MONOS = 10  # number of monomers
DEF_SIM_TIME = 1  # simulation time in seconds
DEF_SG = 1
DEF_INI_MONOS = 2
DEF_INI_RATE = 1e4
DEF_RANDOM_SEED = None

DEF_VAL = 'default_value'
CONFIG_KEY = 'config_key'
DEF_CFG_VALS = {TEMP: DEF_TEMP, E_A_KCAL_MOL: DEF_E_A_KCAL_MOL, E_A_J_PART: None, SG_RATIO: DEF_SG,
                MAX_MONOS: DEF_MAX_MONOS, SIM_TIME: DEF_SIM_TIME, INI_MONOS: DEF_INI_MONOS,
                RANDOM_SEED: DEF_RANDOM_SEED,
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
        raise IOError(f"Could not find specified configuration file: {f_loc}")

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
    parser = argparse.ArgumentParser(description='Create lignin chain(s) as described in:\n  Orella, M., '
                                                 'Gani, T. Z. H., Vermaas, J. V., Stone, M. L., Anderson, E. M., '
                                                 'Beckham, G. T., \n  Brushett, Fikile R., Roman-Leshkov, Y. (2019). '
                                                 'Lignin-KMC: A Toolkit for Simulating Lignin Biosynthesis.\n  '
                                                 'ACS Sustainable Chemistry & Engineering. '
                                                 'https://doi.org/10.1021/acssuschemeng.9b03534.\n\n  '
                                                 'By default, the activation energies from this reference will be '
                                                 'used, as specified in Tables S1 and S2.\n  Alternately, the user '
                                                 f"may specify values, which should be specified as a dict of dict "
                                                 f"of dicts in a \n  specified configuration file (specified with '-c')"
                                                 f" using the '{E_A_KCAL_MOL}' or '{E_A_J_PART}'\n  parameters with "
                                                 f"corresponding units (kcal/mol or joules/particle, respectively).\n"
                                                 f"  The format is (bond_type: monomer(s) involved: units involved: "
                                                 f"ea_vals), for example:\n      "
                                                 f"ea_dict = {{{OX}: {{0: {{{MONOMER}: 0.9, {DIMER}: 6.3}}, "
                                                 f"1: ""{{{MONOMER}: 0.6, {DIMER}: " f"2.2}}}}, ...}}\n  "
                                                 f"where 0: {LIGNIN_SUBUNITS[0]}, 1: {LIGNIN_SUBUNITS[1]}, "
                                                 f"2: {LIGNIN_SUBUNITS[2]}.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", help="The location of the configuration file in ini format. This file "
                                               "can be used to overwrite default \nvalues such as for energies.",
                        default=None, type=read_cfg)
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_INI_MONOS)
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME} s.", default=DEF_SIM_TIME)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-sg", "--sg_ratio", help=f"The S:G (guaiacol:syringyl) ratio. "
                                                  f"The default is {DEF_SG}.", default=DEF_SG)
    parser.add_argument("-t", "--temperature_in_k", help=f"The temperature (in K) at which to model lignin "
                                                         f"biosynthesis. The default is {DEF_TEMP} K.",
                        default=DEF_TEMP)
    parser.add_argument("-r", "--random_seed", help="Random seed value to be used for testing.", default=None)

    args = None
    try:
        args = parser.parse_args(argv)
        if args.config is None:
            args.config = {INI_MONOS: args.initial_num_monomers, SIM_TIME: args.length_simulation,
                           MAX_MONOS: args.max_num_monomers, SG_RATIO: args.sg_ratio,
                           TEMP: args.temperature_in_k, RANDOM_SEED: args.random_seed,
                           E_A_KCAL_MOL: DEF_E_A_KCAL_MOL, E_A_J_PART: None}
        else:
            arg_def_dict = {args.initial_num_monomers: {DEF_VAL: DEF_INI_MONOS, CONFIG_KEY: INI_MONOS},
                            args.length_simulation: {DEF_VAL: DEF_SIM_TIME, CONFIG_KEY: SIM_TIME},
                            args.max_num_monomers: {DEF_VAL: DEF_MAX_MONOS, CONFIG_KEY: MAX_MONOS},
                            args.sg_ratio: {DEF_VAL: DEF_SG, CONFIG_KEY: SG_RATIO},
                            args.temperature_in_k: {DEF_VAL: DEF_TEMP, CONFIG_KEY: TEMP},
                            args.random_seed: {DEF_VAL: DEF_RANDOM_SEED, CONFIG_KEY: RANDOM_SEED}}
            for arg_val, arg_dict in arg_def_dict.items():
                config_key = arg_dict[CONFIG_KEY]
                def_val = arg_dict[DEF_VAL]
                if arg_val != def_val:
                    args.config[config_key] = arg_val

    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def calc_rates(temp, ea_j_part_dict=None, ea_kcal_mol_dict=None):
    """
    Uses temperature and provided activation energy to calculate rates using the Eyring equation
    dictionary formats: dict = {rxn_type: {substrate(s): {sub_lengths (e.g. (monomer, monomer)): value, ...}, ...}, ...}

    Only ea_j_part_dict or ea_kcal_mol_dict are needed; if both are provided, only ea_j_part_dict will be used

    :param temp: float, temperature in K
    :param ea_j_part_dict: dictionary of activation energies in Joule/particle, in format noted above
    :param ea_kcal_mol_dict: dictionary of activation energies in Joule/particle, in format noted above
    :return: rxn_rates: dict of reaction rates units of 1/s
    """
    # copy config value(s) to convenient names
    temp = temp

    # want activation energies in J/particle; user can provide them in those units or in kcal/mol
    if ea_j_part_dict is None:
        ea_j_part_dict = {
            rxn_type: {substrate: {sub_len: ea_kcal_mol_dict[rxn_type][substrate][sub_len] * KCAL_MOL_TO_J_PART
                                   for sub_len in ea_kcal_mol_dict[rxn_type][substrate]}
                       for substrate in ea_kcal_mol_dict[rxn_type]} for rxn_type in ea_kcal_mol_dict}
    # TODO: check if need to change to solution state...
    rxn_rates = {}
    for rxn_type in ea_j_part_dict:
        rxn_rates[rxn_type] = {}
        for substrate in ea_j_part_dict[rxn_type]:
            rxn_rates[rxn_type][substrate] = {}
            for substrate_type in ea_j_part_dict[rxn_type][substrate]:
                rate = KB * temp / H * np.exp(-ea_j_part_dict[rxn_type][substrate][substrate_type] / KB / temp)
                rxn_rates[rxn_type][substrate][substrate_type] = rate
    return rxn_rates


def create_initial_monomers(pct_s, monomer_draw):
    # TODO: If want more than 2 monomer options, need to change logic
    # if mon_choice < pct_s, make it an S; that is, the evaluation comes back True (=1='S');
    #     otherwise, get False = 0 = 'G'. Since only two options (True/False) only works for 2 monomers
    try:
        return [Monomer(int(mono_type_draw < pct_s), i) for i, mono_type_draw in enumerate(monomer_draw)]
    except TypeError as e:
        if "'<' not supported between instances of 'float' and 'NoneType'" in e.args[0]:
            raise InvalidDataError(f"A float is required for the sg_ratio; instead found: {pct_s}")
        else:
            raise InvalidDataError(e)


def create_initial_events(monomer_draw, pct_s, rxn_rates):
    return [Event(OX, [i], rxn_rates[OX][int(mono_type_draw < pct_s)][MONOMER])
            for i, mono_type_draw in enumerate(monomer_draw)]


def create_initial_state(initial_events, initial_monomers):
    return {i: {MONOMER: initial_monomers[i], AFFECTED: {initial_events[i]}} for i in range(len(initial_monomers))}


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

    try:
        # need rates before we can start modeling reactions
        rxn_rates = calc_rates(cfg[TEMP], ea_j_part_dict=cfg[E_A_J_PART], ea_kcal_mol_dict=cfg[E_A_KCAL_MOL])

        # decide on initial monomers, based on given SG_RATIO
        pct_s = cfg[SG_RATIO] / (1 + cfg[SG_RATIO])
        if cfg[RANDOM_SEED]:
            np.random.seed(int(cfg[RANDOM_SEED]))
        ini_num_monos = cfg[INI_MONOS]
        monomer_draw = np.random.rand(ini_num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)

        # initial event must be oxidation to create reactive species; all monomers a chance of being oxidized
        initial_events = create_initial_events(monomer_draw, pct_s, rxn_rates)

        # When the monomers and starting events have been initialized, they are grouped into the "state" and "events"
        # which are necessary to start the simulation. The final pieces of information needed to run_kmc the simulation
        # are the maximum number of monomers that should be studied and the final simulation time.
        initial_state = create_initial_state(initial_events, initial_monomers)
        initial_events.append(Event(GROW, [], rate=DEF_INI_RATE, bond=cfg[SG_RATIO]))

        # begin simulation
        result = run_kmc(rxn_rates, initial_state, initial_events, n_max=cfg[MAX_MONOS], t_max=cfg[SIM_TIME],
                         sg_ratio=cfg[SG_RATIO])
        # show results
        summary = analyze_adj_matrix(result[ADJ_MATRIX])
        adj_analysis_to_stdout(summary)
    except InvalidDataError as e:
        warning(e)
        return INVALID_DATA
    except KeyError as e:
        warning(e)
        return IO_ERROR

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)

# !/usr/bin/env python
# coding=utf-8

# lignin subunits (phenylpropanoids)
G = 'guaiacol'  # (coniferyl)
S = 'syringyl'  # (sinapyl)
C = 'caffeoyl'
H = 'p-hydroxyphenyl'
S4 = 'S4'
G4 = 'G4'
G7 = 'G7'
LIGNIN_SUBUNITS = {0: G, 1: S, 2: C}  # perhaps to be added later: 3: H
MONOLIG_OHS = {0: 'coniferyl', 1: 'sinapyl', 2: 'caffeoyl'}  # perhaps to be added later: 3: 'p-coumaryl'
SG_RATIO = 'sg_ratio'
INI_MONOS = 'initial_num_monomers'
MAX_MONOS = 'max_num_monomers'
SIM_TIME = 'length_sim'

# bond types
AO4 = 'ao4'
B1 = 'b1'
B1_ALT = 'b1alt'
B5 = 'b5'
BB = 'bb'
BO4 = 'bo4'
C5C5 = '55'
C5O4 = '5o4'

# reaction types
Q = 'hydration'
OX = 'oxidation'
GROW = 'grow'

DIMER = 'dimer'
MONOMER = 'monomer'
MON_MON = (MONOMER, MONOMER)
MON_DIM = (MONOMER, DIMER)
DIM_MON = (DIMER, MONOMER)
DIM_DIM = (DIMER, DIMER)

TIME = 'time'
TEMP = 'temperature_in_K'
E_A_KCAL_MOL = 'e_a_in_kcal_mol'
E_A_J_PART = 'e_a_in_j_particle'

AFFECTED = 'affected'
ADJ_MATRIX = 'adjacency_matrix'
MONO_LIST = 'monomers'

CHAIN_LEN = 'Chain Lengths'
ATOMS = 'Atoms'
BONDS = 'Bonds'
RCF_YIELDS = 'RCF Yields'
RCF_BONDS = 'RCF Bonds'


# Data

# Default activation energies input in kcal/mol from Gani et al., ACS Sustainable Chem. Eng. 2019, 7, 15, 13270-13277,
#     https://doi.org/10.1021/acssuschemeng.9b02506,
#     as described in Orella et al., ACS Sustainable Chem. Eng. 2019, https://doi.org/10.1021/acssuschemeng.9b03534
DEF_E_A_KCAL_MOL = {C5O4: {(0, 0): {(MONOMER, MONOMER): 11.2, (MONOMER, DIMER): 14.6, (DIMER, MONOMER): 14.6,
                                    (DIMER, DIMER): 4.4},
                           (1, 0): {(MONOMER, MONOMER): 10.9, (MONOMER, DIMER): 14.6, (DIMER, MONOMER): 14.6,
                                    (DIMER, DIMER): 4.4},
                           (2, 2): {(MONOMER, MONOMER): 11.9, (MONOMER, DIMER): 11.9,
                                    (DIMER, MONOMER): 11.9, (DIMER, DIMER): 11.9}},
                    C5C5: {(0, 0): {(MONOMER, MONOMER): 12.5, (MONOMER, DIMER): 15.6, (DIMER, MONOMER): 15.6,
                                    (DIMER, DIMER): 3.8},
                           (2, 2): {(MONOMER, MONOMER): 10.6, (MONOMER, DIMER): 10.6,
                                    (DIMER, MONOMER): 10.6, (DIMER, DIMER): 10.6}},
                    B5: {(0, 0): {(MONOMER, MONOMER): 5.5, (MONOMER, DIMER): 5.8, (DIMER, MONOMER): 5.8,
                                  (DIMER, DIMER): 5.8},
                         (0, 1): {(MONOMER, MONOMER): 5.5, (MONOMER, DIMER): 5.8, (DIMER, MONOMER): 5.8,
                                  (DIMER, DIMER): 5.8},
                         (2, 2): {(MONOMER, MONOMER): 1.9, (MONOMER, DIMER): 5.8,
                                  (DIMER, MONOMER): 5.8, (DIMER, DIMER): 5.8}},
                    BB: {(0, 0): {(MONOMER, MONOMER): 5.2, (MONOMER, DIMER): 5.2, (DIMER, MONOMER): 5.2,
                                  (DIMER, DIMER): 5.2},
                         (1, 0): {(MONOMER, MONOMER): 6.5, (MONOMER, DIMER): 6.5, (DIMER, MONOMER): 6.5,
                                  (DIMER, DIMER): 6.5},
                         (1, 1): {(MONOMER, MONOMER): 5.2, (MONOMER, DIMER): 5.2, (DIMER, MONOMER): 5.2,
                                  (DIMER, DIMER): 5.2},
                         (2, 2): {(MONOMER, MONOMER): 7.2, (MONOMER, DIMER): 7.2,
                                  (DIMER, MONOMER): 7.2, (DIMER, DIMER): 7.2}},
                    BO4: {(0, 0): {(MONOMER, MONOMER): 6.3, (MONOMER, DIMER): 6.2, (DIMER, MONOMER): 6.2,
                                   (DIMER, DIMER): 6.2},
                          (1, 0): {(MONOMER, MONOMER): 9.1, (MONOMER, DIMER): 6.2,
                                   (DIMER, MONOMER): 6.2, (DIMER, DIMER): 6.2},
                          (0, 1): {(MONOMER, MONOMER): 8.9, (MONOMER, DIMER): 6.2,
                                   (DIMER, MONOMER): 6.2, (DIMER, DIMER): 6.2},
                          (1, 1): {(MONOMER, MONOMER): 9.8, (MONOMER, DIMER): 10.4,
                                   (DIMER, MONOMER): 10.4},
                          (2, 2): {(MONOMER, MONOMER): 4.9, (MONOMER, DIMER): 1.3,
                                   (DIMER, MONOMER): 1.3, (DIMER, DIMER): 1.3}},
                    AO4: {(0, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (1, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (0, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (1, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7},
                          (2, 2): {(MONOMER, MONOMER): 20.7, (MONOMER, DIMER): 20.7,
                                   (DIMER, MONOMER): 20.7, (DIMER, DIMER): 20.7}},
                    B1: {(0, 0): {(MONOMER, DIMER): 9.6, (DIMER, MONOMER): 9.6, (DIMER, DIMER): 9.6},
                         (1, 0): {(MONOMER, DIMER): 11.7, (DIMER, MONOMER): 11.7, (DIMER, DIMER): 11.7},
                         (0, 1): {(MONOMER, DIMER): 10.7, (DIMER, MONOMER): 10.7, (DIMER, DIMER): 10.7},
                         (1, 1): {(MONOMER, DIMER): 11.9, (DIMER, MONOMER): 11.9, (DIMER, DIMER): 11.9},
                         (2, 2): {(MONOMER, DIMER): 9.6, (DIMER, MONOMER): 9.6, (DIMER, DIMER): 9.6}},
                    OX: {0: {MONOMER: 0.9, DIMER: 6.3}, 1: {MONOMER: 0.6, DIMER: 2.2}, 2: {MONOMER: 0.9, DIMER: 0.9}},
                    Q: {0: {MONOMER: 11.1, DIMER: 11.1}, 1: {MONOMER: 11.7, DIMER: 11.7},
                        2: {MONOMER: 11.1, DIMER: 11.1}}}
DEF_E_A_KCAL_MOL[BB][(0, 1)] = DEF_E_A_KCAL_MOL[BB][(1, 0)]

cLigEnergies = {
    'ao4': {},
    'b1': {},
    'ox': {},
    'q': {}}

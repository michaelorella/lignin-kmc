# !/usr/bin/env python
# coding=utf-8

from common_wrangler.common import InvalidDataError

# lignin subunits (phenylpropanoids)
G = 'guaiacol'  # (coniferyl alcohol)
S = 'syringyl'  # (sinapyl alcohol)
C = 'caffeoyl'  # (caffeyl alcohol)
H = 'p-hydroxyphenyl'
S4 = 'S4'
G4 = 'G4'
G7 = 'G7'
S7 = 'S7'
LIGNIN_SUBUNITS = [G, S, H, C]
# Dict below likely to be changed when H added
INT_TO_TYPE_DICT = {0: G, 1: S}
MONOLIG_OHS = {G: 'coniferyl', S: 'sinapyl', H: 'p-coumaryl', C: 'caffeoyl'}
ADD_RATE = 'add_rate'
INI_MONOS = 'initial_num_monomers'
MAX_MONOS = 'max_num_monomers'
SIM_TIME = 'length_simulation'
RANDOM_SEED = 'random_seed'
CHAIN_ID = 'chain_id'
PSF_FNAME = 'psf_fname'
TOPPAR_DIR = "toppar_dir"
# specifying a random seed is not enough for reliable testing, as a generated float can differ by machine precision
# Thus, MAX_NUM_DECIMAL is used to round floats to lower-than machine precision
MAX_NUM_DECIMAL = 8

# bond types
AO4 = 'ao4'
B1 = 'b1'
B1_ALT = 'b1alt'
B5 = 'b5'
BB = 'bb'
BO4 = 'bo4'
C5C5 = '55'
C5O4 = '5o4'
BOND_TYPE_LIST = [BO4, BB, B5, B1, C5O4, AO4, C5C5]

# reaction types other than bond formation
Q = 'hydration'
OX = 'oxidation'
GROW = 'grow'

OLIGOMER = 'oligomer'
MONOMER = 'monomer'
MON_MON = (MONOMER, MONOMER)
MON_OLI = (MONOMER, OLIGOMER)
OLI_MON = (OLIGOMER, MONOMER)
OLI_OLI = (OLIGOMER, OLIGOMER)

TIME = 'time'
TEMP = 'temperature_in_k'
E_BARRIER_KCAL_MOL = 'e_barrier_in_kcal_mol'
E_BARRIER_J_PART = 'e_barrier_in_j_particle'

AFFECTED = 'affected'
ADJ_MATRIX = 'adjacency_matrix'
MONO_LIST = 'monomers'

ATOMS = 'Atoms'
BONDS = 'Bonds'
CHAIN_LEN = 'Chain Lengths'
CHAIN_MONOS = 'Chain Monomers'
CHAIN_BRANCHES = 'Chain Branches'
CHAIN_BRANCH_COEFF = 'Chain Branch Coefficients'
RCF_BONDS = 'RCF Bonds'
RCF_YIELDS = 'RCF Yields'
RCF_MONOS = 'RCF Monomers'
RCF_BRANCHES = 'RCF Branches'
RCF_BRANCH_COEFF = 'RCF Branch Coefficients'

DEF_TCL_FNAME = "psfgen.tcl"
DEF_CHAIN_ID = 'L'
DEF_PSF_FNAME = 'lignin'
DEF_TOPPAR = "toppar/"

# Data

# Default Gibbs free energy barriers input in kcal/mol (at 298.15 K and 1 atm) from Gani et al., ACS Sustainable Chem.
#     Eng. 2019, 7, 15, 13270-13277, https://doi.org/10.1021/acssuschemeng.9b02506, as described in Orella et al., ACS
#     Sustainable Chem. Eng. 2019, https://doi.org/10.1021/acssuschemeng.9b03534
# Per Terry Gani: the solution state correction is not needed because this barrier is based on the TS vs. the
#     the reactant (hydrogen-bonded) complex
DEF_E_BARRIER_KCAL_MOL = {C5O4: {(G, G): {(MONOMER, MONOMER): 11.2, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (S, G): {(MONOMER, MONOMER): 10.9, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (C, C): {(MONOMER, MONOMER): 11.9, (MONOMER, OLIGOMER): 11.9,
                                          (OLIGOMER, MONOMER): 11.9, (OLIGOMER, OLIGOMER): 11.9}},
                          C5C5: {(G, G): {(MONOMER, MONOMER): 12.5, (MONOMER, OLIGOMER): 15.6,
                                          (OLIGOMER, MONOMER): 15.6, (OLIGOMER, OLIGOMER): 3.8},
                                 (C, C): {(MONOMER, MONOMER): 10.6, (MONOMER, OLIGOMER): 10.6,
                                          (OLIGOMER, MONOMER): 10.6, (OLIGOMER, OLIGOMER): 10.6}},
                          B5: {(G, G): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (G, S): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (C, C): {(MONOMER, MONOMER): 1.9, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8}},
                          BB: {(G, G): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (S, G): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (G, S): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (S, S): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (C, C): {(MONOMER, MONOMER): 7.2, (MONOMER, OLIGOMER): 7.2, (OLIGOMER, MONOMER): 7.2,
                                        (OLIGOMER, OLIGOMER): 7.2}},
                          BO4: {(G, G): {(MONOMER, MONOMER): 6.3, (MONOMER, OLIGOMER): 6.2, (OLIGOMER, MONOMER): 6.2,
                                         (OLIGOMER, OLIGOMER): 6.2},
                                (S, G): {(MONOMER, MONOMER): 9.1, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (G, S): {(MONOMER, MONOMER): 8.9, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (S, S): {(MONOMER, MONOMER): 9.8, (MONOMER, OLIGOMER): 10.4,
                                         (OLIGOMER, MONOMER): 10.4, (OLIGOMER, OLIGOMER): 10.4},
                                (C, C): {(MONOMER, MONOMER): 4.9, (MONOMER, OLIGOMER): 1.3,
                                         (OLIGOMER, MONOMER): 1.3, (OLIGOMER, OLIGOMER): 1.3}},
                          AO4: {(G, G): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (S, G): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (G, S): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (S, S): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (C, C): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7}},
                          B1: {(G, G): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6},
                               (S, G): {(MONOMER, OLIGOMER): 11.7, (OLIGOMER, MONOMER): 11.7,
                                        (OLIGOMER, OLIGOMER): 11.7},
                               (G, S): {(MONOMER, OLIGOMER): 10.7, (OLIGOMER, MONOMER): 10.7,
                                        (OLIGOMER, OLIGOMER): 10.7},
                               (S, S): {(MONOMER, OLIGOMER): 11.9, (OLIGOMER, MONOMER): 11.9,
                                        (OLIGOMER, OLIGOMER): 11.9},
                               (C, C): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6}},
                          OX: {G: {MONOMER: 0.9, OLIGOMER: 6.3}, S: {MONOMER: 0.6, OLIGOMER: 2.2},
                               C: {MONOMER: 0.9, OLIGOMER: 0.9}},
                          Q: {G: {MONOMER: 11.1, OLIGOMER: 11.1}, S: {MONOMER: 11.7, OLIGOMER: 11.7},
                              C: {MONOMER: 11.1, OLIGOMER: 11.1}}}

# These were calculated at 298 K from the DEF_E_BARRIER_KCAL_MOL
DEF_RXN_RATES = {C5O4: {(G, G): {MON_MON: 38335.8499454595, MON_OLI: 123.42064219497, OLI_MON: 123.42064219497,
                                 OLI_OLI: 3698619760.88446},
                        (S, G): {MON_MON: 63607.2504079451, MON_OLI: 123.42064219497, OLI_MON: 123.42064219497,
                                 OLI_OLI: 3698619760.88446},
                        (C, C): {MON_MON: 11762.5514428752, MON_OLI: 11762.5514428752, OLI_MON: 11762.5514428752,
                                 OLI_OLI: 11762.5514428752}},
                 C5C5: {(G, G): {MON_MON: 4272.6614650748, MON_OLI: 22.8235247337417, OLI_MON: 22.8235247337417,
                                 OLI_OLI: 10182226127.8256},
                        (C, C): {MON_MON: 105537.827130874, MON_OLI: 105537.827130874, OLI_MON: 105537.827130874,
                                 OLI_OLI: 105537.827130874}},
                 B5: {(G, G): {MON_MON: 577742199.338807, MON_OLI: 348203044.762421, OLI_MON: 348203044.762421,
                               OLI_OLI: 348203044.762421},
                      (G, S): {MON_MON: 577742199.338807, MON_OLI: 348203044.762421, OLI_MON: 348203044.762421,
                               OLI_OLI: 348203044.762421},
                      (C, C): {MON_MON: 251508346652.295, MON_OLI: 348203044.762421, OLI_MON: 348203044.762421,
                               OLI_OLI: 348203044.762421}},
                 BB: {(G, G): {MON_MON: 958596008.614987, MON_OLI: 958596008.614987, OLI_MON: 958596008.614987,
                               OLI_OLI: 958596008.614987},
                      (S, G): {MON_MON: 106838800.559028, MON_OLI: 106838800.559028, OLI_MON: 106838800.559028,
                               OLI_OLI: 106838800.559028},
                      (G, S): {MON_MON: 106838800.559028, MON_OLI: 106838800.559028, OLI_MON: 106838800.559028,
                               OLI_OLI: 106838800.559028},
                      (S, S): {MON_MON: 958596008.614987, MON_OLI: 958596008.614987, OLI_MON: 958596008.614987,
                               OLI_OLI: 958596008.614987},
                      (C, C): {MON_MON: 32781244.9563151, MON_OLI: 32781244.9563151, OLI_MON: 32781244.9563151,
                               OLI_OLI: 32781244.9563151}},
                 BO4: {(G, G): {MON_MON: 149737307.995724, MON_OLI: 177268075.446635, OLI_MON: 177268075.446635,
                                OLI_OLI: 177268075.446635},
                       (S, G): {MON_MON: 1327137.06459331, MON_OLI: 177268075.446635, OLI_MON: 177268075.446635,
                                OLI_OLI: 177268075.446635},
                       (G, S): {MON_MON: 1860016.49544686, MON_OLI: 177268075.446635, OLI_MON: 177268075.446635,
                                OLI_OLI: 177268075.446635},
                       (S, S): {MON_MON: 407204.170932281, MON_OLI: 147913.960505052, OLI_MON: 147913.960505052,
                                OLI_OLI: 147913.960505052},
                       (C, C): {MON_MON: 1590512704.08189, MON_OLI: 692397441265.176, OLI_MON: 692397441265.176,
                                OLI_OLI: 692397441265.176}},
                 AO4: {(G, G): {MON_MON: 0.00416923882420833, MON_OLI: 0.00416923882420833,
                                OLI_MON: 0.00416923882420833, OLI_OLI: 0.00416923882420833},
                       (S, G): {MON_MON: 0.00416923882420833, MON_OLI: 0.00416923882420833,
                                OLI_MON: 0.00416923882420833, OLI_OLI: 0.00416923882420833},
                       (G, S): {MON_MON: 0.00416923882420833, MON_OLI: 0.00416923882420833,
                                OLI_MON: 0.00416923882420833, OLI_OLI: 0.00416923882420833},
                       (S, S): {MON_MON: 0.00416923882420833, MON_OLI: 0.00416923882420833,
                                OLI_MON: 0.00416923882420833, OLI_OLI: 0.00416923882420833},
                       (C, C): {MON_MON: 0.00416923882420833, MON_OLI: 0.00416923882420833,
                                OLI_MON: 0.00416923882420833, OLI_OLI: 0.00416923882420833}},
                 B1: {(G, G): {MON_OLI: 570707.046887361, OLI_MON: 570707.046887361, OLI_OLI: 570707.046887361},
                      (S, G): {MON_OLI: 16485.5163012079, OLI_MON: 16485.5163012079, OLI_OLI: 16485.5163012079},
                      (G, S): {MON_OLI: 89147.1861838573, OLI_MON: 89147.1861838573, OLI_OLI: 89147.1861838573},
                      (S, S): {MON_OLI: 11762.5514428752, OLI_MON: 11762.5514428752, OLI_OLI: 11762.5514428752},
                      (C, C): {MON_OLI: 570707.046887361, OLI_MON: 570707.046887361, OLI_OLI: 570707.046887361}},
                 OX: {G: {MONOMER: 1360058186601.25, OLIGOMER: 149737307.995724},
                      S: {MONOMER: 2256623024338.81, OLIGOMER: 151583132040.064},
                      C: {MONOMER: 1360058186601.25, OLIGOMER: 1360058186601.25}},
                 Q: {G: {MONOMER: 45384.2962145191, OLIGOMER: 45384.2962145191},
                     S: {MONOMER: 16485.5163012079, OLIGOMER: 16485.5163012079},
                     C: {MONOMER: 45384.2962145191, OLIGOMER: 45384.2962145191}}
                 }

# define dictionary for atoms within each monomer
ATOM_BLOCKS = {G: ('C 0 0 0 0 \n' +  # 1
                   'C 0 0 0 0 \n' +  # 2
                   'C 0 0 0 0 \n' +  # 3
                   'C 0 0 0 0 \n' +  # 4
                   'C 0 0 0 0 \n' +  # 5
                   'C 0 0 0 0 \n' +  # 6
                   'C 0 0 0 0 \n' +  # 7
                   'C 0 0 0 0 \n' +  # 8
                   'C 0 0 0 0 \n' +  # 9
                   'O 0 0 0 0 \n' +  # 9-OH
                   'O 0 0 0 0 \n' +  # 3-OMe
                   'C 0 0 0 0 \n' +  # 3-OMe
                   'O 0 0 0 0 \n'),  # 4-OH
               S: ('C 0 0 0 0 \n' +  # 1
                   'C 0 0 0 0 \n' +  # 2
                   'C 0 0 0 0 \n' +  # 3
                   'C 0 0 0 0 \n' +  # 4
                   'C 0 0 0 0 \n' +  # 5
                   'C 0 0 0 0 \n' +  # 6
                   'C 0 0 0 0 \n' +  # 7
                   'C 0 0 0 0 \n' +  # 8
                   'C 0 0 0 0 \n' +  # 9
                   'O 0 0 0 0 \n' +  # 9-OH
                   'O 0 0 0 0 \n' +  # 3-OMe
                   'C 0 0 0 0 \n' +  # 3-OMe
                   'O 0 0 0 0 \n' +  # 4-OH
                   'O 0 0 0 0 \n' +  # 5-OMe
                   'C 0 0 0 0 \n'),  # 5-OMe
               C: ('C 0 0 0 0 \n' +  # 1
                   'C 0 0 0 0 \n' +  # 2
                   'C 0 0 0 0 \n' +  # 3
                   'C 0 0 0 0 \n' +  # 4
                   'C 0 0 0 0 \n' +  # 5
                   'C 0 0 0 0 \n' +  # 6
                   'C 0 0 0 0 \n' +  # 7
                   'C 0 0 0 0 \n' +  # 8
                   'C 0 0 0 0 \n' +  # 9
                   'O 0 0 0 0 \n' +  # 9-OH
                   'O 0 0 0 0 \n' +  # 3-OH
                   'O 0 0 0 0 \n'),  # 4-OH
               G4: ('C 0 0 0 0 \n' +  # 1
                    'C 0 0 0 0 \n' +  # 2
                    'C 0 0 0 0 \n' +  # 3
                    'C 0 0 0 0 \n' +  # 4
                    'C 0 0 0 0 \n' +  # 5
                    'C 0 0 0 0 \n' +  # 6
                    'C 0 0 0 0 \n' +  # 7
                    'C 0 0 0 0 \n' +  # 8
                    'C 0 0 0 0 \n' +  # 9
                    'O 0 0 0 0 \n' +  # 9-OH
                    'O 0 0 0 0 \n' +  # 3-OMe
                    'C 0 0 0 0 \n' +  # 3-OMe
                    'O 0 0 0 0 RAD=2\n'),  # 4-O
               S4: ('C 0 0 0 0 \n' +  # 1
                    'C 0 0 0 0 \n' +  # 2
                    'C 0 0 0 0 \n' +  # 3
                    'C 0 0 0 0 \n' +  # 4
                    'C 0 0 0 0 \n' +  # 5
                    'C 0 0 0 0 \n' +  # 6
                    'C 0 0 0 0 \n' +  # 7
                    'C 0 0 0 0 \n' +  # 8
                    'C 0 0 0 0 \n' +  # 9
                    'O 0 0 0 0 \n' +  # 9-OH
                    'O 0 0 0 0 \n' +  # 3-OMe
                    'C 0 0 0 0 \n' +  # 3-OMe
                    'O 0 0 0 0 RAD=2\n' +  # 4-O
                    'O 0 0 0 0 \n' +  # 5-OMe
                    'C 0 0 0 0 \n')}  # 5-OMe

# Similarly define dictionary for bonds within each monomer -
# NOTE: THESE MAY NEED TO CHANGE DEPENDING ON INTER-UNIT LINKAGES
BOND_BLOCKS = {G7: ('1 1  2  \n' +  # Aromatic ring 1->2
                    '2 2  3  \n' +  # Aromatic ring 2->3
                    '1 3  4  \n' +  # Aromatic ring 3->4
                    '1 4  5  \n' +  # Aromatic ring 4->5
                    '2 5  6  \n' +  # Aromatic ring 5->6
                    '1 6  1  \n' +  # Aromatic ring 6->1
                    '2 1  7  \n' +  # Quinone methide propyl tail 1->A
                    '1 7  8  \n' +  # Propyl tail A->B
                    '1 8  9  \n' +  # Propyl tail B->G
                    '1 9  10 \n' +  # Gamma hydroxyl G->OH
                    '1 3  11 \n' +  # 3 methoxy 3->O
                    '1 11 12 \n' +  # 3 methoxy O->12
                    '2 4  13 \n'),  # 4 ketone 4->O
               G: ('2 1  2  \n' +  # Aromatic ring 1->2
                   '1 2  3  \n' +  # Aromatic ring 2->3
                   '2 3  4  \n' +  # Aromatic ring 3->4
                   '1 4  5  \n' +  # Aromatic ring 4->5
                   '2 5  6  \n' +  # Aromatic ring 5->6
                   '1 6  1  \n' +  # Aromatic ring 6->1
                   '1 1  7  \n' +  # Ring - propyl tail 1->A
                   '2 7  8  \n' +  # Alkene propyl tail A->B
                   '1 8  9  \n' +  # Propyl tail B->G
                   '1 9  10 \n' +  # Gamma hydroxyl G->OH
                   '1 3  11 \n' +  # 3 methoxy 3->O
                   '1 11 12 \n' +  # 3 methoxy O->12
                   '1 4  13 \n'),  # 4 hydroxyl 4->OH
               S7: ('1 1  2  \n' +  # Aromatic ring 1->2
                    '2 2  3  \n' +  # Aromatic ring 2->3
                    '1 3  4  \n' +  # Aromatic ring 3->4
                    '1 4  5  \n' +  # Aromatic ring 4->5
                    '2 5  6  \n' +  # Aromatic ring 5->6
                    '1 6  1  \n' +  # Aromatic ring 6->1
                    '2 1  7  \n' +  # Quinone methide 1->A
                    '1 7  8  \n' +  # Propyl tail A->B
                    '1 8  9  \n' +  # Propyl tail B->G
                    '1 9  10 \n' +  # Gamma hydroxyl G->OH
                    '1 3  11 \n' +  # 3 methoxy 3->O
                    '1 11 12 \n' +  # 3 methoxy O->12
                    '2 4  13 \n' +  # 4 ketone 4->O
                    '1 5  14 \n' +  # 5 methoxy 5->O
                    '1 14 15 \n'),  # 5 methoxy O->15
               S: ('2 1  2  \n' +  # Aromatic ring 1->2
                   '1 2  3  \n' +  # Aromatic ring 2->3
                   '2 3  4  \n' +  # Aromatic ring 3->4
                   '1 4  5  \n' +  # Aromatic ring 4->5
                   '2 5  6  \n' +  # Aromatic ring 5->6
                   '1 6  1  \n' +  # Aromatic ring 6->1
                   '1 1  7  \n' +  # Ring - propyl tail 1->A
                   '2 7  8  \n' +  # Alkene propyl tail A->B
                   '1 8  9  \n' +  # Propyl tail B->G
                   '1 9  10 \n' +  # Gamma hydroxyl G->OH
                   '1 3  11 \n' +  # 3 methoxy 3->O
                   '1 11 12 \n' +  # 3 methoxy O->12
                   '1 4  13 \n' +  # 4 hydroxyl 4->OH
                   '1 5  14 \n' +  # 5 methoxy 5->O
                   '1 14 15 \n'),  # 5 methoxy O->15
               C: ('2 1  2  \n' +  # Aromatic ring 1->2
                   '1 2  3  \n' +  # Aromatic ring 2->3
                   '2 3  4  \n' +  # Aromatic ring 3->4
                   '1 4  5  \n' +  # Aromatic ring 4->5
                   '2 5  6  \n' +  # Aromatic ring 5->6
                   '1 6  1  \n' +  # Aromatic ring 6->1
                   '1 1  7  \n' +  # Ring - propyl tail 1->A
                   '2 7  8  \n' +  # Alkene propyl tail A->B
                   '1 8  9  \n' +  # Propyl tail B->G
                   '1 9  10 \n' +  # Gamma hydroxyl G->OH
                   '1 3  11 \n' +  # 3 hydroxyl 3->O
                   '1 4  12 \n')}  # 4 hydroxyl 4->OH


class Event:
    """
    Class definition for the event_dict that occur during lignification process. While more specific than the monomer
    class, this class is also easily extensible such that other reactivity could be incorporated in the future. The
    most important features lie in the event dictionary that specifies the changes that need to occur for a given
    reaction.

    ATTRIBUTES:
        key     -- str  --  Name of the bond that is being formed using the traditional nomenclature
                            (e.g. C5o4, BO4, C5C5 etc.)
        index   -- list --  N x 1 list of integers containing the unique monomer identifiers involved in the reaction
                            (N = 2 for bimolecular, 1 for unimolecular)
        rate    -- float--  Scalar floating point value with the rate of this particular reaction event
        bond    -- list --  2 x 1 list of the updates that need to occur to the adjacency matrix to reflect the
                            bond formations

    METHODS:
        N/A no declared public methods

    Events are compared based on the monomers that are involved in the event, the specific bond being formed, and the
    value updates to the adjacency matrix.
    """
    # create a dict that maps event keys onto numerical changes in monomer
    # state where the value is a tuple of (new reactant active point, openPos0, openPos1)
    eventDict = {BO4: ((-1, 7), (), (7,)), BB: ((0, 0), (), ()),
                 C5C5: ((0, 0), (), ()), C5O4: ((-1, 0), (), ()),
                 B5: ((-1, 0), (), ()), B1: ((0, 7), (), (7,)),
                 AO4: ((-1, 4), (), ()), Q: (0, (1,), ()), OX: (4, (), ())}

    activeDict = {(4, 8): (0, 1), (8, 4): (1, 0), (4, 5): (0, 1), (5, 4): (1, 0), (5, 8): (0, 1), (8, 5): (1, 0),
                  (1, 8): (0, 1), (8, 1): (1, 0), (4, 7): (0, 1), (7, 4): (1, 0), (5, 5): (0, 1), (8, 8): (0, 1)}

    def __init__(self, event_name, ids, rate=0, bond=()):
        """
        Straightforward constructor that takes the name of bond being formed, the indices of affected monomers, the
        rate of the event, and the updates to the adjacency matrix.

        :param event_name: str, assigned to key attribute - has the name of the bond being formed or the reaction
                               occurring
        :param ids:  int, N x 1 list assigned to index - keeps track of the monomers affected by this event
        :param rate: float, the rate of the event (make sure units are consistent - I typically use GHz)
        :param bond: int, 2 x 1 list of updates to the adjacency matrix when a bond is formed
        :return: new instance of an event object that corresponds to the attributes provided
        """
        self.key = event_name
        self.index = ids
        self.rate = rate
        self.bond = bond

    def __str__(self):
        if self.key == Q or self.key == OX:
            msg = f'Performing {self.key} on index {str(self.index[0])}'
        else:
            msg = f'Forming {self.key} bond between indices {str(self.index)} ({ADJ_MATRIX} update {str(self.bond)})'
        return msg

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.index == other.index and self.bond == other.bond and self.key == other.key

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        # Note: changed from invoking python's hash function to provide more consistent output for testing
        #     see https://docs.python.org/3/reference/datamodel.html#object.__hash__
        #     "By default, the __hash__() values of str and bytes objects are “salted” with an unpredictable random
        #     value. Although they remain constant within an individual Python process, they are not predictable
        #     between repeated invocations of Python."
        # Original hash method, which did not allow testing because of random salting:
        #    return hash((tuple(self.index), self.key, self.bond))
        # Replacement hash method which gave consistent results, but unexpected changed outcomes
        #    key_as_num = sum([ord(x) % 32 for x in self.key])
        #    return key_as_num + sum(self.index) * 1000 + int(self.rate * 10000)
        # The bash below is repeatable and gives similar results to the original hash. It is not directly invoked
        #     in the package (unsettling that the hash method used for dictionary keys would change results), but it
        #     is left in case there is any behind the scenes hashing
        if not self.index:
            index_join = 0
        else:
            index_join = int("".join([str(x) for x in self.index]))
        index_bytes = index_join.to_bytes((index_join.bit_length() + 7) // 8, 'big')
        key_bytes = self.key.encode()
        bond_list_str = "".join([str(x) for x in self.bond])
        bond_list_bytes = bond_list_str.encode()
        event_bytes = b''.join([index_bytes, key_bytes, bond_list_bytes])
        # the hash call below "right-sizes" the value
        return hash(int.from_bytes(event_bytes, 'big'))


class Monomer:
    """
    Class definition for monomer objects. This is highly generic such that it can be easily extended to more types of
    monolignols. The class is primarily used for storing information about each monolignol included in a simulation of
    polymerization.

    ATTRIBUTES:
        identity    -- int     --  unique integer for indexing monomer (also the hash value)
        type        -- str     --  monolignol variety
                                        G = coniferyl alcohol
                                        S = sinapyl alcohol
                                        C = caffeoyl alcohol
        parent      -- Monomer --  monomer object that has the smallest unique identifier in a chain (can be used for
                                    sizing fragments)
        size        -- int     --  integer with the size of the fragment if parent == self
        active      -- int     --  integer with the location [1-9] of the active site on the monomer (-1 means
                                    inactive)
        open        -- set     --  set of units with location [1-9] of open positions on the monomer
        connectedTo -- set     --  set of integer identities of other monomers that this monomer is connected to

    METHODS:
        N/A - no defined public methods

    Monomers are mutable objects, but compare and hash only on the identity, which should be treated as a constant
    """

    def __init__(self, unit_type, i):
        """
        Constructor for the Monomer class, which sets the properties depending on what monolignol is being represented.
        The only attributes that need to be set are the species [0-2], and the unique integer identifier. Everything
        else will be computed from these values. The active site is initially set to 0, indicating that the monomer is
        not oxidized, but can be eventually. The open positions are either {4,5,8} or {4,8} depending on whether a
        5-methoxy is present. The parent is set to be self until connections occur. The set of monomers that are
        connected to begin as just containing the self's identity.

        Example calls are below:
            mon = Monomer(G, 0) # Makes a guaiacol unit monomer with ID = 0
            mon = Monomer(S, 0) # Makes a syringol unit monomer with ID = 0 (not recommended to repeat IDs)
            mon = Monomer(H, 0) # Makes a caffeoyl unit monomer with ID = 0
            mon = Monomer(S, n) # Makes a sinapyl alcohol with ID = n

        :param unit_type: str, monomer type
        :param i: int, unique identifier for the monomer
        Outputs:
            New instance of a monomer object with the desired attributes
        """

        self.identity = i
        self.type = unit_type
        self.parent = self
        self.size = 1

        # The active attribute will be the position of an active position, if 0
        # the monomer is not activated yet, -1 means it can never be activated
        self.active = 0
        if unit_type == G or unit_type == C:
            self.open = {4, 5, 8}
        elif unit_type == S:
            self.open = {4, 8}
        else:
            # todo: update once H is added
            raise InvalidDataError(f"Encountered unit type {unit_type},  but only the following types are "
                                   f"currently available: 'G' ({G}), 'S' ({S}), 'C' ({C})")
        self.connectedTo = {i}

    def __str__(self):
        return f'{self.identity}: {MONOLIG_OHS[self.type]} alcohol is connected to {self.connectedTo} and active at ' \
               f'position {self.active}'

    def __repr__(self):
        representation = f'{self.identity}: {MONOLIG_OHS[self.type]} alcohol \n'
        return representation

    def __eq__(self, other):  # Always compare monomers by identity alone. This should always be a unique identifier
        return self.identity == other.identity

    def __lt__(self, other):
        return self.identity < other.identity

    def __hash__(self):
        return self.identity

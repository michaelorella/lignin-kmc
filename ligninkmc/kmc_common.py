# !/usr/bin/env python
# coding=utf-8

# lignin subunits (phenylpropanoids)
G = 'G'  # guaiacol (coniferyl)
S = 'S'  # syringyl (sinapyl)
C = 'C'  # caffeoyl
H = 'H'  # p-hydroxyphenyl
S4 = 'S4'
G4 = 'G4'
G7 = 'G7'
LIGNIN_SUBUNITS = {0: G, 1: S, 2: C, 3: H}
MONOLIG_OHS = {0: 'coniferyl', 1: 'sinapyl', 2: 'caffeoyl', 3: 'p-coumaryl'}
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

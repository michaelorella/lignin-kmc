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
LIGNIN_SUBUNITS = {0: G, 1: S, 2: C}  # perhaps to be added later: 3: H
MONOLIG_OHS = {0: 'coniferyl', 1: 'sinapyl', 2: 'caffeoyl'}  # perhaps to be added later: 3: 'p-coumaryl'
SG_RATIO = 'sg_ratio'
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

# reaction types
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
DEF_E_BARRIER_KCAL_MOL = {C5O4: {(0, 0): {(MONOMER, MONOMER): 11.2, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (1, 0): {(MONOMER, MONOMER): 10.9, (MONOMER, OLIGOMER): 14.6,
                                          (OLIGOMER, MONOMER): 14.6, (OLIGOMER, OLIGOMER): 4.4},
                                 (2, 2): {(MONOMER, MONOMER): 11.9, (MONOMER, OLIGOMER): 11.9,
                                          (OLIGOMER, MONOMER): 11.9, (OLIGOMER, OLIGOMER): 11.9}},
                          C5C5: {(0, 0): {(MONOMER, MONOMER): 12.5, (MONOMER, OLIGOMER): 15.6,
                                          (OLIGOMER, MONOMER): 15.6, (OLIGOMER, OLIGOMER): 3.8},
                                 (2, 2): {(MONOMER, MONOMER): 10.6, (MONOMER, OLIGOMER): 10.6,
                                          (OLIGOMER, MONOMER): 10.6, (OLIGOMER, OLIGOMER): 10.6}},
                          B5: {(0, 0): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (0, 1): {(MONOMER, MONOMER): 5.5, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8},
                               (2, 2): {(MONOMER, MONOMER): 1.9, (MONOMER, OLIGOMER): 5.8, (OLIGOMER, MONOMER): 5.8,
                                        (OLIGOMER, OLIGOMER): 5.8}},
                          BB: {(0, 0): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (1, 0): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (0, 1): {(MONOMER, MONOMER): 6.5, (MONOMER, OLIGOMER): 6.5, (OLIGOMER, MONOMER): 6.5,
                                        (OLIGOMER, OLIGOMER): 6.5},
                               (1, 1): {(MONOMER, MONOMER): 5.2, (MONOMER, OLIGOMER): 5.2, (OLIGOMER, MONOMER): 5.2,
                                        (OLIGOMER, OLIGOMER): 5.2},
                               (2, 2): {(MONOMER, MONOMER): 7.2, (MONOMER, OLIGOMER): 7.2, (OLIGOMER, MONOMER): 7.2,
                                        (OLIGOMER, OLIGOMER): 7.2}},
                          BO4: {(0, 0): {(MONOMER, MONOMER): 6.3, (MONOMER, OLIGOMER): 6.2, (OLIGOMER, MONOMER): 6.2,
                                         (OLIGOMER, OLIGOMER): 6.2},
                                (1, 0): {(MONOMER, MONOMER): 9.1, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (0, 1): {(MONOMER, MONOMER): 8.9, (MONOMER, OLIGOMER): 6.2,
                                         (OLIGOMER, MONOMER): 6.2, (OLIGOMER, OLIGOMER): 6.2},
                                (1, 1): {(MONOMER, MONOMER): 9.8, (MONOMER, OLIGOMER): 10.4,
                                         (OLIGOMER, MONOMER): 10.4, (OLIGOMER, OLIGOMER): 10.4},
                                (2, 2): {(MONOMER, MONOMER): 4.9, (MONOMER, OLIGOMER): 1.3,
                                         (OLIGOMER, MONOMER): 1.3, (OLIGOMER, OLIGOMER): 1.3}},
                          AO4: {(0, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (1, 0): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (0, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (1, 1): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7},
                                (2, 2): {(MONOMER, MONOMER): 20.7, (MONOMER, OLIGOMER): 20.7,
                                         (OLIGOMER, MONOMER): 20.7, (OLIGOMER, OLIGOMER): 20.7}},
                          B1: {(0, 0): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6},
                               (1, 0): {(MONOMER, OLIGOMER): 11.7, (OLIGOMER, MONOMER): 11.7,
                                        (OLIGOMER, OLIGOMER): 11.7},
                               (0, 1): {(MONOMER, OLIGOMER): 10.7, (OLIGOMER, MONOMER): 10.7,
                                        (OLIGOMER, OLIGOMER): 10.7},
                               (1, 1): {(MONOMER, OLIGOMER): 11.9, (OLIGOMER, MONOMER): 11.9,
                                        (OLIGOMER, OLIGOMER): 11.9},
                               (2, 2): {(MONOMER, OLIGOMER): 9.6, (OLIGOMER, MONOMER): 9.6,
                                        (OLIGOMER, OLIGOMER): 9.6}},
                          OX: {0: {MONOMER: 0.9, OLIGOMER: 6.3}, 1: {MONOMER: 0.6, OLIGOMER: 2.2},
                               2: {MONOMER: 0.9, OLIGOMER: 0.9}},
                          Q: {0: {MONOMER: 11.1, OLIGOMER: 11.1}, 1: {MONOMER: 11.7, OLIGOMER: 11.7},
                              2: {MONOMER: 11.1, OLIGOMER: 11.1}}}

# These were calculated at 298 K from the DEF_E_BARRIER_KCAL_MOL
DEF_RXN_RATES = {C5O4: {(0, 0): {MON_MON: 38335.5972148372, MON_OLI: 123.419593715543, OLI_MON: 123.419593715543,
                                 OLI_OLI: 3698609451.84164},
                        (1, 0): {MON_MON: 63606.8417529500, MON_OLI: 123.419593715543, OLI_MON: 123.419593715543,
                                 OLI_OLI: 3698609451.84164},
                        (2, 2): {MON_MON: 11762.4692901771, MON_OLI: 11762.4692901771, OLI_MON: 11762.4692901771,
                                 OLI_OLI: 11762.4692901771}},
                 C5C5: {(0, 0): {MON_MON: 4272.63018912086, MON_OLI: 22.8233180720356, OLI_MON: 22.8233180720356,
                                 OLI_OLI: 10182201166.0217},
                        (2, 2): {MON_MON: 105537.166803781, MON_OLI: 105537.166803781, OLI_MON: 105537.166803781,
                                 OLI_OLI: 105537.166803781}},
                 B5: {(0, 0): {MON_MON: 577740233.381881, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315},
                      (0, 1): {MON_MON: 577740233.381881, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315},
                      (2, 2): {MON_MON: 251507997491.634, MON_OLI: 348201801.431315, OLI_MON: 348201801.431315,
                               OLI_OLI: 348201801.431315}},
                 BB: {(0, 0): {MON_MON: 958592907.607318, MON_OLI: 958592907.607318, OLI_MON: 958592907.607318,
                               OLI_OLI: 958592907.607318},
                      (1, 0): {MON_MON: 106838377.218107, MON_OLI: 106838377.218107, OLI_MON: 106838377.218107,
                               OLI_OLI: 106838377.218107},
                      (0, 1): {MON_MON: 106838377.218107, MON_OLI: 106838377.218107, OLI_MON: 106838377.218107,
                               OLI_OLI: 106838377.218107},
                      (1, 1): {MON_MON: 958592907.607318, MON_OLI: 958592907.607318, OLI_MON: 958592907.607318,
                               OLI_OLI: 958592907.607318},
                      (2, 2): {MON_MON: 32781102.2219828, MON_OLI: 32781102.2219828, OLI_MON: 32781102.2219828,
                               OLI_OLI: 32781102.2219828}},
                 BO4: {(0, 0): {MON_MON: 149736731.431189, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (1, 0): {MON_MON: 1327129.87498242, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (0, 1): {MON_MON: 1860006.62719604, MON_OLI: 177267402.79460, OLI_MON: 177267402.794600,
                                OLI_OLI: 177267402.794600},
                       (1, 1): {MON_MON: 407201.805441432, MON_OLI: 147913.051594236, OLI_MON: 147913.051594236,
                                OLI_OLI: 147913.051594236},
                       (2, 2): {MON_MON: 1590507825.87210, MON_OLI: 692396712512.577, OLI_MON: 692396712512.577,
                                OLI_OLI: 692396712512.577}},
                 AO4: {(0, 0): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (1, 0): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (0, 1): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (1, 1): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265},
                       (2, 2): {MON_MON: 0.00416918917397265, MON_OLI: 0.00416918917397265,
                                OLI_MON: 0.00416918917397265, OLI_OLI: 0.00416918917397265}},
                 B1: {(0, 0): {MON_OLI: 570703.795464849, OLI_MON: 570703.795464849, OLI_OLI: 570703.795464849},
                      (1, 0): {MON_OLI: 16485.4030071542, OLI_MON: 16485.4030071542, OLI_OLI: 16485.4030071542},
                      (0, 1): {MON_OLI: 89146.6234207596, OLI_MON: 89146.6234207596, OLI_OLI: 89146.6234207596},
                      (1, 1): {MON_OLI: 11762.4692901771, OLI_MON: 11762.4692901771, OLI_OLI: 11762.4692901771},
                      (2, 2): {MON_OLI: 570703.795464849, OLI_MON: 570703.795464849, OLI_OLI: 570703.795464849}},
                 OX: {0: {MONOMER: 1360057059567.54, OLIGOMER: 149736731.431189},
                      1: {MONOMER: 2256621533195.09, OLIGOMER: 151582896154.443},
                      2: {MONOMER: 1360057059567.54, OLIGOMER: 1360057059567.54}},
                 Q: {0: {MONOMER: 45383.9995564285, OLIGOMER: 45383.9995564285},
                     1: {MONOMER: 16485.4030071542, OLIGOMER: 16485.4030071542},
                     2: {MONOMER: 45383.9995564285, OLIGOMER: 45383.9995564285}}
                 }


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
        key_as_num = sum([ord(x) % 32 for x in self.key])
        return key_as_num + sum(self.index) * 1000 + int(self.rate * 10000)


class Monomer:
    """
    Class definition for monomer objects. This is highly generic such that it can be easily extended to more types of
    monolignols. The class is primarily used for storing information about each monolignol included in a simulation of
    polymerization.

    ATTRIBUTES:
        identity    -- uint     --  unique integer for indexing monomer (also the hash value)
        type        -- uint     --  integer switch for monolignol variety
                                        0 = coniferyl alcohol
                                        1 = sinapyl alcohol
                                        2 = caffeoyl alcohol
        parent      -- Monomer  --  monomer object that has the smallest unique identifier in a chain (can be used for
                                    sizing fragments)
        size        -- uint     --  integer with the size of the fragment if parent == self
        active      -- int      --  integer with the location [1-9] of the active site on the monomer (-1 means
                                    inactive)
        open        -- set      --  set of units with location [1-9] of open positions on the monomer
        connectedTo -- set      --  set of integer identities of other monomers that this monomer is connected to

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
            mon = Monomer(0, 0) # Makes a guaiacol unit monomer with ID = 0
            mon = Monomer(1, 0) # Makes a syringol unit monomer with ID = 0 (not recommended to repeat IDs)
            mon = Monomer(2, 0) # Makes a caffeoyl unit monomer with ID = 0
            mon = Monomer(1, n) # Makes a sinapyl alcohol with ID = n

        :param unit_type: int    -- integer switch of the monomer type
        :param i: int    -- unique identifier for the monomer
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
        if unit_type == 0 or unit_type == 2:
            self.open = {4, 5, 8}
        elif unit_type == 1:
            self.open = {4, 8}
        else:
            raise InvalidDataError(f'Encountered unit type {unit_type},  but only the following types are '
                                   f'currently available: {MONOLIG_OHS}')
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

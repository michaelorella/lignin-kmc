# !/usr/bin/env python
# coding=utf-8

from ligninkmc.kmc_common import (MONOLIG_OHS)


class Monomer:
    """
    Class definition for monomer objects. This is highly generic such that it can be easily extended to more types of
    monolignols. The class is primarily used for storing information about each monolignol included in a simulation of
    polymerization.

    ATTRIBUTES:
        identity    -- uint     --  unique integer for indexing monomers (also the hash value)
        type        -- uint     --  integer switch for monolignol variety
                                        0 = coniferyl alcohol
                                        1 = sinapyl alcohol
                                        2 = caffeoyl alcohol)
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

    def __init__(self, unit, i):
        """
        Constructor for the Monomer class, which sets the properties depending on what monolignol is being represented.
        The only attributes that need to be set are the species [0-2], and the unique integer identifier. Everything
        else will be computed from these values. The active site is initially set to 0, indicating that the monomer is
        not oxidized, but can be eventually. The open positions are either {4,5,8} or {4,8} depending on whether a
        5-methoxy is present. The parent is set to be self until connections occur. The set of monomers that are
        connected to begin as just containing the self's identity.

        Example calls are below:
            mon = Monomer(0,0) #Makes a guaiacol unit monomer with ID = 0
            mon = Monomer(1,0) #Makes a syringol unit monomer with ID = 0 (not recommended to repeat IDs)
            mon = Monomer(2,0) #Makes a caffeoyl unit monomer with ID = 0
            mon = Monomer(1,n) #Makes a sinapyl alcohol with ID = n

        :param unit: --  uint    -- integer switch of the monomer type
        :param i:    --  uint    -- unique identifier for the monomer
        Outputs:
            New instance of a monomer object with the desired attributes
        """

        self.identity = i
        self.type = unit
        self.parent = self
        self.size = 1

        # The active attribute will be the position of an active position, if 0
        # the monomer is not activated yet, -1 means it can never be activated
        self.active = 0
        if unit == 0:
            self.open = {4, 5, 8}
        elif unit == 1:
            self.open = {4, 8}
        else:
            self.open = {4, 5, 8}

        self.connectedTo = {i}

    def __str__(self):
        return f'{self.identity}: {MONOLIG_OHS[self.type]} alcohol is connected to unit {self.connectedTo} and ' \
               f'active at position {self.active}'

    def __repr__(self):
        # representation
        return f'{self.identity}: {MONOLIG_OHS[self.type]} alcohol \n'

    def __eq__(self, other):  # Always compare monomers by identity alone. This should always be a unique identifier
        return self.identity == other.identity

    def __lt__(self, other):
        return self.identity < other.identity

    def __hash__(self):
        return self.identity

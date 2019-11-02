# !/usr/bin/env python
# coding=utf-8

from ligninkmc.kmc_common import (AO4, B1, B5, BB, BO4, C5C5, C5O4, OX, Q)


class Event:
    """
    Class definition for the events that occur during lignification process. While more specific than the monomer
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
    # create dictionary that maps event keys onto numerical changes in monomer
    # state where the value is a tuple of (new reactant active point,openPos0,openPos1)
    eventDict = {BO4: ((-1, 7), (), (7,)),
                 BB: ((0, 0), (), ()),
                 C5C5: ((0, 0), (), ()),
                 C5O4: ((-1, 0), (), ()),
                 B5: ((-1, 0), (), ()),
                 B1: ((0, 7), (), (7,)),
                 AO4: ((-1, 4), (), ()),
                 Q: (0, (1,), ()),
                 OX: (4, (), ())
                 }

    # Create dictionary to properly order the event indices
    activeDict = {(4, 8): (0, 1),
                  (8, 4): (1, 0),
                  (4, 5): (0, 1),
                  (5, 4): (1, 0),
                  (5, 8): (0, 1),
                  (8, 5): (1, 0),
                  (1, 8): (0, 1),
                  (8, 1): (1, 0),
                  (4, 7): (0, 1),
                  (7, 4): (1, 0),
                  (5, 5): (0, 1),
                  (8, 8): (0, 1)}

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
            msg = ('Performing ' + self.key + ' on ' + str(self.index) + str(self.bond) + '\n')
        else:
            msg = ('Forming ' + self.key + ' bond between ' + str(self.index) + str(self.bond) + '\n')
        return msg

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.index == other.index and self.bond == other.bond and self.key == other.key

    def __hash__(self):
        return hash((tuple(self.index), self.key, self.bond))

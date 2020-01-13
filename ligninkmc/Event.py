OX = 'oxidation'
Q = 'hydration'
ADJ_MATRIX = 'adjacency_matrix'


class Event:
    '''
    Class definition for the events that occur during lignification process. While more specific than the monomer class, this class is also easily extensible such that other reactivity could be incorporated in the future. The most important features lie in the event dictionary that specifies the changes that need to occur for a given reaction. 

    ATTRIBUTES:
        key     -- str  --  Name of the bond that is being formed using the traditional nomenclature (e.g. '5o4', 'bo4', '55' etc.)
        index   -- list --  N x 1 list of integers containing the unique monomer identifiers involved in the reaction (N = 2 for bimolecular, 1 for unimolecular)
        rate    -- float--  Scalar floating point value with the rate of this particular reaction event
        bond    -- list --  2 x 1 list of the updates that need to occur to the adjacency matrix to reflect the bond formations

    METHODS:
        N/A no declared public methods

    Events are compared based on the monomers that are involved in the event, the specific bond being formed, and the value updates to the adjacency matrix. 
    '''

    
    #create dictionary that maps event keys onto numerical changes in monomer
    #state where the value is a tuple of (new reactant active point,openPos0,openPos1)
    eventDict = {'bo4':((-1,7),(),(7,)),
                 'bb':((0,0),(),()),
                 '55':((0,0),(),()),
                 '5o4':((-1,0),(),()),
                 'b5':((-1,0),(),()),
                 'b1':((0,7),(),(7,)),
                 'ao4':((-1,4),(),()),
                 'q':(0, (1,), ()),
                 'ox':(4, (), ())
                 }

    #Create dictionary to properly order the event indices
    activeDict = {(4,8):(0,1),
                  (8,4):(1,0),
                  (4,5):(0,1),
                  (5,4):(1,0),
                  (5,8):(0,1),
                  (8,5):(1,0),
                  (1,8):(0,1),
                  (8,1):(1,0),
                  (4,7):(0,1),
                  (7,4):(1,0),
                  (5,5):(0,1),
                  (8,8):(0,1)}
                 

    def __init__(self,eventName,ids,rate=0,bond=()):
        '''
        Straightforward constructor that takes the name of bond being formed, the indices of affected monomers, the rate of the event, and the updates to the adjacency matrix.

        Inputs:
            eventName   -- str  -- assigned to key attribute - has the name of the bond being formed or the reaction occurring
            ids         -- int  -- N x 1 list assigned to index - keeps track of the monomers affected by this event
            rate        -- float-- the rate of the event (make sure units are consistent - I typically use GHz)
            bond        -- int  -- 2 x 1 list of updates to the adjacency matrix when a bond is formed

        Outputs:
            new instance of an event object that corresponds to the attributes provided
        '''
        self.key = eventName
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
    
    def __eq__(self,other):
        return self.index == other.index and self.bond == other.bond and self.key == other.key
    
    def __hash__(self):
        # attempt at repeatable hash
        # if not self.index:
        #     index_join = 0
        # else:
        #     index_join = int("".join([str(x) for x in self.index]))
        # index_bytes = index_join.to_bytes((index_join.bit_length() + 7) // 8, 'big')
        # key_bytes = self.key.encode()
        # bond_list_str = "".join([str(x) for x in self.bond])
        # bond_list_bytes =  bond_list_str.encode()
        # event_bytes = b''.join([index_bytes, key_bytes, bond_list_bytes])
        # temp = int.from_bytes(event_bytes, 'big')
        # # the hash call below "right-sizes" the value, but since it is hashing an int, will be repeatable
        # event_hash = hash(temp)

        # # original hash
        event_hash = hash ((tuple(self.index), self.key, self.bond))

        return event_hash

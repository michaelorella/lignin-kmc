'''
Written by:     Michael Orella
Original:       6 June 2018
Last Edited:    21 December 2018

Code base for simulating the in planta polymerization of monolignols through generic Gillespie algorithm adaptations.
Monolignols can be handled as either coniferyl alcohol, sinapyl alcohol, or caffeoyl alcohol, however extensions
should be easy given implementation choices. Within the module, there are two classes - monomers and events - code
for analyzing the results of a simulation and for running an individual simulation. Use cases for the module are
shown below.
#
# >>> import ligninkmc as kmc
# >>> mons = [ kmc.Monomer ( 1 , i ) for i in range(5) ]
# >>> startEvs = [ kmc.Event ( OX , [i] , rates[OX][1]['monomer'] ) for i in range(5) ]
# >>> state = { mons[i] : {startEvents[i]} for i in range(5) }
# >>> events = { startEvents[i] for i in range(5) }
# >>> events.add( kmc.Event( 'grow' , [ ] , rate = 0 , bond = 1 ) )
# >>> res = kmc.run_kmc( tFinal = 1e9 , rates = rates, initialState = state, initialEvents = events)

{'monomers': _____ , 'adjacency_matrix': _______ , 'time': ______ }
'''

#Import python packages for data processing
from collections import Counter

import scipy.sparse as sp
import numpy as np

#Import classes in this package
from common_wrangler.common import round_sig_figs, InvalidDataError
import copy

G = 'guaiacol'  # (coniferyl alcohol)
S = 'syringyl'  # (sinapyl alcohol)
C = 'caffeoyl'  # (caffeyl alcohol)
INT_TO_TYPE_DICT = {0: G, 1: S}

ADJ_MATRIX = 'adjacency_matrix'
MONOMER = 'monomer'
OLIGOMER = 'oligomer'
# bond types
AO4 = 'ao4'
B1 = 'b1'
B1_ALT = 'b1alt'
B5 = 'b5'
BB = 'bb'
BO4 = 'bo4'
C5C5 = '55'
C5O4 = '5o4'
OX = 'oxidation'
Q = 'hydration'
GROW = 'grow'

MAX_NUM_DECIMAL = 8


class Monomer:
    '''
    Class definition for monomer objects. This is highly generic such that it can be easily extended to more types of monolignols. The class is primarily used for storing information about each monolignol included in a simulation of polymerization.

    ATTRIBUTES:
        identity    -- uint     --  unique integer for indexing monomers (also the hash value)
        type        -- uint     --  integer switch for monolignol variety (0 = coniferyl alcohol, 1 = sinapyl alcohol, 2 = caffeoyl alcohol)
        parent      -- Monomer  --  monomer object that has the smallest unique identifier in a chain (can be used for sizing fragments)
        size        -- uint     --  integer with the size of the fragment if parent == self
        active      -- int      --  integer with the location [1-9] of the active site on the monomer (-1 means inactive)
        open        -- set      --  set of units with location [1-9] of open positions on the monomer
        connectedTo -- set      --  set of integer identities of other monomers that this monomer is connected to

    METHODS:
        N/A - no defined public methods

    Monomers are mutable objects, but compare and hash only on the identity, which should be treated as a constant
    '''

    #Types
    # 0 == Guaiacol
    # 1 == Syringol
    # 2 == Caffeoyl
    def __init__(self,unit,i):
        '''
        Constructor for the Monomer class, which sets the properties depending on what monolignol is being represented. The only attributes that need to be set are the species [0-2], and the unique integer identifier. Everything else will be computed from these values. The active site is initially set to 0, indicating that the monomer is not oxidized, but can be eventually. The open positions are either {4,5,8} or {4,8} depending on whether a 5-methoxy is present. The parent is set to be self until connections occur. The set of monomers that are connectedTo begin as just containing the self's identity.

        Inputs:
            unit    --  uint    -- integer switch of the monomer type
            i       --  uint    -- unique identifier for the monomer
        Outputs:
            New instance of a monomer object with the desired attributes

        Example calls are below:
        #
        # >>> mon = Monomer(0,0) #Makes a guaiacol unit monomer with ID = 0
        # >>> mon = Monomer(1,0) #Makes a syringol unit monomer with ID = 0 (not recommended to repeat IDs)
        # >>> mon = Monomer(2,0) #Makes a caffeoyl unit monomer with ID = 0
        # >>> mon = Monomer(1,n) #Makes a sinapyl alcohol with ID = n
        '''

        self.identity = i
        self.type = unit
        self.parent = self
        self.size = 1

        #The active attribute will be the position of an active position, if 0
        #the monomer is not activated yet, -1 means it can never be activated
        self.active = 0
        if unit == G:
            self.open = {4,5,8}
        elif unit == S:
            self.open = {4,8}
        else:
            self.open = {4,5,8}

        self.connectedTo = {i}

    def __str__(self):
        trans = {G:'coniferyl',S:'sinapyl', C:'caffeoyl'}
        return f'{self.identity}: {trans[self.type]} alcohol is connected to {self.connectedTo} and active at {self.active}'

    def __repr__(self):
        trans = {G:'coniferyl',S:'sinapyl',C:'caffeoyl'}
        representation = f'{self.identity}: {trans[self.type]} alcohol \n'
        return representation

    def __eq__(self,other): #Always compare monomers by identity alone. This should always be a unique identifier
        return self.identity == other.identity

    def __lt__(self,other):
        return self.identity < other.identity

    def __hash__(self):
        return self.identity


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
    activeDict = {(4,8):(0,1), (8,4):(1,0), (4,5):(0,1), (5,4):(1,0), (5,8):(0,1), (8,5):(1,0), (1,8):(0,1),
                  (8,1):(1,0), (4,7):(0,1), (7,4):(1,0), (5,5):(0,1), (8,8):(0,1)}

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
        if not self.index:
            index_join = 0
        else:
            index_join = int("".join([str(x) for x in self.index]))
        index_bytes = index_join.to_bytes((index_join.bit_length() + 7) // 8, 'big')
        key_bytes = self.key.encode()
        bond_list_str = "".join([str(x) for x in self.bond])
        bond_list_bytes =  bond_list_str.encode()
        event_bytes = b''.join([index_bytes, key_bytes, bond_list_bytes])
        temp = int.from_bytes(event_bytes, 'big')
        # the hash call below "right-sizes" the value, but since it is hashing an int, will be repeatable
        event_hash = hash(temp)

        # # original hash
        # event_hash = hash ((tuple(self.index), self.key, self.bond))

        return  event_hash

# ANALYSIS CODE #

def findFragments(adj = None):
    '''
    Implementation of a modified depth first search on the adjacency matrix provided to identify isolated graphs within the superstructure. This allows us to easily track the number of isolated fragments and the size of each of these fragments. This implementation does not care about the specific values within the adjacency matrix, but effectively treats the adjacency matrix as boolean.

    Inputs:
        adj     --  DOK_MATRIX  -- NxN sparse matrix in dictionary of keys format that contains all of the connectivity information for the current lignification state

    Outputs:
        A set of sets of the unique integer identifiers for the monomers contained within each fragment.

    >>> a = sp.dok_matrix((2,2))
    >>> findFragments(a)

    {{0},{1}}

    >>> a.resize((5,5))
    >>> a[0,1] = 1; a[1,0] = 1; a[0,2] = 1; a[2,0] = 1; a[3,4] = 1; a[4,3] = 1
    >>> findFragments(a)

    {{0,1,2},{3,4}}

    # >>> a = sp.dok_matrix((5,5))
    # >>> a[0,4] = 1; a[4,0] = 1
    # >>> findFragments(a)

    {{0,4},{1},{2},{3}}
    '''
    remainingNodes = list(range(adj.get_shape()[0]))
    currentNode = 0
    connectedFragments = [set()]
    connectionStack = []

    csradj = adj.tocsr(copy = True)

    while currentNode is not None:
        #Indicate that we are currently visiting this node by removing it
        remainingNodes.remove(currentNode)

        #Add to the currentFragment
        currentFragment = connectedFragments[-1]

        #Look for what's connected to this row
        connections = {node for node in csradj[currentNode].indices}

        #Add these connections to our current fragment
        currentFragment.update({currentNode})

        #Visit any nodes that the current node is connected to that still need to be visited
        connectionStack.extend([node for node in connections if (node in remainingNodes and node not in connectionStack)])

        #Get the next node that should be visited
        if len(connectionStack) != 0:
            currentNode = connectionStack.pop()
        elif len(remainingNodes) != 0:
            currentNode = remainingNodes[0]
            connectedFragments.append(set())
        else:
            currentNode = None
    return connectedFragments

def fragmentSize(frags = None):
    '''
    A rigorous way to analyze the size of fragments that have been identified using the findFragments(adj) tool. Makes a dictionary of monomer identities mapped to the length of the fragment that contains them.

    Inputs:
        frags   -- set of sets -- The set of monomer identifier sets that were output from the findFragments code, or the sets of monomers that are connected to each other

    Outputs:
        Dictionary mapping the identity of each monomer [0,N-1] to the length of the fragment that it is found in


    >>> frags = {{0},{1}}
    >>> fragmentSize(frags)

    { 0:1 , 1:1 }

    >>> frags = {{0,4,2},{1,3}}
    >>> fragmentSize(frags)

    { 0:3 , 1:2 , 2:3 , 3:2 , 4:3 }

    >>> frags = {{0,1,2,3,4}}
    >>> fragmentSize(frags)

    { 0:5 , 1:5 , 2:5 , 3:5 , 4:5 }
    '''
    sizes = {}
    for fragment in frags:
        length = len(fragment)
        for node in fragment:
            sizes[node] = length


# noinspection DuplicatedCode
def break_bond_type(adj = None, bondType = None):
    '''
    Function for removing all of a certain type of bond from the adjacency matrix. This is primarily used for the analysis at the end of the simulations when in silico RCF should occur. The update happens via conditional removal of the matching values in the adjacency matrix.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated, and needs certain bonds removed
        bondType-- str          -- the string containing the bond type that should be broken. These are the standard nomenclature, except for b1alt, which removes the previous bond between the beta position and another monomer on the monomer that is bound through 1

    Outputs:
        new dok_matrix adjacency matrix for the connections that remain after the desired bond was broken

    # >>> a = sp.dok_matrix((5,5))
    # >>> a[1,0] = 4; a[0,1] = 8; a[2,3] = 8; a[3,2] = 8
    # >>> break_bond_type(a,'bo4').todense()

    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,8,0],
     [0,0,8,0,0],
     [0,0,0,0,0]]

    # >>> a = sp.dok_matrix( [[0,4,0,0,0],
    #                         [8,0,1,0,0],
    #                         [0,8,0,0,0],
    #                         [0,0,0,0,0],
    #                         [0,0,0,0,0]])
    # >>> break_bond_type(a,'b1alt')

    [[0,0,0,0,0],
     [0,0,1,0,0],
     [0,8,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]]
    '''
    newAdj = adj.todok(1) #Copy the matrix into a new matrix

    breakage = {'b1': (lambda b_row, b_col : (adj[(row,col)] == 1 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and
                                                                                          adj[(col,row)] == 1)) ,
                'b1alt': (lambda b_row, b_col : (adj[(row,col)] == 1 and
                                            adj[(col,row)] == 8) or
                                           (adj[(row,col)] == 8 and
                                            adj[(col,row)] == 1)) ,
                'b5': (lambda b_row, b_col : (adj[(row,col)] == 5 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and
                                                                                          adj[(col,row)] == 5)) ,
                'bo4': (lambda b_row, b_col : (adj[(row,col)] == 4 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and
                                                                                           adj[(col,row)] == 4)) ,
                'ao4': (lambda b_row, b_col : (adj[(row,col)] == 4 and adj[(col,row)] == 7) or (adj[(row,col)] == 7 and
                                                                                           adj[(col,row)] == 4)) ,
                '5o4': (lambda b_row, b_col : (adj[(row,col)] == 4 and adj[(col,row)] == 5) or (adj[(row,col)] == 5 and
                                                                                           adj[(col,row)] == 4)) ,
                'bb': (lambda b_row, b_col: (adj[(row,col)] == 8 and adj[(col,row)] == 8)) ,
                '55': (lambda b_row, b_col : (adj[(row,col)] == 5 and adj[(col,row)] == 5)) }

    for el in adj.keys():
        row = el[0]; col = el[1]

        if breakage[bondType](row,col) and bondType != 'b1alt':
            newAdj[(row,col)] = 0; newAdj[(col,row)] = 0
        elif breakage[bondType](row,col):
            if adj[(row,col)] == 1: #The other 8 is in this row
                data = adj.tocoo().getrow(row).data
                cols = adj.tocoo().getrow(row).indices
                idx = 0
                for i, idx in enumerate(cols):
                    if data[i] == 8:
                        break

                newAdj[(row, idx)] = 0
                newAdj[(idx, row)] = 0
            else: #The other 8 is in the other row
                data = adj.tocoo().getrow(col).data
                cols = adj.tocoo().getrow(col).indices
                idx = 0
                for i,idx in enumerate(cols):
                    if data[i] == 8:
                        break
                newAdj[(col, idx)] = 0
                newAdj[(idx, col)] = 0

    return newAdj

def count_bonds(adj = None):
    '''
    Counter for the different bonds that are present in the adjacency matrix. Primarily used for getting easy analysis of the properties of a simulated lignin from the resulting adjacency matrix.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated

    Outputs:
        dictionary mapping bond strings to the frequency of that specific bond

    >>> a = sp.dok_matrix( [[0,8,0,0,0],
                            [4,0,8,0,0],
                            [0,5,0,8,0],
                            [0,0,8,0,4],
                            [0,0,0,8,0]] )
    >>> count_bonds(a)

    { 'bo4':2 , 'b1' : 0 , 'bb' : 1 , 'b5' : 1 , '55' : 0 , 'ao4' : 0 , '5o4' : 0 }
    '''
    count = {'bo4':0,'b1':0,'bb':0,'b5':0,'55':0,'ao4':0,'5o4':0}
    bonds = {(4,8):'bo4',(8,4):'bo4',(8,1):'b1',(1,8):'b1',(8,8):'bb',(5,5):'55',(8,5):'b5',(5,8):'b5',(7,4):'ao4',
             (4,7):'ao4',(5,4):'5o4',(4,5):'5o4'}

    for el in sp.triu(adj).todok().keys(): #Don't double count by looking only at the upper triangular keys
        row = el[0]; col = el[1]

        bond = (adj[(row,col)],adj[(col,row)])
        count[bonds[bond]] += 1

    return count

def count_oligomer_yields(adj = None):
    '''
    Use the depth first search implemented in findFragments(adj) to locate individual fragments, and then determine the sizes of these individual fragments to obtain estimates of simulated oligomeric yields.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated

    Outputs:
        A Counter dictionary mapping the length of fragments to the number of occurrences of that length

    >>> a = sp.dok_matrix( [ [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> count_oligomer_yields(a)

    { 1 : 5 }

    >>> a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> count_oligomer_yields(a)

    { 2 : 1 , 1 : 3 }

    >>> a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,8,0],
                             [0,0,5,0,0],
                             [0,0,0,0,0] ] )
    >>> count_oligomer_yields(a)

    { 2 : 2 , 1 : 1 }

    >>> a = sp.dok_matrix( [ [0,4,8,0,0],
                             [8,0,0,0,0],
                             [5,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> count_oligomer_yields(a)

    { 3 : 1 , 1 : 2 }
    '''

    #Figure out what is still connected by using the determineCycles function, and look at the length of each connected piece
    oligomers = findFragments(adj = adj)
    counts = Counter(map(len,oligomers))
    return counts

def analyze(adjacency = None):
    '''
    Performs the analysis for a single simulation to extract the relevant macroscopic properties, such as both the simulated frequency of different oligomer sizes and the number of each different type of bond before and after in silico RCF. The specific code to handle each of these properties is written in the count_bonds(adj) and count_oligomer_yields(adj) specifically.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
        nodes   -- list         -- list of monomer objects that have identities matching the indices of the adjacency matrix

    Outputs:
        A dictionary of keywords to the desired result - e.g. Chain Lengths, RCF Yields, Bonds, and RCF Bonds

    # >>> a = sp.dok_matrix( [ [0,0,0,0,0],
    #                          [0,0,0,0,0],
    #                          [0,0,0,0,0],
    #                          [0,0,0,0,0],
    #                          [0,0,0,0,0] ] )
    # >>> analyze(a)

    { 'Chain Lengths' : output from count_oligomer_yields(a) , 'Bonds' : output from count_bonds(a) ,
    'RCF Yields' : output from count_oligomer_yields(a') where a' has bonds broken , 'RCF Bonds' : output from count_bonds(a') }
    '''

    #Remove any excess b1 bonds from the matrix, e.g. bonds that should be
    #broken during synthesis
    adjacency = break_bond_type (adj = adjacency, bondType ='b1alt')

    #Examine the initial polymers before any bonds are broken
    yields = count_oligomer_yields(adj = adjacency)
    bondDistributions = count_bonds(adj = adjacency)

    #Simulate the RCF process at complete conversion by breaking all of the
    #alkyl C-O bonds that were formed during the reaction
    rcfAdj = break_bond_type (adj = break_bond_type (adj =
                                           break_bond_type (adj = adjacency, bondType ='bo4'), bondType ='ao4')
                              , bondType = '5o4')

    #Now count the bonds and yields remaining
    rcfYields = count_oligomer_yields(adj = rcfAdj)
    rcfBonds = count_bonds(adj = rcfAdj)

    return {'Chain Lengths': yields, 'Bonds': bondDistributions, 'RCF Yields': rcfYields,'RCF Bonds': rcfBonds}

# Other methods

def quickFragSize(monomer = None):
    '''
    An easy check on a specific monomer to tell if it is a monomer or involved in a dimer. This is used over the detailed fragmentSize(frags) calculation in the simulation of lignification for performance benefits. However, extensions beyond dimers would be difficult, if it is found that there are significant impacts on chain length > dimer

    Inputs:
        monomer -- Monomer -- the monomer object that we want to know if it is bound to anything else (i.e. if it is truly a monomer still)

    Outputs:
        String either 'monomer' or OLIGOMER if it is connected to nothing else, or isn't respectively

    '''
    if monomer.type == G and monomer.open == {4,5,8}: #Guaiacol monomer
        return 'monomer'
    elif monomer.type == S and monomer.open == {4,8}: #Syringol monomer
        return 'monomer'
    elif monomer.type == C and monomer.open == {4,5,8}:  # Caffeoyl monomer
        return 'monomer'
    return OLIGOMER

def updateEvents(monomers = None, adj = None, lastEvent = None, events=None, rateVec = None, r = None, maxMon = 500):
    '''
    The meat of the implementation for lignification specific KMC. This method determines what the possible events are in a given state, where the state is the current simulation state. Most of the additional parameters in this method are added for performance benefits rather than necessity.

    Inputs:
        monomers    --  dictionary  -- monomers is a dictionary that maps the index of each monomer in the simulation to the monomer itself and the events that would be effected by a change to the monomer key. This makes it easy to quickly determine which of the events in the simulation need to be updated and which should not be changed.
        adj         --  dok_matrix  -- The current state of the simulation represented by the adjacency matrix containing all of the monomers and the bonds between them (if any)
        lastEvent   --  Event       -- The previous Event that occurred, which will tell us what monomers were effected. When combined with the state dictionary, this allows for efficient updating of the set of events that are possible
        events      --  dictionary  -- The set of all possible unique events that must be updated and returned from this method, implemented in a hash map where the event hash value is the key
        rateVec     --  dictionary  -- The rates of all of the unique events implemented in a hash map where the Event hash value is the key
        r           --  dictionary  -- The dictionary of the possible rates involved in each reaction, where the possible reactions are 'ox','b1','5o4','ao4','55','b5','q','ao4', and 'grow'. These are calculated a priori from DFT
        maxMon      --  uint        -- The maximum number of monomers that should be stored in the simulation

    Outputs:
        N/A - mutates the set of events and the rate vector
    '''
    #Map the monomer active state to the possible events it can do
    if events is None:
        events = {}
    possibleEvents = {0:[[OX,1,r[OX]]],
                      4:[['b1',2,r['b1'],[1,8]],
                         ['5o4',2,r['5o4'],[4,5]],
                         ['ao4',2,r['ao4'],[4,7]],
                         ['bo4',2,r['bo4'],[4,8]],
                         ['55',2,r['55'],[5,5]],
                         ['b5',2,r['b5'],[5,8]],
                         ['bb',2,r['bb'],[8,8]]],
                      7:[[Q,1,r[Q]],
                         ['ao4',2,r['ao4'],[7,4]]],
                      -1:[[]]
                      }
    #Only do these for bonding and oxidation events, any growth does not actually change the possible events
    if lastEvent.key != 'grow':
        #Remove the last event that we just did from the set of events that can be performed
        leHash = hash(lastEvent)
        del(events[leHash])
        del(rateVec[leHash])
        
        #Make sure to keep track of which partners have been "cleaned" already - i.e. what monomers have already had all of the old events removed
        cleanedPartners = set() 

        #Get indices of monomers that were acted upon
        affectedMonomers = lastEvent.index

        #Update events from the perspective of each monomer that was just affected individually
        #This prevents having to re-search the entire space every time, saving significant computation
        for monId in affectedMonomers:
            #Get the affected monomer
            mon = monomers[monId][MONOMER]
                    
            #Get the sets of activated monomers that we could bind with
            ox = set(); quinone = set()
            
            otherIDs = [x for x in monomers if x != monId]
            for other in otherIDs:
                #Don't allow connections that would cyclize the polymer!
                if monomers[other][MONOMER].active == 4 and monomers[other][MONOMER].identity not in mon.connectedTo:
                    ox.add(monomers[other][MONOMER])
                elif monomers[other][MONOMER].active == 7 and monomers[other][MONOMER].identity not in mon.connectedTo:
                    quinone.add(monomers[other][MONOMER])
            bondingPartners = {'bo4': ox, 'b5':ox, '5o4':ox,'55':ox, 'bb':ox, 'b1':ox, 'ao4':quinone}

            #Obtain the events that are affected by a change to the monomer that was just acted on
            eventsToBeModified = monomers[monId]['affected']

            #Get the codes for the events that are possible based on how the current monomer behaves
            activePos = mon.active
            newEventList = possibleEvents[activePos]

            #Take any events to be modified out of the set so that we can replace with the updated events
            for event in eventsToBeModified:
                evHash = hash(event)
                if evHash in events:
                    del(events[evHash]) 
                    del(rateVec[evHash])

            #Overwrite the old events that could have been modified from this monomer being updated
            monomers[monId]['affected'] = set()
            curN,_ = adj.get_shape()

            for item in newEventList:
                if item and item[1] == 1: #Unimolecular reaction event
                    size = quickFragSize(monomer = mon)

                    rate = item[2][mon.type][size] / curN
                    
                    #Add the event to the set of events modifiable by changing the monomer, and update the set of all 
                    #events at the next time step
                    monomers[monId]['affected'].add( Event( item[0] , [mon.identity] , rate ) )
                    
                elif item and item[1] == 2: #Bimolecular reaction event
                    bond = tuple(item[3])
                    alt = (bond[1],bond[0])
                    for partner in bondingPartners[item[0]]: 
                        #Sanitize the set of events that can be effected
                        if partner not in cleanedPartners:
                            #Remove any old events from 
                            monomers[partner.identity]['affected'].difference_update(eventsToBeModified)
                            cleanedPartners.add(partner)
                        
                        index = [ mon.identity , partner.identity ]
                        back = [ partner.identity , mon.identity ]
                        
                        #Add the bond from one monomer to the other in the default config
                        size = (quickFragSize(monomer = mon),quickFragSize(monomer = partner))
                        if bond[0] in mon.open and bond[1] in partner.open:
                            try:
                                rate = item[2][(mon.type, partner.type)][size] / (curN ** 2)
                            except KeyError:
                                print(item[0])
                                print((mon.identity,partner.identity))
                                adj.maxprint = adj.nnz
                                print(adj)
                                print(size)
                                raise InvalidDataError("Could not find rate")

                            #Add this to both the monomer and it's bonding partners list of events that need to be modified
                            #upon manipulation of either monomer
                            monomers[monId]['affected'].add ( Event ( item[0] , index , rate , bond ) ) # this -> other
                            monomers[partner.identity]['affected'].add ( Event ( item[0] , index , rate , bond ) ) # this -> other
                            
                            
                            #Switch the order
                            monomers[monId]['affected'].add ( Event ( item[0] , back , rate , alt ) ) #other -> this
                            monomers[partner.identity]['affected'].add ( Event ( item[0] , back , rate , alt ) ) #other -> this
                        
                        
                        #Add the bond from one monomer to the other in the reverse config if not symmetric
                        if item[0] != 'bb' and item[0] != '55': #non-symmetric bond
                            if bond[1] in mon.open and bond[0] in partner.open:
                                #Adjust the rate using the correct monomer types
                                try:
                                    rate = item[2][(partner.type,mon.type)][(size[1],size[0])] / ( curN ** 2 ) 
                                except KeyError:
                                    print(item[0])
                                    print((mon.identity,partner.identity))
                                    adj.maxprint = adj.nnz
                                    print(adj)
                                    print(size)
                                    raise InvalidDataError("Again, could not find rate")
                                
                                monomers[monId]['affected'].add ( Event ( item[0] , index , rate , alt ) ) # this -> other alt
                                monomers[partner.identity]['affected'].add ( Event ( item[0] , index , rate , alt ) ) # this -> other alt
                                
                                #Switch the order
                                monomers[monId]['affected'].add ( Event ( item[0] , back , rate , bond ) ) # other -> this alt
                                monomers[partner.identity]['affected'].add ( Event ( item[0] , back , rate , bond ) ) # other -> this alt
                    
                    #END LOOP OVER PARTNERS
                #END UNIMOLECULAR/BIMOLECULAR BRANCH
            #END LOOP OVER NEW REACTION POSSIBILITIES
            for event in monomers[monId]['affected']:
                evHash = hash(event)
                events[evHash] = event
                rateVec[evHash] = event.rate
        #END LOOP OVER MONOMERS THAT WERE AFFECTED BY LAST EVENT
    else:
        curN,_ = adj.get_shape()

        #If the system has grown to the maximum size, make sure to delete the
        #event for adding more monomers
        if curN >= maxMon:
            leHash = hash(lastEvent)
            del(events[leHash])
            del(rateVec[leHash])


        #Reflect the larger system volume
        for i in rateVec:
            if events[i].key != 'grow':
                rateVec[i] = rateVec[i] * (curN - 1) / curN

        #Add an event to oxidize the monomer that was just added to the 
        #simulation
        oxidation = Event( OX , [curN - 1] , r[OX][monomers[curN - 1][MONOMER].type]['monomer'] )
        monomers[curN - 1]['affected'].add( oxidation )
        evHash = hash( oxidation )
        events[evHash] = oxidation
        rateVec[evHash] = oxidation.rate / curN

def connect(mon1,mon2):
    if mon1.parent == mon1:
        if mon2.parent == mon2:
            parent = mon1 if mon1.identity <= mon2.identity else mon2
            mon1.parent = parent
            mon2.parent = parent
            parent.size = mon1.size + mon2.size
            return parent
        else:
            parent = connect(mon1,mon2.parent)
            mon1.parent = parent
            mon2.parent = parent
            return parent
    else:
        parent = connect(mon1.parent,mon2)
        mon2.parent = parent
        mon1.parent = parent
        return parent

def connectedSize(mon = None):
    if mon == mon.parent:
        return mon.size
    else:
        return connectedSize(mon = mon.parent)

def doEvent(event = None,state = None, adj = None, sg_ratio=None, random_seed=None):
    '''
    The second key component of the lignin implementation of the Monte Carlo algorithm, this method actually executes the chosen event on the current state and modifies it to reflect the updates.

    Inputs:
        event   -- Event      -- The event object that should be executed on the current state
        state   -- dictionary -- The dictionary of dictionaries that contains the state information for each monomer
        adj     -- dok_matrix -- The adjacency matrix in the current state

    Outputs:
        N/A - mutates the list of monomers and adjacency matrix instead
    '''
    indices = event.index
    monomers = [state[i][MONOMER] for i in state]
    if len(indices) == 2: #Doing bimolecular reaction, need to adjust adj

        #Get the tuple of values corresponding to bond and state updates and
        #unpack them
        vals = event.eventDict[event.key]
        stateUpdates = vals[0]
        bondUpdates = event.bond
        order = event.activeDict[bondUpdates]

        #Get the monomers that were being reacted in the correct order
        mon0 = monomers[indices[0]]
        mon1 = monomers[indices[1]]

        connect(mon0,mon1)

        #Make the update to the state and adjacency matrix,
        #Rows are perspective of bonds FROM indices[0] and columns perspective of bonds TO indices[0]
        adj[(indices[0],indices[1])] = bondUpdates[0]
        adj[(indices[1],indices[0])] = bondUpdates[1]
        
        #remove the position that was just active
        mon0.open -= {bondUpdates[0]}
        mon1.open -= {bondUpdates[1]}

        #Update the activated nature of the monomer
        mon0.active = stateUpdates[order[0]]
        mon1.active = stateUpdates[order[1]]
        
        #Add any additional opened positions based on what just reacted
        mon0.open |= set(vals[1+order[0]])
        mon1.open |= set(vals[1+order[1]])

        if mon0.active == 7 and mon1.type == C:
            mon0.active = 0
            mon0.open -= {7}

        if mon1.active == 7 and mon0.type == C:
            mon1.active = 0
            mon1.open -= {7}
        
        #Decided to break bond between alpha and ring position later (i.e. after all synthesis occurred) when a B1 bond is formed
        #This is primarily to make it easier to see what the fragment that needs to break is for visualization purposes
        
        mon0.connectedTo.update(mon1.connectedTo)
        for mon in monomers:
            if mon.identity in mon0.connectedTo:
                mon.connectedTo = mon0.connectedTo
                    
    elif len(indices) == 1:
        if event.key == Q:
            mon = monomers[indices[0]]
            mon.active = 0
            mon.open.remove(7); mon.open.add(1)
        elif event.key == OX:
            mon = monomers[indices[0]]

            #Make the monomer appear oxidized
            mon.active = 4
        else:
            print('Unexpected event')
    else:
        if event.key == 'grow':
            currentSize,_ = adj.get_shape()

            #Add another monomer to the adjacency matrix
            adj.resize((currentSize+1,currentSize+1))

            #Add another monomer to the state
            if monomers and monomers[-1].type == C:
                monType = C
            else:
                pct = sg_ratio / (1 + sg_ratio)
                if random_seed:
                    # to prevent the same choice every iteration, add a changing integer
                    np.random.seed(random_seed + currentSize)
                    rand_num = np.around(np.random.rand(), MAX_NUM_DECIMAL)
                else:
                    rand_num = np.random.rand()
                monType = INT_TO_TYPE_DICT[int(rand_num < pct)]
            newMon = Monomer(monType, currentSize)
            state[currentSize] = {MONOMER:newMon,'affected':set()}


def run_kmc(nMax, tFinal, rates, initialState, initialEvents, dynamics=False, sg_ratio=None,
            random_seed=None):
    '''
    Performs the Gillespie algorithm using the specific event and update implementations described by doEvent and updateEvents specifically. The initial state and events in that state are constructed and passed to the run_kmc method, along with the possible rates of different bond formation events, the maximum number of monomers that should be included in the simulation and the total simulation time.

    Inputs:
        nMax        -- uint         -- The maximum number of monomers in the simulation
        tFinal      -- float        -- The final simulation time (units depend on units of rates)
        rates       -- dictionary   -- The rate of each of the possible events
        initialState-- dictionary   -- The dictionary mapping the index of each monomer to a dictionary with the monomer and the set of events that a change to this monomer would impact
        initialEvents- dictionary   -- The dictionary mapping event hash values to those events

    Outputs:
        Dictionary with the simulation times, adjacency matrix, and list of monomers at the end of the simulation

    Example usage assuming that rates have been defined as dictionary of dictionary of dictionaries:
    rates['event type'][(monomer 1 type, monomer 2 type)][(monomer 1 frag size, monomer 2 frag size)]
    # >>> mons = [Monomer(1, i) for i in range(5)]
    # >>> evs = [Event()]
    # >>> state = {mons[i]:{evs[i]} for i in range(5)}
    # >>> evs.add(Event('grow'))
    # >>> run_kmc(nMax = 5 , tFinal = 10 , rates = rates , initialState = state, initialEvents = set(evs))

    {'time' : , 'monomers' : , 'adjacency_matrix' : }
    '''
    state = copy.deepcopy(initialState)
    events = copy.deepcopy(initialEvents)

    # Current number of monomers
    n = len(state.keys())
    adj = sp.dok_matrix((n,n))
    t = [0, ]

    #Calculate the rates of all of the events available at the current state
    rvec = {}

    #Build the dictionary of events
    eventDict = {}
    hash_list = []
    for event in events:
        event_hash = hash(event)
        if event_hash in hash_list:
            if event.key != GROW:
                raise InvalidDataError("Oh no, there is a duplicate hash!!")
        else:
            hash_list.append(event_hash)
        # todo: figure out why hash value matters--here is where it is used as the keys for a couple dicts
        rvec[event_hash] = event.rate
        eventDict[event_hash] = event

    if dynamics:
        adjList = [adj.copy()]
        monList = [[copy.copy(state[i][MONOMER]) for i in state]]
    else:
        adjList = []
        monList = []

    #Run the Gillespie algorithm
    while t[-1] < tFinal and len(eventDict) > 0:        
        #Find the total rate for all of the possible events and choose which event to do
        hashes = list(rvec.keys())
        allRates = list(rvec.values())
        rtot = np.sum(allRates)

        if random_seed:
            # don't want same dt for every iteration, so add to seed with each iteration
            np.random.seed(random_seed + len(t))
            # don't let machine precision change the dt on different platforms
            rand_num = np.around(np.random.rand(), MAX_NUM_DECIMAL)
        else:
            rand_num = np.random.rand()

        j = np.random.choice(hashes, p=allRates/rtot)
        event = eventDict[j]
        
        #See how much time has passed before this event happened
        # See how much time has passed before this event happened; rounding to reduce platform dependency
        dt = round_sig_figs((1 / rtot) * np.log(1 / rand_num))
        t.append(t[-1] + dt)

        #Do the event and update the state
        doEvent(event,state,adj, sg_ratio=sg_ratio)

        if dynamics:
            adjList.append(adj.copy())
            monList.append([copy.copy(state[i][MONOMER]) for i in state])

        #Check the new state for what events are possible
        updateEvents(monomers = state, adj = adj, lastEvent = event, events = eventDict, rateVec = rvec,
                     r = rates, maxMon = nMax)

    if dynamics:

        return {'time':t,'monomers':monList,'adjacency_matrix':adjList}    
    return {'time':t,'monomers':[state[i][MONOMER] for i in state],'adjacency_matrix':adj}

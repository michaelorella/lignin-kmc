'''
The above code is the traditional use case for running the kmc code, assuming that there are rates for the individual events. Below is the use case for analyzing the results obtained from a single simulation of lignification.

>>> adj = res['adjacency_matrix']
>>> mons = res['monomers']
>>> t = res['time']
>>> analysis = kmc.analyze( adjacency = adj , nodes = mons )

{'Chain Lengths': ______ ,'Bonds': _______ ,'RCF Yields': ________ ,'RCF Bonds': _______}
'''

import scipy.sparse as sp
from collections import Counter

################################################################################
###########################ANALYSIS CODE########################################
################################################################################
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

    >>> a = sp.dok_matrix((5,5))
    >>> a[0,4] = 1; a[4,0] = 1;
    >>> findFragments(a)

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

def breakBond(adj = None, bondType = None):
    '''
    Function for removing all of a certain type of bond from the adjacency matrix. This is primarily used for the analysis at the end of the simulations when in silico RCF should occur. The update happens via conditional removal of the matching values in the adjacency matrix.

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated, and needs certain bonds removed
        bondType-- str          -- the string containing the bond type that should be broken. These are the standard nomenclature, except for b1alt, which removes the previous bond between the beta position and another monomer on the monomer that is bound through 1

    Outputs:
        new dok_matrix adjacency matrix for the connections that remain after the desired bond was broken

    >>> a = sp.dok_matrix((5,5))
    >>> a[1,0] = 4; a[0,1] = 8; a[2,3] = 8; a[3,2] = 8;
    >>> breakBond(a,'bo4').todense()

    [[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,8,0],
     [0,0,8,0,0],
     [0,0,0,0,0]]

    >>> a = sp.dok_matrix( [[0,4,0,0,0],
                            [8,0,1,0,0],
                            [0,8,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]])
    >>> breakBond(a,'b1alt')

    [[0,0,0,0,0],
     [0,0,1,0,0],
     [0,8,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0]]
    '''
    newAdj = adj.todok(1) #Copy the matrix into a new matrix
    
    breakage = {'b1': (lambda row,col : (adj[(row,col)] == 1 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and 
                                                                                         adj[(col,row)] == 1)) ,
                'b1alt': (lambda row,col : (adj[(row,col)] == 1 and
                                            adj[(col,row)] == 8) or
                                            (adj[(row,col)] == 8 and
                                             adj[(col,row)] == 1)) ,
                'b5': (lambda row,col : (adj[(row,col)] == 5 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and 
                                                                                         adj[(col,row)] == 5)) , 
                'bo4': (lambda row,col : (adj[(row,col)] == 4 and adj[(col,row)] == 8) or (adj[(row,col)] == 8 and 
                                                                                         adj[(col,row)] == 4)) , 
                'ao4': (lambda row,col : (adj[(row,col)] == 4 and adj[(col,row)] == 7) or (adj[(row,col)] == 7 and 
                                                                                         adj[(col,row)] == 4)) , 
                '5o4': (lambda row,col : (adj[(row,col)] == 4 and adj[(col,row)] == 5) or (adj[(row,col)] == 5 and 
                                                                                         adj[(col,row)] == 4)) , 
                'bb': (lambda row,col : (adj[(row,col)] == 8 and adj[(col,row)] == 8)) , 
                '55': (lambda row,col : (adj[(row,col)] == 5 and adj[(col,row)] == 5)) }
    
    for el in adj.keys():
        row = el[0]; col = el[1];
        
        if breakage[bondType](row,col) and bondType != 'b1alt':
            newAdj[(row,col)] = 0; newAdj[(col,row)] = 0;
        elif breakage[bondType](row,col):
            if adj[(row,col)] == 1: #The other 8 is in this row
                data = adj.tocoo().getrow(row).data
                cols = adj.tocoo().getrow(row).indices
                for i,idx in enumerate(cols):
                    if data[i] == 8:
                        break

                newAdj[(row,idx)] = 0; newAdj[(idx,row)] = 0;
            else: #The other 8 is in the other row
                data = adj.tocoo().getrow(col).data
                cols = adj.tocoo().getrow(col).indices
                for i,idx in enumerate(cols):
                    if data[i] == 8:
                        break
                newAdj[(col,idx)] = 0; newAdj[(idx,col)] = 0;
            
    
    return newAdj

def countBonds(adj = None):
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
    >>> countBonds(a)

    { 'bo4':2 , 'b1' : 0 , 'bb' : 1 , 'b5' : 1 , '55' : 0 , 'ao4' : 0 , '5o4' : 0 }
    '''
    count = {'bo4':0,'b1':0,'bb':0,'b5':0,'55':0,'ao4':0,'5o4':0}
    bonds = {(4,8):'bo4',(8,4):'bo4',(8,1):'b1',(1,8):'b1',(8,8):'bb',(5,5):'55',(8,5):'b5',(5,8):'b5',(7,4):'ao4',
             (4,7):'ao4',(5,4):'5o4',(4,5):'5o4'}
    
    for el in sp.triu(adj).todok().keys(): #Don't double count by looking only at the upper triangular keys
        row = el[0]; col = el[1];
        
        bond = (adj[(row,col)],adj[(col,row)])
        count[bonds[bond]] += 1
        
    return count

def countYields(adj = None):
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
    >>> countYields(a)

    { 1 : 5 }

    >>> a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> countYields(a)

    { 2 : 1 , 1 : 3 }

    >>> a = sp.dok_matrix( [ [0,4,0,0,0],
                             [8,0,0,0,0],
                             [0,0,0,8,0],
                             [0,0,5,0,0],
                             [0,0,0,0,0] ] )
    >>> countYields(a)

    { 2 : 2 , 1 : 1 }

    >>> a = sp.dok_matrix( [ [0,4,8,0,0],
                             [8,0,0,0,0],
                             [5,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> countYields(a)

    { 3 : 1 , 1 : 2 }
    '''

    #Figure out what is still connected by using the determineCycles function, and look at the length of each connected piece
    oligomers = findFragments(adj = adj)
    counts = Counter(map(len,oligomers))
    return counts

def analyze(adjacency = None, nodes = None):
    '''
    Performs the analysis for a single simulation to extract the relevant macroscopic properties, such as both the simulated frequency of different oligomer sizes and the number of each different type of bond before and after in silico RCF. The specific code to handle each of these properties is written in the countBonds(adj) and countYields(adj) specifically. 

    Inputs:
        adj     -- DOK_MATRIX   -- the adjacency matrix for the lignin polymer that has been simulated
        nodes   -- list         -- list of monomer objects that have identities matching the indices of the adjacency matrix

    Outputs:
        A dictionary of keywords to the desired result - e.g. Chain Lengths, RCF Yields, Bonds, and RCF Bonds

    >>> a = sp.dok_matrix( [ [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0] ] )
    >>> analyze(a)

    { 'Chain Lengths' : output from countYields(a) , 'Bonds' : output from countBonds(a) , 'RCF Yields' : output from countYields(a') where a' has bonds broken , 'RCF Bonds' : output from countBonds(a') }
    '''

    #Remove any excess b1 bonds from the matrix, e.g. bonds that should be
    #broken during synthesis
    adjacency = breakBond ( adj = adjacency , bondType = 'b1alt' )
    
    #Examine the initial polymers before any bonds are broken
    yields = countYields(adj = adjacency)
    bondDistributions = countBonds(adj = adjacency)

    #Simulate the RCF process at complete conversion by breaking all of the
    #alkyl C-O bonds that were formed during the reaction
    rcfAdj = breakBond ( adj = breakBond ( adj = 
        breakBond ( adj = adjacency , bondType = 'bo4' ) , bondType = 'ao4' )
         , bondType = '5o4' ) 

    #Now count the bonds and yields remaining
    rcfYields = countYields(adj = rcfAdj)
    rcfBonds = countBonds(adj = rcfAdj)

    return {'Chain Lengths':yields,'Bonds':bondDistributions,
            'RCF Yields':rcfYields,'RCF Bonds':rcfBonds}
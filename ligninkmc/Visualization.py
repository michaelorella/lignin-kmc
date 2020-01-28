from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
import itertools
import re

from ligninkmc.KineticMonteCarlo import C, S, G

DrawingOptions.bondLineWidth = 1.2


def generateMol(adj,nodeList): 
    #define dictionary for atoms within each monomer
    atomBlocks = {G:( 'C 0 0 0 0 \n' + #1
                        'C 0 0 0 0 \n' + #2
                        'C 0 0 0 0 \n' + #3
                        'C 0 0 0 0 \n' + #4
                        'C 0 0 0 0 \n' + #5
                        'C 0 0 0 0 \n' + #6
                        'C 0 0 0 0 \n' + #7
                        'C 0 0 0 0 \n' + #8
                        'C 0 0 0 0 \n' + #9
                        'O 0 0 0 0 \n' + #9-OH
                        'O 0 0 0 0 \n' + #3-OMe
                        'C 0 0 0 0 \n' + #3-OMe
                        'O 0 0 0 0 \n'), #4-OH
                 S: ( 'C 0 0 0 0 \n' + #1
                        'C 0 0 0 0 \n' + #2
                        'C 0 0 0 0 \n' + #3
                        'C 0 0 0 0 \n' + #4
                        'C 0 0 0 0 \n' + #5
                        'C 0 0 0 0 \n' + #6
                        'C 0 0 0 0 \n' + #7
                        'C 0 0 0 0 \n' + #8
                        'C 0 0 0 0 \n' + #9
                        'O 0 0 0 0 \n' + #9-OH
                        'O 0 0 0 0 \n' + #3-OMe
                        'C 0 0 0 0 \n' + #3-OMe
                        'O 0 0 0 0 \n' + #4-OH
                        'O 0 0 0 0 \n' + #5-OMe
                        'C 0 0 0 0 \n'),  #5-OMe
                 C: ( 'C 0 0 0 0 \n' + #1
                        'C 0 0 0 0 \n' + #2
                        'C 0 0 0 0 \n' + #3
                        'C 0 0 0 0 \n' + #4
                        'C 0 0 0 0 \n' + #5
                        'C 0 0 0 0 \n' + #6
                        'C 0 0 0 0 \n' + #7
                        'C 0 0 0 0 \n' + #8
                        'C 0 0 0 0 \n' + #9
                        'O 0 0 0 0 \n' + #9-OH
                        'O 0 0 0 0 \n' + #3-OH
                        'O 0 0 0 0 \n'), #4-OH
                 'G4':( 'C 0 0 0 0 \n' + #1
                        'C 0 0 0 0 \n' + #2
                        'C 0 0 0 0 \n' + #3
                        'C 0 0 0 0 \n' + #4
                        'C 0 0 0 0 \n' + #5
                        'C 0 0 0 0 \n' + #6
                        'C 0 0 0 0 \n' + #7
                        'C 0 0 0 0 \n' + #8
                        'C 0 0 0 0 \n' + #9
                        'O 0 0 0 0 \n' + #9-OH
                        'O 0 0 0 0 \n' + #3-OMe
                        'C 0 0 0 0 \n' + #3-OMe
                        'O 0 0 0 0 RAD=2\n'), #4-O
                 'S4':( 'C 0 0 0 0 \n' + #1
                        'C 0 0 0 0 \n' + #2
                        'C 0 0 0 0 \n' + #3
                        'C 0 0 0 0 \n' + #4
                        'C 0 0 0 0 \n' + #5
                        'C 0 0 0 0 \n' + #6
                        'C 0 0 0 0 \n' + #7
                        'C 0 0 0 0 \n' + #8
                        'C 0 0 0 0 \n' + #9
                        'O 0 0 0 0 \n' + #9-OH
                        'O 0 0 0 0 \n' + #3-OMe
                        'C 0 0 0 0 \n' + #3-OMe
                        'O 0 0 0 0 RAD=2\n' + #4-O
                        'O 0 0 0 0 \n' + #5-OMe
                        'C 0 0 0 0 \n')} #5-OMe

    #Similarly define dictionary for bonds within each monomer - 
    #NOTE: THESE MAY NEED TO CHANGE DEPENDING ON INTERUNIT LINKAGES
    bondBlocks = {'G7':('1 1  2  \n' +  #Aromatic ring 1->2
                        '2 2  3  \n' +  #Aromatic ring 2->3
                        '1 3  4  \n' +  #Aromatic ring 3->4
                        '1 4  5  \n' +  #Aromatic ring 4->5
                        '2 5  6  \n' +  #Aromatic ring 5->6
                        '1 6  1  \n' +  #Aromatic ring 6->1
                        '2 1  7  \n' +  #Quinone methide propyl tail 1->A
                        '1 7  8  \n' +  #Propyl tail A->B
                        '1 8  9  \n' +  #Propyl tail B->G
                        '1 9  10 \n' + #Gamma hydroxyl G->OH
                        '1 3  11 \n' + #3 methoxy 3->O
                        '1 11 12 \n' + #3 methoxy O->12
                        '2 4  13 \n' ),#4 ketone 4->O
                  G:( '2 1  2  \n' +  #Aromatic ring 1->2
                        '1 2  3  \n' +  #Aromatic ring 2->3
                        '2 3  4  \n' +  #Aromatic ring 3->4
                        '1 4  5  \n' +  #Aromatic ring 4->5
                        '2 5  6  \n' +  #Aromatic ring 5->6
                        '1 6  1  \n' +  #Aromatic ring 6->1
                        '1 1  7  \n' +  #Ring - propyl tail 1->A
                        '2 7  8  \n' +  #Alkene propyl tail A->B
                        '1 8  9  \n' +  #Propyl tail B->G
                        '1 9  10 \n' + #Gamma hydroxyl G->OH
                        '1 3  11 \n' + #3 methoxy 3->O
                        '1 11 12 \n' + #3 methoxy O->12
                        '1 4  13 \n' ),#4 hydroxyl 4->OH
                  'S7':('1 1  2  \n' +  #Aromatic ring 1->2
                        '2 2  3  \n' +  #Aromatic ring 2->3
                        '1 3  4  \n' +  #Aromatic ring 3->4
                        '1 4  5  \n' +  #Aromatic ring 4->5
                        '2 5  6  \n' +  #Aromatic ring 5->6
                        '1 6  1  \n' +  #Aromatic ring 6->1
                        '2 1  7  \n' +  #Quinone methide 1->A
                        '1 7  8  \n' +  #Propyl tail A->B
                        '1 8  9  \n' +  #Propyl tail B->G
                        '1 9  10 \n' + #Gamma hydroxyl G->OH
                        '1 3  11 \n' + #3 methoxy 3->O
                        '1 11 12 \n' + #3 methoxy O->12
                        '2 4  13 \n' + #4 ketone 4->O
                        '1 5  14 \n' + #5 methoxy 5->O
                        '1 14 15 \n' ),#5 methoxy O->15
                  S:( '2 1  2  \n' +  #Aromatic ring 1->2
                        '1 2  3  \n' +  #Aromatic ring 2->3
                        '2 3  4  \n' +  #Aromatic ring 3->4
                        '1 4  5  \n' +  #Aromatic ring 4->5
                        '2 5  6  \n' +  #Aromatic ring 5->6
                        '1 6  1  \n' +  #Aromatic ring 6->1
                        '1 1  7  \n' +  #Ring - propyl tail 1->A
                        '2 7  8  \n' +  #Alkene propyl tail A->B
                        '1 8  9  \n' +  #Propyl tail B->G
                        '1 9  10 \n' + #Gamma hydroxyl G->OH
                        '1 3  11 \n' + #3 methoxy 3->O
                        '1 11 12 \n' + #3 methoxy O->12
                        '1 4  13 \n' + #4 hydroxyl 4->OH
                        '1 5  14 \n' + #5 methoxy 5->O
                        '1 14 15 \n' ),#5 methoxy O->15
                  C:( '2 1  2  \n' +  #Aromatic ring 1->2
                        '1 2  3  \n' +  #Aromatic ring 2->3
                        '2 3  4  \n' +  #Aromatic ring 3->4
                        '1 4  5  \n' +  #Aromatic ring 4->5
                        '2 5  6  \n' +  #Aromatic ring 5->6
                        '1 6  1  \n' +  #Aromatic ring 6->1
                        '1 1  7  \n' +  #Ring - propyl tail 1->A
                        '2 7  8  \n' +  #Alkene propyl tail A->B
                        '1 8  9  \n' +  #Propyl tail B->G
                        '1 9  10 \n' + #Gamma hydroxyl G->OH
                        '1 3  11 \n' + #3 hydroxyl 3->O
                        '1 4  12 \n')} #4 hydroxyl 4->OH
    
    molStr = '\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\nM  V30 BEGIN CTAB\n' #Header information
    molAtomBlocks = 'M  V30 BEGIN ATOM\n'
    molBondBlocks = 'M  V30 BEGIN BOND\n'
    atomlineno = 1
    bondlineno = 1
    monomerStartIdxBond = []
    monomerStartIdxAtom = []
    removed = {'bonds':0,'atoms':0}

    sitePositions = {1: {x: 0 for x in [G, S, C]}, 4: {C: 11, S: 12, G: 12}, 5: {x: 4 for x in [G, S, C]},
                     7: {x: 6 for x in [G, S, C]}, 8: {x: 7 for x in [G, S, C]}, 10: {x: 9 for x in [G, S, C]}}
    ALPHA_BETA_ALKENE_LOCATION = 7
    ALPHA_RING_LOCATION = 6
    ALPHA = 7
    BETA = 8
    
    #Build the individual monomers before they are linked by anything
    for i , mon in enumerate(nodeList):
        if mon.type == G:
            if mon.active == 0 or mon.active == -1:
                atomBlock = atomBlocks[G]
                bondBlock = bondBlocks[G]
            elif mon.active == 4:
                atomBlock = atomBlocks['G4']
                bondBlock = bondBlocks[G]
            elif mon.active == 7:
                atomBlock = atomBlocks[G]
                bondBlock = bondBlocks['G7']
        elif mon.type == S:
            if mon.active == 0 or mon.active == -1:
                atomBlock = atomBlocks[S]
                bondBlock = bondBlocks[S]
            elif mon.active == 4:
                atomBlock = atomBlocks['S4']
                bondBlock = bondBlocks[S]
            elif mon.active == 7:
                atomBlock = atomBlocks[S]
                bondBlock = bondBlocks['S7']
        elif mon.type == C:
            atomBlock = atomBlocks[C]
            bondBlock = bondBlocks[C]
        
        
        #Extract each of the individual atoms from this monomer to add to the aggregate file
        lines = atomBlock.splitlines(keepends=True)
        
        #Figure out what atom and bond number this monomer is starting at
        monomerStartIdxBond.append(bondlineno)
        monomerStartIdxAtom.append(atomlineno)

        #Loop through the lines of the atom block and add the necessary prefixes to the lines, using a continuing atom index
        for line in lines:
            molAtomBlocks += f'M  V30 {atomlineno} {line}'
            atomlineno += 1
        #END ATOM AGGREGATION

        #Extract each of the individual bonds that defines the monomer skeleton and add it into the aggregate string
        lines = bondBlock.splitlines(keepends=True)

        #Recall where this monomer started
        startindex = monomerStartIdxAtom[-1] - 1 #So that we can add the defined bond indices to this start index to get the bond defs

        #Loop through the lines of the bond block and add necessary prefixes to the lines, then modify as needed after
        for line in lines:
            #Extract the definining information about the monomer skeleton
            bondVals = re.split(' +',line)

            #The first element is the bond order, followed by the indices of the atoms that are connected by this bond.
            #These indices need to be updated based on the true index of this monomer, not the 1 -> ~15 indices that it started with
            bondOrder = bondVals[0]
            bondConnects = [ int ( bondVals[1] ) + startindex , int ( bondVals[2] ) + startindex ]

            #Now save the true string for defining this bond, along with the cumulative index of the bond
            molBondBlocks += f'M  V30 {bondlineno} {bondOrder} {bondConnects[0]} {bondConnects[1]} \n'
            bondlineno += 1
        #END BOND AGGREGATION
    #END MONOMER AGGREGATION

    #Now that we have all of the monomers in one string, we just need to add the bonds and atoms that are defined by the adjacency matrix
    bonds = molBondBlocks.splitlines(keepends = True)
    atoms = molAtomBlocks.splitlines(keepends = True)

    breakAlkene = {(4,8):True,(5,8):True,(8,8):True,(5,5):False,(1,8):True,(4,7):False,(4,5):False}
    hydrate = {(4,8):True,(5,8):False,(8,8):False,(5,5):False,(1,8):True,(4,7):False,(4,5):False}
    beta = {(4,8):1,(8,4):0,(5,8):1,(8,5):0,(1,8):1,(8,1):0}
    makeAlphaRing = {(4,8):False,(5,8):True,(8,8):True,(5,5):False,(1,8):False,(4,7):False,(4,5):False}

    #Start by looping through the adjacency matrix, one bond at a time (corresponds to a pair iterator)
    pairedAdj = zip ( * [ iter(dict(adj)) ] * 2 )

    for pair in pairedAdj:
        #Find the types of bonds and indices associated with each of the elements in the adjacency matrix
        #Indices are extracted as tuples (row,col) and we just want the row for each
        monomerIndices = [x[0] for x in pair]

        #Get the monomers corresponding to the indices in this bond
        mons = [None,None]

        for mon in nodeList:
            if mon.identity == monomerIndices[0]:
                mons[0] = mon
            elif mon.identity == monomerIndices[1]:
                mons[1] = mon

        #Now just extract the bond types
        bond = [adj[p] for p in pair]

        #Get the indices of the atoms being bound -> Count from where the monomer starts, and add however many is needed to reach the desired position for that specific monomer type and bonding site
        atomindices = [ monomerStartIdxAtom [ monomerIndices [ i ] ] + sitePositions [ bond [ i ] ] [ mons [ i ].type ] for i in range(2) ]

        #Make the string to add to the molfile
        bondstring = f'M  V30 {bondlineno} 1 {atomindices[0]} {atomindices[1]} \n'
        bondlineno += 1
        
        #Append the newly created bond to the file
        bonds.append(bondstring)

        #Check if the alkene needs to be modified to a single bond
        if breakAlkene[ tuple ( sorted(bond) ) ]:
            for index in range(2):
                if adj[pair[index]] == 8 and mons[index].active != 7: #Monomer index is bound through beta
                    #Find the bond index corresponding to alkene bond
                    alkeneBondIndex = monomerStartIdxBond [ monomerIndices [ index ] ] + ALPHA_BETA_ALKENE_LOCATION - removed['bonds']

                    #Get all of the required bond information (index,order,monIdx1,monIdx2)
                    bondVals = re.split(' +',bonds[alkeneBondIndex])[2:]
                    try:
                        assert( int ( bondVals[0] ) == alkeneBondIndex + removed['bonds'])
                    except AssertionError:
                        print(f'Expected index: {bondVals[0]} Index obtained: {alkeneBondIndex}')

                    #Decrease the bond order by 1
                    bonds[alkeneBondIndex] = f'M  V30 {bondVals[0]} 1 {bondVals[2]} {bondVals[3]} \n'
        
        #Check if we need to add water to the alpha position

        if hydrate[ tuple ( sorted(bond) ) ] and 7 not in adj [ monomerIndices [ beta [ tuple (bond) ] ] ].values() and mons [ beta [ tuple(bond) ] ].active != 7 :
            if mons [ int ( not beta [ tuple(bond) ] ) ].type != 2 :
                #We should actually only be hydrating BO4 bonds when the alpha position is unoccupied (handled by second clause above)
                
                #Find the location of the alpha position
                alphaIndex = monomerStartIdxAtom [ monomerIndices [ beta [ tuple(bond) ] ] ] + sitePositions [ 7 ] [ mons [ beta [ tuple(bond) ] ].type ]

                #Add the alpha hydroxyl O atom
                atoms.append(f'M  V30 {atomlineno} O 0 0 0 0 \n')
                atomlineno += 1

                #Make the bond
                bonds.append(f'M  V30 {bondlineno} 1 {alphaIndex} {atomlineno-1} \n')
                bondlineno += 1
            else:
                #Make the benzodioxane linkage
                alphaIndex = monomerStartIdxAtom [ monomerIndices [ beta [ tuple(bond) ] ] ] + sitePositions [ 7 ] [ mons [ beta [ tuple(bond) ] ].type ]
                hydroxyIndex = monomerStartIdxAtom [ monomerIndices [ int ( not beta [ tuple (bond) ] ) ] ] + ( sitePositions [ 4 ] [ mons [ beta [ tuple(bond) ] ].type ] - 1 ) #subtract 1 to move from 4-OH to 3-OH

                #Make the bond
                bonds.append(f'M  V30 {bondlineno} 1 {alphaIndex} {hydroxyIndex} \n')
                bondlineno += 1

        #Check if there will be a ring involving the alpha position
        if makeAlphaRing [ tuple ( sorted(bond) ) ]:
            otherSite = {(5,8):4, (8,8): 10}
            for index in range(2):
                if adj[ pair[index] ] == 8: #This index is bound through beta (will get alpha connection)
                    #Find the location of the alpha position and the position that cyclizes with alpha
                    alphaIndex = monomerStartIdxAtom [ monomerIndices [ index ] ] + sitePositions [ 7 ] [ mons [ index ].type ]
                    otherIndex = monomerStartIdxAtom [ monomerIndices [ int (not index) ] ] + sitePositions [ otherSite [ tuple ( sorted ( bond ) ) ] ] [ mons [ int ( not index ) ].type ]

                    bonds.append(f'M  V30 {bondlineno} 1 {alphaIndex} {otherIndex} \n')
                    bondlineno += 1

        # All kinds of fun things need to happen for the B1 bond --
        # 1 ) Disconnect the original 1 -> A bond that existed from the not beta monomer
        # 2 ) Convert the new primary alcohol to an aldehyde
        if sorted(bond) == [1,8]:
            indexForOne = int ( not beta [ tuple ( bond ) ] )
            #Convert the alpha alcohol on one's tail to an aldehyde
            alphaIndex = monomerStartIdxAtom [ monomerIndices [ indexForOne ] ] + sitePositions [ ALPHA ] [ mons [ indexForOne ].type ]

            #Temporarily join the bonds so that we can find the string
            temp = ''.join(bonds)
            matches = re.findall(f'M  V30 [0-9]+ 1 {alphaIndex} [0-9]+',temp)

            #Find the bond connecting the alpha to the alcohol
            others = []
            for possibility in matches:
                boundAtoms = re.split(' +',possibility)[4:]
                others.extend([int(x) for x in boundAtoms if int(x) != alphaIndex])

            #The oxygen atom should have the greatest index of the atoms bound to the alpha position because it was added last
            oxygenAtomIndex = max(others)
            bonds = re.sub(f'1 {alphaIndex} {oxygenAtomIndex}',f'2 {alphaIndex} {oxygenAtomIndex}',temp).splitlines(keepends=True)

            #Find where the index for the bond is and remove it from the array
            alphaRingBondIndex = monomerStartIdxBond [ monomerIndices [ indexForOne ] ] + ALPHA_RING_LOCATION - removed['bonds']
            del(bonds[alphaRingBondIndex])
            removed['bonds'] += 1

    molBondBlocks = ''.join(bonds)
    molAtomBlocks = ''.join(atoms)

    molAtomBlocks += 'M  V30 END ATOM \n'
    molBondBlocks += 'M  V30 END BOND \n'
    counts = f'M  V30 COUNTS {atomlineno-1 - removed["atoms"]} {bondlineno-1-removed["bonds"]} 0 0 0\n'
    molStr += counts + molAtomBlocks + molBondBlocks + 'M  V30 END CTAB\nM  END'
    
    return molStr


def moltosvg(mol,molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])

    opts = drawer.drawOptions()
    #for i in range(mc.GetNumAtoms()):
    #    opts.atomLabels[i] = mc.GetAtomWithIdx(i).GetSymbol()+str(i+1)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

#!/usr/bin/env python
# coding: utf-8

# # Kinetic Monte Carlo Simulation of Lignin Polymerization
# Written by: Michael Orella <br>
# 16 April 2019 <br>
# 
# The code in this notebook performs calculations that are used in [citation here], which depends on the results that were obtained from [DFT Calculations of monolignol coupling kinetics](https://chemrxiv.org/articles/Quantum_Mechanical_Calculations_Suggest_That_Lignin_Polymerization_Is_Kinetically_Controlled/7334564).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


#LIGNIN-KMC Package
import ligninkmc as kmc

#General Math and LinAlg
import numpy as np
import scipy as sp
import scipy.optimize

#Chemical Drawing
from rdkit import Chem
from rdkit.Chem import AllChem
from IPython.display import SVG

#Plotting
import matplotlib.pyplot as plt

#Profiling and Performance
import cProfile
import time

#Parallelization
import joblib as par

#Serialization
import pickle


# ## Input Data
# The kinetic monte carlo code relies on rates of chemical reactions. The reactions that we are interested in here are the coupling of radicals on either individual monomers or oligomers respectively. The energetics necessary to compute the reaction rates were performed by Terry Gani using [DFT methods](https://chemrxiv.org/articles/Quantum_Mechanical_Calculations_Suggest_That_Lignin_Polymerization_Is_Kinetically_Controlled/7334564) for SG lignin and similar methods for C lignin. Once the reaction energies were calculated, they could be inputted using the Erying equation to obtain the actual rate.
# $$ r_i = \dfrac{k_BT}{h}\exp\left(-\dfrac{\Delta G_i}{k_BT}\right) $$
# 
# Throughout the code, monomers are kept track of individually through the state, so there are no reaction degeneracies occuring, and bond events can be tracked faithfully.

# In[3]:


kb = 1.38064852e-23 # J / K
h = 6.62607004e-34 # J s
temp = 298.15 #K
kcalToJoule = 4184 / 6.022140857e23

#Input energy information
#Select which energy set you want to use
energies = {'5o4':{(0,0):{('monomer','monomer'):11.2,('monomer','dimer'):14.6,
                          ('dimer','monomer'):14.6,('dimer','dimer'):4.4},
                   (1,0):{('monomer','monomer'):10.9,('monomer','dimer'):14.6,
                          ('dimer','monomer'):14.6,('dimer','dimer'):4.4}},
            '55':{(0,0):{('monomer','monomer'):12.5,('monomer','dimer'):15.6,
                          ('dimer','monomer'):15.6,('dimer','dimer'):3.8}},
            'b5':{(0,0):{('monomer','monomer'):5.5,('monomer','dimer'):5.8,
                          ('dimer','monomer'):5.8,('dimer','dimer'):5.8},
                  (0,1):{('monomer','monomer'):5.5,('monomer','dimer'):5.8,
                          ('dimer','monomer'):5.8,('dimer','dimer'):5.8}},
            'bb':{(0,0):{('monomer','monomer'):5.2,('monomer','dimer'):5.2,('dimer','monomer'):5.2,('dimer','dimer'):5.2},
                  (1,0):{('monomer','monomer'):6.5,('monomer','dimer'):6.5,('dimer','monomer'):6.5,('dimer','dimer'):6.5},
                  (1,1):{('monomer','monomer'):5.2,('monomer','dimer'):5.2,('dimer','monomer'):5.2,('dimer','dimer'):5.2}},
            'bo4':{(0,0):{('monomer','monomer'):6.3,('monomer','dimer'):6.2,
                          ('dimer','monomer'):6.2,('dimer','dimer'):6.2},
                   (1,0):{('monomer','monomer'):9.1,('monomer','dimer'):6.2,
                          ('dimer','monomer'):6.2,('dimer','dimer'):6.2},
                   (0,1):{('monomer','monomer'):8.9,('monomer','dimer'):6.2,
                          ('dimer','monomer'):6.2,('dimer','dimer'):6.2},
                   (1,1):{('monomer','monomer'):9.8,('monomer','dimer'):10.4,
                          ('dimer','monomer'):10.4}},
            'ao4':{(0,0):{('monomer','monomer'):20.7,('monomer','dimer'):20.7,
                          ('dimer','monomer'):20.7,('dimer','dimer'):20.7},
                   (1,0):{('monomer','monomer'):20.7,('monomer','dimer'):20.7,
                          ('dimer','monomer'):20.7,('dimer','dimer'):20.7},
                   (0,1):{('monomer','monomer'):20.7,('monomer','dimer'):20.7,
                          ('dimer','monomer'):20.7,('dimer','dimer'):20.7},
                   (1,1):{('monomer','monomer'):20.7,('monomer','dimer'):20.7,
                          ('dimer','monomer'):20.7,('dimer','dimer'):20.7}},
            'b1':{(0,0):{('monomer','dimer'):9.6,
                          ('dimer','monomer'):9.6,('dimer','dimer'):9.6},
                  (1,0):{('monomer','dimer'):11.7,
                          ('dimer','monomer'):11.7,('dimer','dimer'):11.7},
                  (0,1):{('monomer','dimer'):10.7,
                          ('dimer','monomer'):10.7,('dimer','dimer'):10.7},
                  (1,1):{('monomer','dimer'):11.9,
                          ('dimer','monomer'):11.9,('dimer','dimer'):11.9}},
            'ox':{0:{'monomer':0.9,'dimer':6.3},1:{'monomer':0.6,'dimer':2.2}},
            'q':{0:{'monomer':11.1,'dimer':11.1},1:{'monomer':11.7,'dimer':11.7}}}
energies['bb'][(0,1)] = energies['bb'][(1,0)]

#Correct units of the energies
energiesev = {bond : {monType : {size : energies[bond][monType][size] * kcalToJoule for size in energies[bond][monType]}
                    for monType in energies[bond] } 
            for bond in energies }

#Calculate the rates of reaction
rates = {bond : {monType : { size : kb * temp / h * np.exp ( - energiesev[bond][monType][size] / kb / temp ) 
                            for size in energies[bond][monType]} 
                 for monType in energies[bond] } #Make rates 1/ns instead of 1/s
         for bond in energies}


# In[4]:


cLigEnergies = {'5o4':{(2,2):{('monomer','monomer'):11.9,('monomer','dimer'):11.9,
                          ('dimer','monomer'):11.9,('dimer','dimer'):11.9}},
            '55':{(2,2):{('monomer','monomer'):10.6,('monomer','dimer'):10.6,
                          ('dimer','monomer'):10.6,('dimer','dimer'):10.6}},
            'b5':{(2,2):{('monomer','monomer'):1.9,('monomer','dimer'):5.8,
                          ('dimer','monomer'):5.8,('dimer','dimer'):5.8}},
            'bb':{(2,2):{('monomer','monomer'):7.2,('monomer','dimer'):7.2,
                         ('dimer','monomer'):7.2,('dimer','dimer'):7.2}},
            'bo4':{(2,2):{('monomer','monomer'):4.9,('monomer','dimer'):1.3,
                          ('dimer','monomer'):1.3,('dimer','dimer'):1.3}},
            'ao4':{(2,2):{('monomer','monomer'):20.7,('monomer','dimer'):20.7,
                          ('dimer','monomer'):20.7,('dimer','dimer'):20.7}},
            'b1':{(2,2):{('monomer','dimer'):9.6,
                          ('dimer','monomer'):9.6,('dimer','dimer'):9.6}},
            'ox':{2:{'monomer':0.9,'dimer':0.9}},
            'q':{2:{'monomer':11.1,'dimer':11.1}}}

#Correct units of the energies
cLigEnergiesev = {bond : {monType : {size : cLigEnergies[bond][monType][size] * kcalToJoule for size in cLigEnergies[bond][monType]}
                    for monType in cLigEnergies[bond] }
            for bond in cLigEnergies }

#Calculate the rates of reaction
cRates = {bond : {monType : { size : kb * temp / h * np.exp ( - cLigEnergiesev[bond][monType][size] / kb / temp ) 
                            for size in cLigEnergies[bond][monType] } 
                 for monType in cLigEnergies[bond] }
         for bond in cLigEnergies}


# ## Sensitivity Analyses Examples
# The meat of the results and discussion for our paper lay in the predictions of how lignin composition should change with different sets of parameters used for lignification. These calculations were performed on desktop hardware over about a week's period, but for the sake of explanation, shorter examples are used here. We investigated the impact of S to G ratio and addition rate primarily.

# ### SG Sensitivity
# The first analysis performed is the dependence of monomer yields and bond contents on SG ratio and addition rate, where we selected multiple SG ratios between 0.1 and 10 and ran the simulations for these scenarios.

# In[6]:


sg_opts = [0.1,0.2,0.25,0.33,0.5,1,2,3,4,5,10]

fun = par.delayed(kmc.run)

for additionRate in np.logspace(0,14,15):
    resultsToSave = []

    for sg in sg_opts:
        #Set the percentage of S
        pct = sg / (1 + sg)

        #Make choices about what kinds of monomers there are
        n = 2
        rands = np.random.rand(n)

        #Initialize the monomers, events, and state
        mons = [ kmc.Monomer( int ( sOrG < pct ) , i ) for i,sOrG in zip ( range(n) , rands ) ]
        startEvents = [ kmc.Event ( 'ox' , [i] , rates['ox'][ int( sOrG < pct ) ]['monomer'] ) for i,sOrG in zip ( range(n) , rands) ]

        state = { i : {'mon' : mons[i] , 'affected' : {startEvents[i]} } for i in range(n) }
        events = { startEvents[i] for i in range(n) }
        events.add(kmc.Event('grow',[],rate = additionRate,bond = sg))

        results = par.Parallel(n_jobs = 16,verbose = 5)([fun(nMax = 500, tFinal = 1e5,rates = rates,initialState = state,initialEvents = events)
                                            for _ in range(100)])
        resultsToSave.append(results)
        print('Completed sensitivity iteration: ' + str(sg))

        t = time.localtime()
        print('Finished on ' + str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) 
                + ' at ' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec))

    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\2start\\{additionRate:0.0e}.pkl','wb+') as file:
        pickle.dump(resultsToSave,file)


# In[11]:


for additionRate in np.logspace(0,14,15):
    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\2start\\{additionRate:0.0e}.pkl','rb') as file:
        raw = pickle.load(file)
    
    results = [0]*len(raw)
    for i in range(len(raw)):
        results[i] = par.Parallel(n_jobs = 8)( [ par.delayed(kmc.analyze)(adjacency = raw[i][j]['adjacency_matrix'], 
                                                                          nodes = raw[i][j]['monomers'] ) 
                                                 for j in range( len ( raw[i] ) ) ] )
        
        t = time.localtime()
        
        print('Finished on ' + str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) 
                + ' at ' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec))

        
    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\2start\\{additionRate:0.0e}_analysis.pkl','wb+') as file:
        pickle.dump(results,file)


# ### Start Number Sensitivity
# The first analysis performed is the dependence of monomer yields and bond contents on SG ratio and addition rate, where we selected multiple SG ratios between 0.1 and 10 and ran the simulations for these scenarios.

# In[6]:


sg_opts = [0.1,0.2,0.25,0.33,0.5,1,2,3,4,5,10]

fun = par.delayed(kmc.run)

for startNum in [2,50,100,250,499]:
    resultsToSave = []

    for sg in sg_opts:
        #Set the percentage of S
        pct = sg / (1 + sg)

        #Make choices about what kinds of monomers there are
        n = startNum
        rands = np.random.rand(n)

        #Initialize the monomers, events, and state
        mons = [ kmc.Monomer( int ( sOrG < pct ) , i ) for i,sOrG in zip ( range(n) , rands ) ]
        startEvents = [ kmc.Event ( 'ox' , [i] , rates['ox'][ int( sOrG < pct ) ]['monomer'] ) for i,sOrG in zip ( range(n) , rands) ]

        state = { i : {'mon' : mons[i] , 'affected' : {startEvents[i]} } for i in range(n) }
        events = { startEvents[i] for i in range(n) }
        events.add(kmc.Event('grow',[],rate = 1e0,bond = sg))

        results = par.Parallel(n_jobs = 16)([fun(nMax = 500, tFinal = 1e5,rates = rates,initialState = state,initialEvents = events)
                                            for _ in range(100)])
        resultsToSave.append(results)
        print('Completed sensitivity iteration: ' + str(sg))

        t = time.localtime()
        print('Finished on ' + str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) 
                + ' at ' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec))

    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\startNum_1e0add\\{startNum:0n}.pkl','wb+') as file:
        pickle.dump(resultsToSave,file)
        print('Saving.... completed')


# In[7]:


for startNum in [2,50,100,250,499]:
    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\startNum_1e0add\\{startNum:0n}.pkl','rb') as file:
        raw = pickle.load(file)
    
    results = [0]*len(raw)
    for i in range(len(raw)):
        results[i] = par.Parallel(n_jobs = 8)( [ par.delayed(kmc.analyze)(adjacency = raw[i][j]['adjacency_matrix'], 
                                                                          nodes = raw[i][j]['monomers'] ) 
                                                 for j in range( len ( raw[i] ) ) ] )
        
        t = time.localtime()
        
        print('Finished on ' + str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) 
                + ' at ' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec))

        
    with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\startNum_1e0add\\{startNum:0n}_analysis.pkl','wb+') as file:
        pickle.dump(results,file)


# ## C-lignin example

# In[8]:


fun = par.delayed(kmc.run)

mons = [ kmc.Monomer( 2 , i ) for i in range(n) ]
startEvents = [ kmc.Event ( 'ox' , [i] , cRates['ox'][ 2 ]['monomer'] ) for i in range(n) ]

state = { i : {'mon' : mons[i] , 'affected' : {startEvents[i]} } for i in range(n)}
events = { startEvents[i] for i in range(n)}
events.add(kmc.Event('grow',[],rate=1e4))

res = kmc.run(nMax = 10,tFinal = 1, rates = cRates, initialState = state, initialEvents = events)

#Make choices about what kinds of monomers there are
n = 2

#Initialize the monomers, events, and state
mons = [ kmc.Monomer( 2 , i ) for i in range(n) ]
startEvents = [ kmc.Event ( 'ox' , [i] , cRates['ox'][ 2 ]['monomer'] ) for i in range(n) ]

state = { i : {'mon' : mons[i] , 'affected' : {startEvents[i]} } for i in range(n) }
events = { startEvents[i] for i in range(n) }
events.add( kmc.Event('grow',[],rate = 1e0 ) )

results = par.Parallel(n_jobs = 16)([fun(nMax = 500, tFinal = 1e5,rates = cRates,initialState = state,initialEvents = events)
                                    for _ in range(100)])

with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\cLignin\\slow_results.pkl','wb+') as file:
    pickle.dump(results,file)


# In[11]:


with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\cLignin\\slow_results.pkl','rb') as file:
    results = pickle.load(file)
    
analysis = par.Parallel(n_jobs = 16)( [ par.delayed(kmc.analyze)(adjacency = results[i]['adjacency_matrix'], 
                                                                          nodes = results[i]['monomers'] ) 
                                                 for i in range( len ( results ) ) ] )

with open(f'C:\\Users\\MikeOrella\\Documents\\GitHub\\lignin-kmc\\results\\cLignin\\slow_results_analysis.pkl','wb+') as file:
    pickle.dump(analysis,file)


# In[12]:


print(analysis)


# In[ ]:





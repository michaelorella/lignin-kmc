# lignin-KMC
The official source for kinetic Monte Carlo polymerization packages developed to understand native lignin 
polymerization. Predict desired lignin structures from first principle energetic calculations. Using this code, you can 
output either predicted properties from an ensemble of lignin structures, or the molfile for a single simulation. 
Currently the code-base supports caffeoyl alcohol (C), sinapyl alcohol (S), and coniferyl alcohol (G).

Code Style: Python standard

# Motivation
Within the lignin community, there is a discrepancy in the understanding of how lignin polymerizes within the plant 
cell wall. This code was developed to help address some of these questions. In essence, the goal of this project was to 
use first principles calculations of the kinetics of monomer couplings that was able to make qualitative predictions to 
make quantitative predictions within this framework.

# Use

The following sections contain information on how to use lignin-KMC. lignin-KMC was developed and tested in a Windows 
10 environment, but should function anywhere that Python is installed.

## Framework
This project runs on Python 3.6 with the following packages installed:
- SciPy 1.1.0
- NumPy 1.15.1
- MatPlotLib 2.2.2
- JobLib 0.12.2
- rdKit 2018.03.4.0

## Installation
For users with no Python experience, a helpful guide to installing Python via miniconda or anaconda can be found 
[here](https://conda.io/docs/user-guide/install/index.html). Once Python has been installed, you will need to use 
[conda commands](https://conda.io/docs/user-guide/tasks/manage-environments.html) to install the dependencies listed 
above. [RDKit](https://www.rdkit.org/docs/Install.html) must be installed from the `rdkit` channel (using the -c flag). 

Navigate to the directory where you would like the local copy of the source code to exist, and then clone the entire 
repository using:
```
git clone https://github.com/michaelorella/lignin-kmc
```

In the root, you will find a file titled `environment.yml`. This file contains all of the dependencies listed above, 
with the versions tested. To create your own environment mirroring this one, run the following command in the terminal 
(or Anaconda Prompt on Windows):
```
conda env create -f environment.yml
```

An alternative to get the most recent packages (this is not tested and therefore less stable):
```
conda create -c rdkit -n lignin_kmc rdkit=2018.03.4.0 python=3.6.6 scipy=1.1.0 numpy=1.15.1 matplotlib=2.2.2 
joblib=0.12.2 jupyter=1.0.0
```

Once the environment has been created, you can install the lignin-KMC module using a Python `import` statement. The 
necessary package to import is `ligninkmc`. To do this, start the environment you just created in conda and run Python:
```
(base) ~/ conda activate lignin_kmc
(lignin_kmc) ~/ Python

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.

>>> import ligninkmc as kmc
```

Congratulations! Lignin-KMC is now installed! With this basic installation, you will have access to the functions 
`run`, `generateMol`, `moltosvg`, and `analyze` and the classes `Monomer` and `Event`.

## Examples
For these examples, I will assume that the rates have already been obtained and have been input as a 3-dimensional 
dictionary (bond type, monomer types, oligomer sizes) to transition state energies (relative to the reactant energies) 
measured in Joules. For more complete examples including the definition of the energies, see the files 
`~/Lignin Polymerization Notebook.ipynb` and `~/Example.ipynb`. 

The first step of these simulations is to initialize monomers and events that will start the simulation. For this 
simple example, we will make all S-type lignin with 5 monomers. There will be no possibility of adding monomers to the 
simulation as time goes on. Finally, the only events at the start are oxidations of the monolignol. This information is 
then compiled by the `run` function in the module that executes the Gillespie algorithm on the *in silico* lignin 
state. The output from the simulation is a single dictionary data structure that contains a list of the monomer objects 
in their state at the end of the simulation, the adjacency matrix that describes connectivity between the monomers, and 
the times at which events were executed.
```
>>> mons = [ kmc.Monomer ( 1 , i ) for i in range(5) ]
>>> startEvs = [ kmc.Event ( 'ox' , [i] , rates['ox'][1]['monomer'] ) for i in range(5) ]
>>> state = { i : { 'mons' : mons[i] , 'affected' : {startEvents[i]} } for i in range(5) }
>>> events = { startEvents[i] for i in range(5) }
>>> events.add( kmc.Event( 'grow' , [ ] , rate = 0 , bond = 1 ) )
>>> res = kmc.run( tFinal = 1e9 , rates = rates, initialState = state, initialEvents = events)

{'monomers': _____ , 'adjacency_matrix': _______ , 'time': ______ }
```
Once this output has been obtained, it is more helpful to extract meaningful information from the data structure. This 
can be done using the same `kmc` module that ran the simulation, as shown below.
```
>>> adj = res['adjacency_matrix']
>>> mons = res['monomers']
>>> t = res['time']
>>> analysis = kmc.analyze( adjacency = adj , nodes = mons )

{'Chain Lengths': ______ ,'Bonds': _______ ,'RCF Yields': ________ ,'RCF Bonds': _______}
```
Another way to analyze your simulation result instead of examining the measurable properties would be to use the `vis` 
module to convert the simulation output to a single molfile that can then be manipulated with RDKit into different 
output styles for further analysis. An example of this is shown below:
```
>>> block = vis.generateMol(res['adjacency_matrix'],res['monomers'])
>>> from rdkit import Chem
>>> molecule = Chem.MolFromMolBlock(block)
>>> Chem.MolToMolFile(molecule,'./molfile_example.mol')
>>> Chem.MolToSmiles(molecule)
```

## API Reference

### Monomer.py

#### CLASSES

__Monomer__(type, index)
- type = {0,1,2} = a switch that indicates whether the monomer is G = 0, S = 1, or C = 2. Extensions to include more 
  monomers would need to expand this definition
- index = Z+ = a number that should be unique for all monomers in a simulation. This is returned as the hash value and 
  is the tie in to the adjacency matrix

The class that contains information about each monomer in the simulation, most importantly tracking the index and the 
monomer type.

#### FUNCTIONS

### Event.py

#### CLASSES

__Event__(key, index, rate, bond)
- key = str = name of the event that is taking place (e.g. '5o4','ox',etc.)
- index = [Z+,Z+] = list of indices to the monomers that are affected by this event
- rate = R+ = the rate of the event that is occuring (units consistent with time units in simulation, but otherwise 
  meaningless)
- bond = [Z+,Z+] = list of changes that need to be made to the adjacency matrix to perform the event

The class that is used to define events, which can be unpacked by the `run` function to execute the events occurring in 
the simulation.

#### FUNCTIONS

### Analysis.py

#### CLASSES

#### FUNCTIONS

__findFragments__(adj = None)
- adj = DOK_Matrix = NxN sparse matrix in the dictionary of keys format
- return = {{},{},...,{}} = set of sets of connected components within the adjacency matrix 

Identifies connected component subgraphs of the supergraph `adj`, which can be used to determine yields or identify 
whether two monomers are connected.

__fragmentSize__(frags = None)
- frags = {{},{},...,{}} = set of sets returned from `findFragments`
- return = dict() = dictionary mapping all of the indices in frags to the length of the fragment they were contained in

Gets the sizes for all of the monomers in the simulation

__breakBond__(adj, bondType)
- adj = DOK_Matrix = adjacency matrix
- bondType = str = the bond that should be broken
- return = DOK_Matrix = new adjacency matrix after bonds were broken

Modifies the adjacency matrix to reflect certain bonds specified by `bondType` being broken. This is used primarily 
when we are evaluating the effect that reductive cleavage treatment would have on the simulated lignin. As of now, 
all bonds of a given type are removed without discrimination.

__countBonds__(adj)
- adj = DOK_Matrix = adjacency matrix
- return = dict() = a dictionary containing all bond strings mapped to the number of times they occur within `adj`

Used for evaluating the frequency of different linkages within a simulated lignin

__countYields__(adj = None)
- adj = DOK_Matrix = adjacency matrix
- return = Counter() = maps the size of an oligomer to the number of occurrences within the adjacency matrix

Used to count the yields of monomers, dimers, etc. when the simulation is complete

__analyze__(adj = None, nodes = None)
- adj = DOK_Matrix = adjacency matrix
- nodes = [Monomer,Monomer,...,Monomer] = list of monomers in the simulation
- return = dict() = maps different measurable quantities to their values predicted from the simulation

Aggregates all of the analysis that we would want to do into one simple function. At the end of every simulation, we 
can then directly analyze the simulation output to identify the original chain lengths, original bond distributions, 
and these same properties post ether bond cleavage.

### KineticMonteCarlo.py

#### CLASSES

#### FUNCTIONS

__quickFragSize__(monomer = None)
- monomer = Monomer = instance of monomer object that we want to check
- return = str = 'monomer' or 'dimer' 

Uses the open positions attribute of the monomer object to determine whether there is anything connected to this monomer yet

__updateEvents__(monomers = None, adj = None, lastEvent = None, events = {}, rateVec = None, r = None, maxMon = 500)
- monomers = dict() = maps the index of the monomer to the object and the events that a change to this index would effect
- adj = DOK_Matrix = adjacency matrix
- lastEvent = Event = the previous event that occurred
- events = dict() = map the hash value of each event to the unique event - this is all of the possible events at the 
  current state after the method is run
- rateVec = dict() = map the hash value of each event to the rate of that event
- r = dict() = the rates that are obtained *a priori* from DFT calculations
- maxMon = Z+ = the maximum number of monomers in the simulation
- return = None

Mutates the dictionary of events and rateVec that are passed to the function. These mutations are done so that the 
entire state doesn't need to be reconstructed on every iteration of the simulation. Once these changes are made, the 
event choice is ready to be made and the chosen event can be performed.

__doEvent__(event = None, state = None, adj = None)
- event = Event = the event that was chosen to be performed
- state = dict() = the dictionary mapping monomer indices to the monomer object and events that would be changed by a 
  change to the monomer
- adj = DOK_Matrix = adjacency matrix

Updates the monomers and adjacency matrix to reflect the execution of the chosen event

__run__(nMax = 10, tFinal = 10, rates = None, initialState = None, initialEvents = None)
- nMax = Z+ = maximum number of monomers in the simulation
- tFinal = R+ = final simulation time (should be units consistent with the rates)
- rates = dict() = the rate of each of the possible events as a 3-d dictionary mapping bond type, monomers sizes, and 
  monomer types to a rate in units consistent with your final time definition
- initialState = dict() = map of monomer index to monomer object and changeable events
- initialEvents = dict() = map of event hashes to event objects
- return = dict() = adjacency matrix, list of monomers, and list of time steps at the end of the simulation

Runs the Gillespie algorithm on the situation specified by the parameters. This is the workhorse of the code, where the 
monomers are changed and linked to simulate the growth of lignin *in planta*.

### Visualization.py

#### CLASSES

#### FUNCTIONS

__generateMol__(adj,nodeList)
- adj = DOK_Matrix = adjacency matrix
- nodeList = [Monomer,Monomer,...,Monomer] = list of monomers output from the simulation
- return = str = molfile contents

Generates a file format as specified by CTAN that represents the molecule that was just simulated by `run`. This file 
can then be used together with rdKit for further visualization or analysis, or any one of your favorite chemical 
drawing software.

__moltosvg__(mol,molSize=(450,150),kekulize=True)
- mol = rdkit.molecule = molecule object
- molSize = (R+,R+) = tuple of pixel dimensions of the drawing area
- kekulize = bool = flag for whether the molecule should be Kekulized or not
- return = svg = svg file with the drawing

Outputs the vector graphic image of the molecule that was just simulated. Useful for easy and quick visualization in 
Jupyter notebooks for simulation results, but doesn't do a great job of 2D structure cleaning. This function is adapted 
from the 2015.03 release of RDKit.

# Credits
[<img src="https://avatars0.githubusercontent.com/u/40570716?s=400&u=7bde054e05bbba59c19cefd3aa2f4c84e2a9dfc6&v=4" height="150" width="150">](https://github.com/michaelorella) | [<img src="https://avatars0.githubusercontent.com/u/17909849?s=460&v=4" height="150" width="150">](https://github.com/terrygani)
--- | --- 
[Michael Orella](https://github.com/michaelorella) | [Terry Gani](https://github.com/terrygani)


# Contribute
Thank you for your interest in adding to our understanding of lignin polymerization!

The following guidelines can help you get started within this project. Use your best judgement and coding knowledge to 
help improve this project, and as always feel free to propose changes to the project or document in a pull request.

## Getting started
Lignin-KMC is built on Python 3. If you are new to Python, refer to the fantastic 
[documentation](https://docs.python.org/3.6/).

## Issues
Before submitting your own issue, make sure the same one doesn't already exist by searching under 
[issues](https://github.com/michaelorella/lignin-kmc/issues). If you don't locate a similar issue already, feel free to 
open a [new issue](https://github.com/michaelorella/lignin-kmc/issues/new). When writing your issue, consider the 
following points and be as detailed as possible:

1. Write step by step directions for replicating your issue
2. Describe what you would expect to happen and what you actually get
3. Provide screenshots of your dependency versions
4. Include your operating system version and Python version

## Pull Requests
Pull requests are always welcome for suggestions to improve either the code or usability. Before submitting the pull 
request, please ensure that your standalone code is working properly.

# License
MIT © Michael Orella

# lignin-KMC
The official source for kinetic Monte Carlo polymerization packages developed to model lignin biosynthesis from first 
principle energetic calculations. Using this code, you can output either predicted properties (e.g. oligomer length 
and bond composition) from an ensemble of lignin structures, or the molfile for a single simulation. 
Currently the package simulates lignin polymerization of guaiacyl (G) and syringyl (S) monolignols (in a specified 
ratio), or from caffeoyl (C) monolignols (only).

Further details are available in the manuscript ["Lignin-KMC: A Toolkit for Simulating Lignin 
Biosynthesis"](https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534). Please cite this paper if you use this 
package.

Code Style: Python standard

# Motivation
Within the lignin community, there is a discrepancy in the understanding of how lignin polymerizes within the plant 
cell wall. This code was developed to help address some of these questions. In essence, the goal of this project was to 
use first principles calculations of the kinetics of monomer couplings that was able to make qualitative predictions to 
make quantitative predictions within this framework.

# Use

The following sections contain information on how to use lignin-KMC.

## Framework
This project runs on Python ≥3.6 with the following packages installed:
- SciPy
- NumPy
- MatPlotLib
- JobLib
- rdKit
- common-wrangler

## Installation
For users with no Python experience, a helpful guide to installing Python via miniconda or anaconda can be found 
[here](https://conda.io/docs/user-guide/install/index.html). Once Python has been installed, you will need to use 
[conda commands](https://conda.io/docs/user-guide/tasks/manage-environments.html) to install the dependencies listed 
above. [RDKit](https://www.rdkit.org/docs/Install.html) must be installed from the `rdkit` channel (using the -c flag).
[common-wrangler](https://pypi.org/project/common-wrangler/) can be installed using pip (`pip install common-wrangler`).

### Basic Use



### Developer Use

Navigate to the directory where you would like the local copy of the source code to exist, and then clone the 
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

Once the environment has been created, you can install the lignin-KMC module using a Python `import` statement. The 
necessary package to import is `ligninkmc`. To do this, start the environment you just created in conda and run Python:
```
(base) ~/ conda activate lignin_kmc
(lignin_kmc) ~/ Python

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.

>>> import ligninkmc as kmc
```

Congratulations! Lignin-KMC is now installed! With this basic installation, you will have access to the function 
(such as `run_kmc`, `generate_mol`, and `analyze_adj_matrix`) and the classes `Monomer` and `Event`.

## Examples
For these examples, I will assume that the rates have already been obtained and have been input as a 3-dimensional 
dictionary (bond type, monomer types, oligomer sizes) to transition state free energy barriers or reaction rates at 
the temperature of interest. For examples (from the definition of the energy barrier dictionary to the simulation of 
lignin biosynthesis), see `~/LigninPolymerizationNotebook.ipynb` and `~/Example.ipynb`. 

## API Reference

### monomer.py

#### CLASSES

__Monomer__(type, index)
- type = {0, 1, 2} = a switch that indicates whether the monomer is G = 0, S = 1, or C = 2. Extensions to include more 
  monomers would be needed to expand this definition
- index = Z+ = a number that should be unique for all monomers in a simulation. This is returned as the hash value and 
  is the tie in to the adjacency matrix

The class that contains information about each monomer in the simulation, most importantly tracking the index and the 
monomer type.

#### FUNCTIONS

### event.py

#### CLASSES

__Event__(key, index, rate, bond)
- key = str = name of the event that is taking place (e.g. '5o4', 'ox', etc.)
- index = [Z+, Z+] = list of indices to the monomers that are affected by this event
- rate = R+ = the rate of the event that is occurring (units consistent with time units in simulation)
- bond = [Z+,Z+] = list of changes that need to be made to the adjacency matrix to perform the event

The class that is used to define events, which can be unpacked by the `run` function to execute the events occurring in 
the simulation.


### analysis.py

#### FUNCTIONS

__find_fragments__(adj)
- adj = dok_matrix = NxN sparse matrix in the dictionary of keys format
- return = [{},{},...,{}] = list of sets of connected components within the adjacency matrix 

Identifies connected component subgraphs of the supergraph `adj`, which can be used to determine yields or identify 
whether two monomers are connected.

__fragment_size__(frags)
- frags = [{},{},...,{}] = list of sets as returned from `find_fragments`
- return = dict() = dictionary mapping all of the indices in frags to the length of the fragment they were contained in

Gets the sizes for all of the monomers in the simulation

__break_bond_type__(adj, bondType)
- adj = scipy dok_matrix = adjacency matrix
- bond_type = str = the bond that should be broken
- return = dok_matrix = new adjacency matrix after bonds were broken

Modifies the adjacency matrix to reflect certain bonds specified by `bondType` being broken. This is used primarily 
when we are evaluating the effect that reductive cleavage treatment would have on the simulated lignin. As of now, 
all bonds of a given type are removed without discrimination.

__count_bonds__(adj)
- adj = dok_matrix = adjacency matrix
- return = dict() = a dictionary containing all bond strings mapped to the number of times they occur within `adj`

Used for evaluating the frequency of different linkages within a simulated lignin

__count_oligomer_yields__(adj)
- adj = dok_matrix = adjacency matrix
- return = dict = maps the size of an oligomer to the number of occurrences within the adjacency matrix

Used to count the yields of monomers, dimers, etc., when the simulation is complete

__analyze_adj_matrix__(adjacency)
- adjacency = dok_matrix = adjacency matrix
- return = dict() = maps different measurable quantities to their values predicted from the simulation

Aggregates analysis of oligomer length and bond types, and these same properties post C-O bond cleavage.

### kmc_functions.py

#### FUNCTIONS

__quick_frag_size__(monomer)
- monomer = Monomer = instance of monomer object that we want to check
- return = str = 'monomer' or 'dimer' 

Uses the open positions attribute of the monomer object to determine whether there is anything connected to this monomer yet

__update_events__(monomers, adj, last_event, events, rate_vec, rate_dict, max_mon=500)
- monomers = dict() = maps the index of the monomer to the object and the events that a change to this index would effect
- adj = dok_matrix = adjacency matrix
- last_event = Event = the previous event that occurred
- events = dict() = map the hash value of each event to the unique event - this is all of the possible events at the 
  current state after the method is run
- rate_vec = dict() = map the hash value of each event to the rate of that event
- rate_dict = dict() = the rates that are obtained *a priori* from DFT calculations
- max_mon = Z+ = the maximum number of monomers in the simulation
- return = None

Mutates the dictionary of events and rateVec that are passed to the function. These mutations are done so that the 
entire state doesn't need to be reconstructed on every iteration of the simulation. Once these changes are made, the 
event choice is ready to be made and the chosen event can be performed.

__do_event__(event, state, adj, sg_ratio=None)
- event = Event = the event that was chosen to be performed
- state = dict() = the dictionary mapping monomer indices to the monomer object and events that would be changed by a 
  change to the monomer
- adj = dok_matrix = adjacency matrix
- sg_ratio = float needed if and only if: a) there are S and G and only S and G, and b) new monomers will be added

Updates the monomers and adjacency matrix to reflect the execution of the chosen event

__run_kmc__(rate_dict, initial_state, initial_events, n_max=10, t_max=10, dynamics=False, random_seed=None, 
            sg_ratio=None)
- rate_dict:  dict -- the rate of each of the possible events as a 3-d dictionary mapping bond type, monomers sizes,  
                      and monomer types to a rate in units consistent with your final time definition
- initial_state: dict  -- The dictionary mapping the index of each monomer to a dictionary with the monomer
-      and the set of events that a change to this monomer would impact
- initial_events: dictionary -- The dictionary mapping event hash values to those events
- n_max: int   -- The maximum number of monomers in the simulation
- t_max: float -- The final simulation time (units depend on units of rates)
- dynamics: boolean -- if True, will keep values for every time step
- random_seed: None or hashable value to aid testing
- sg_ratio: needed if there is S and G and nothing else
- return: dict with the simulation times, adjacency matrix, and list of monomers at the end of the simulation

Runs the Gillespie algorithm on the situation specified by the parameters. This is the workhorse of the code, where the 
monomers are changed and linked to simulate the growth of lignin *in planta*.

### visualization.py

#### FUNCTIONS

__generate_mol__(adj,nodeList)
- adj = dok_matrix = adjacency matrix
- nodeList = [Monomer, Monomer, ..., Monomer] = list of monomers output from the simulation
- return = str = molfile contents

Generates a file format as specified by [CTAN](https://ctan.org/) that represents the molecule that was just simulated 
by `run_kmc`. This file can then be used together with rdKit for further visualization or analysis, or any one of your 
favorite chemical drawing software packages.

# Credits
[<img src="https://avatars0.githubusercontent.com/u/40570716?s=400&u=7bde054e05bbba59c19cefd3aa2f4c84e2a9dfc6&v=4" height="150" width="150">](https://github.com/michaelorella) | [<img src="https://avatars0.githubusercontent.com/u/17909849?s=460&v=4" height="150" width="150">](https://github.com/terrygani)
--- | --- | ---
[Michael Orella](https://github.com/michaelorella) | [Terry Gani](https://github.com/terrygani) | 
[Heather Mayes](https://github.com/team-mayes)


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
request, please ensure that your standalone code is working properly by both running the existing tests and adding 
tests of any new functionality.

# License
MIT © Michael Orella

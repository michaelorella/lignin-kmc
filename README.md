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
- rdKit (must be installed via conda; see below)
- common-wrangler

## Installation

- For users with no Python experience, a helpful guide to installing Python via miniconda or anaconda can be found 
[here](https://conda.io/docs/user-guide/install/index.html). 

- Once Python has been installed, you will need install [RDKit](https://www.rdkit.org/docs/Install.html) using conda. 
  -  To install it in a new environment, run: `conda create -c rdkit -n lignin-kmc rdkit`, followed by 
     `conda activate lignin-kmc` 
  -  If you already have an environment created and want to add rdkit to that environment, instead run: 
     `conda install -c conda-forge rdkit`

-  In either case, Conda will install any missing required dependencies when it does so, and thus this may take a few 
 minutes.
 
- If you do not already have a `$HOME/.local/bin` directory, create one now, before installing lignin-kmc.
 
-  You can then install [lignin-kmc](https://pypi.org/project/common-wrangler/) using  pip (`pip install lignin-kmc`). 
Additional dependencies will again be installed as required.

### Command-Line Use

Lignin-KMC is populated with default variables that make it as easy to run as entering (on a terminal):

`> create_lignin`

This will use all default values (see below) and output which will look something like:
    
    Running Lignin-KMC version 0.2.2. Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534
    
    Lignin KMC created 10 monomers, which formed:
           1 oligomer(s) of chain length 10, with branching coefficient 0.2
    composed of the following bond types and number:
        BO4:    6     BB:    1     B5:    1     B1:    0    5O4:    0    AO4:    0     55:    1
    
    Breaking C-O bonds to simulate RCF results in:
           4 monomer(s) (chain length 1)
           3 dimer(s) (chain length 2)
    with the following remaining bond types and number:
        BO4:    0     BB:    1     B5:    1     B1:    0    5O4:    0    AO4:    0     55:    1
    
    SMILES representation: 
     COc1cc(C(O)C(CO)Oc2c(OC)cc(C3OCC4C(c5cc(OC)c(OC(CO)C(O)c6cc(OC)c(OC(CO)C(O)c7cc(OC)c([O])c(OC)c7)c(-c7cc(C(O)C(CO)Oc8c(OC)cc(C(O)C(CO)Oc9c(OC)cc(C%10Oc%11c(OC)cc(/C=C/CO)cc%11C%10CO)cc9OC)cc8OC)cc(OC)c7OC(CO)C(O)c7cc(OC)c([O])c(OC)c7)c6)c(OC)c5)OCC34)cc2OC)ccc1[O] 



Your output will differ as a pseudo-random number generator is used to model the stochastic nature of chemical reactions.

The default options (number of initial and final monomers, S:G ratio, etc.) can be changed either by the command line 
options shown below, or by using a configuration file. These options can be viewed by entering the help command:

 `> create_lignin -h`
 
     Running Lignin-KMC version 0.2.2. Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534
     
     usage: create_lignin [-h] [-c CONFIG] [-d OUT_DIR] [-f OUTPUT_FORMAT_LIST]
                          [-i INITIAL_NUM_MONOMERS] [-l LENGTH_SIMULATION]
                          [-m MAX_NUM_MONOMERS] [-o OUTPUT_BASENAME]
                          [-r RANDOM_SEED] [-s IMAGE_SIZE] [-sg SG_RATIO]
                          [-t TEMPERATURE_IN_K]
     
     Create lignin chain(s) composed of 'S' (syringyl) and/or 'G' (guaiacol) monolignols, as described in:
       Orella, M., Gani, T. Z. H., Vermaas, J. V., Stone, M. L., Anderson, E. M., Beckham, G. T., 
       Brushett, Fikile R., Roman-Leshkov, Y. (2019). Lignin-KMC: A Toolkit for Simulating Lignin Biosynthesis.
       ACS Sustainable Chemistry & Engineering. https://doi.org/10.1021/acssuschemeng.9b03534. C-Lignin can be 
       modeled with the functions in this package, as shown in ipynb examples in our project package on github 
       (https://github.com/michaelorella/lignin-kmc/), but not currently from the command line. If this 
       functionality is desired, please start a new issue on the github.
     
       By default, the Gibbs free energy barriers from this reference will be used, as specified in Tables S1 and S2.
       Alternately, the user may specify values, which should be specified as a dict of dict of dicts in a 
       specified configuration file (specified with '-c') using the 'e_barrier_in_kcal_mol' or 'e_barrier_in_j_particle'
       parameters with corresponding units (kcal/mol or joules/particle, respectively), in a configuration file 
       (see '-c'). The format is (bond_type: monomer(s) involved: units involved: ea_vals), for example:
           ea_dict = {oxidation: {0: {monomer: 0.9, oligomer: 6.3}, 1: {{{MONOMER}: 0.6, {OLIGOMER}: 2.2}}, ...}
       where 0: guaiacol, 1: syringyl, 2: caffeoyl. The default output is a SMILES string printed to standard out.
     
       All command-line options may alternatively be specified in a configuration file. Command-line (non-default) 
       selections will override configuration file specifications.
     
     optional arguments:
       -h, --help            show this help message and exit
       -c CONFIG, --config CONFIG
                             The location of the configuration file in the 'ini' format. This file can be used to 
                             overwrite default values such as for energies.
       -d OUT_DIR, --out_dir OUT_DIR
                             The directory where output files will be saved. The default is the current directory.
       -f OUTPUT_FORMAT_LIST, --output_format_list OUTPUT_FORMAT_LIST
                             The type(s) of output format to be saved. Provide as a space- or comma-separated list. 
                             The currently supported types are: 'json', 'png', 'smi', 'svg', 'tcl'. 
                             The 'json' option will save a json format of RDKit's 'mol' (molecule) object. The 'tcl' 
                             option will create a file for use with VMD to generate a psf file and 3D molecules, 
                             as described in LigninBuilder, https://github.com/jvermaas/LigninBuilder, 
                             https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.8b05665. 
                             A base name for the saved files can be provided with the '-o' option. Otherwise, the 
                             base name will be 'lignin-kmc-out'.
       -i INITIAL_NUM_MONOMERS, --initial_num_monomers INITIAL_NUM_MONOMERS
                             The initial number of monomers to be included in the simulation. The default is 2.
       -l LENGTH_SIMULATION, --length_simulation LENGTH_SIMULATION
                             The length of simulation (simulation time) in seconds. The default is 1 s.
       -m MAX_NUM_MONOMERS, --max_num_monomers MAX_NUM_MONOMERS
                             The maximum number of monomers to be studied. The default value is 10.
       -o OUTPUT_BASENAME, --output_basename OUTPUT_BASENAME
                             The base name for output file(s). If an extension is provided, it will determine 
                             the type of output. Currently supported output types are: 
                             'json', 'png', 'smi', 'svg', 'tcl'. Multiple output formats can be selected with the 
                             '-f' option. If the '-f' option is selected and no output base name provided, a 
                             default base name of 'lignin-kmc-out' will be used.
       -r RANDOM_SEED, --random_seed RANDOM_SEED
                             A positive integer to be used as a seed value for testing.
       -s IMAGE_SIZE, --image_size IMAGE_SIZE
                             The output size of svg or png files in pixels (provide two integers). The default size 
                             is (1200, 300) pixels.
       -sg SG_RATIO, --sg_ratio SG_RATIO
                             The S:G (guaiacol:syringyl) ratio. The default is 1.
       -t TEMPERATURE_IN_K, --temperature_in_k TEMPERATURE_IN_K
                             The temperature (in K) at which to model lignin biosynthesis. The default is 298.15 K.

For example, to use an S to G ratio of 2.5, 12 initial monomers, and up to 18 monomers (only would not reach this 
if there was insufficient time; the default 1 s will be plenty), with the remaining variables set as their 
default values, enter:

 `> create_lignin -sg 2.5 -i 12 -m 28`

    Running Lignin-KMC version 0.2.2. Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534
    
    Lignin KMC created 18 monomers, which formed:
           1 oligomer(s) of chain length 18, with branching coefficient 0.111
    composed of the following bond types and number:
        BO4:    8     BB:    4     B5:    2     B1:    0    5O4:    3    AO4:    0     55:    0
    
    Breaking C-O bonds to simulate RCF results in:
           6 monomer(s) (chain length 1)
           6 dimer(s) (chain length 2)
    with the following remaining bond types and number:
        BO4:    0     BB:    4     B5:    2     B1:    0    5O4:    0    AO4:    0     55:    0
    
    SMILES representation: 
     COc1cc(C(O)C(CO)Oc2c(OC)cc(C3OCC4C(c5cc(OC)c(Oc6cc(C7OCC8C(c9cc(OC)c(OC(CO)C(O)c%10cc(OC)c(OC(CO)C(O)c%11cc(OC)c%12c(c%11)C(CO)C(c%11cc(OC)c([O])c(OC)c%11)O%12)c(OC)c%10)c(Oc%10c(OC)cc(C%11OCC%12C(c%13cc(OC)c(OC(CO)C(O)c%14cc(OC)c%15c(c%14)C(CO)C(c%14cc(OC)c(OC(CO)C(O)c%16cc(OC)c(OC(CO)C(O)c%17cc(OC)c([O])c(Oc%18c(OC)cc(C(O)C(CO)Oc%19c(OC)cc(C%20OCC%21C(c%22cc(OC)c([O])c(OC)c%22)OCC%20%21)cc%19OC)cc%18OC)c%17)c(OC)c%16)c(OC)c%14)O%15)c(OC)c%13)OCC%11%12)cc%10OC)c9)OCC78)cc(OC)c6OC(CO)C(O)c6cc(OC)c([O])c(OC)c6)c(OC)c5)OCC34)cc2OC)cc(OC)c1[O] 


### Developer Use

Navigate to the directory where you would like the local copy of the source code to exist, and then clone the 
repository using:
```
git clone https://github.com/michaelorella/lignin-kmc
```

In the root, you will find a file titled `environment.yml`. This file contains all of the dependencies listed above 
plus two additional packages required for testing (pytest and joblib), with the versions tested. To create your own 
environment mirroring this one, run the following command in the terminal (or Anaconda Prompt on Windows):

```
conda env create -f environment.yml
```

Once the environment has been created, activate it:

```
conda activate lignin_kmc
```

Congratulations! Lignin-KMC is now installed! With this basic installation, you will have access to the functions 
therein (such as `run_kmc`, `generate_mol`, and `analyze_adj_matrix`) and the classes `Monomer` and `Event`.

## Examples
For examples, see `~/LigninPolymerizationNotebook.ipynb`, `~/Example.ipynb`, and `Dynamics.ipynb`. 

## API Reference

### kmc_common.py

#### CLASSES

__Event__(key, index, rate, bond)
- key = str = name of the event that is taking place (e.g. '5o4', 'ox', etc.)
- index = [int, int] = list of indices to the monomers that are affected by this event
- rate = R+ = the rate of the event that is occurring (units consistent with time units in simulation)
- bond = [int, int] = list of changes that need to be made to the adjacency matrix to perform the event

The class that is used to define events, which can be unpacked by the `run` function to execute the events occurring in 
the simulation.

__Monomer__(type, index)
- type = {0, 1, 2} = a switch that indicates whether the monomer is G = 0, S = 1, or C = 2. Extensions to include more 
  monomers would be needed to expand this definition
- index = int = a number that should be unique for all monomers in a simulation. This is returned as the hash value and 
  is the tie in to the adjacency matrix

The class that contains information about each monomer in the simulation, most importantly tracking the index and the 
monomer type.


### kmc_functions.py

#### FUNCTIONS

__find_fragments__(adj)
- adj = dok_matrix = NxN sparse matrix in the dictionary of keys format
- return = [{}, {}, ..., {}], [int, int, ..., int] = list of sets of connected components within the adjacency matrix 
and a list of ints containing the number of number of branch points found in each fragment.

Identifies connected component subgraphs of the supergraph `adj`, which can be used to determine yields or identify 
whether two monomers are connected.

__fragment_size__(frags)
- frags = [{},{},...,{}] = list of sets as returned from `find_fragments`
- return = dict() = dictionary mapping all of the indices in frags to the length of the fragment they were contained in

Gets the sizes for all of the monomers in the simulation

__quick_frag_size__(monomer)
- monomer = Monomer = instance of monomer object that we want to check
- return = str = 'monomer' or 'oligomer' 

Uses the open positions attribute of the monomer object to determine whether there is anything connected to this monomer yet

__break_bond_type__(adj, bond_type)
- adj = scipy dok_matrix = adjacency matrix
- bond_type = str = the bond that should be broken
- return = dok_matrix = new adjacency matrix after bonds were broken

Modifies the adjacency matrix to reflect certain bonds specified by `bondType` being broken. This is used primarily 
when we are evaluating the effect that reductive cleavage treatment would have on the simulated lignin. As of now, 
all bonds of a given type are removed without discrimination.

__count_bonds__(adj)
- adj = dok_matrix = adjacency matrix
- return = OrderedDict = a dictionary containing all bond strings mapped to the number of times they occur within `adj`

Used for evaluating the frequency of different linkages within a simulated lignin

__count_oligomer_yields__(adj)
- adj = dok_matrix = adjacency matrix
- return = OrderedDict, dict, dict, dict = maps the size of an oligomer to:
     the number of occurrences within the adjacency matrix, 
     the total number of monomers involved in oligomers,
     total number of branch points in oligomers of that length, and
     the branching coefficient for the oligomers of that length

Used to count the yields of monomers, dimers, etc., when the simulation is complete

__analyze_adj_matrix__(adjacency)
- adjacency = dok_matrix = adjacency matrix
- return = dict() = maps different measurable quantities to their values predicted from the simulation

Aggregates analysis of oligomer length and bond types, and these same properties post C-O bond cleavage.

__update_events__(monomers, adj, last_event, events, rate_vec, rate_dict, max_mon=500)
- state_dict = dict() = maps the index of the monomer to the object and the events that a change to this index would effect
- adj = dok_matrix = adjacency matrix
- last_event = Event = the previous event that occurred
- event_dict = dict() = map the hash value of each event to the unique event - this is all of the possible events at the 
  current state after the method is run
- rate_vec = dict() = map the hash value of each event to the rate of that event
- rate_dict = dict() = the reaction rates for all possible reactions, in 1/s or 1/monomer-second
- max_mon = int = the maximum number of monomers in the simulation
- return = None

Mutates the dictionary of events and rateVec that are passed to the function. These mutations are done so that the 
entire state doesn't need to be reconstructed on every iteration of the simulation. Once these changes are made, the 
event choice is ready to be made and the chosen event can be performed.

__do_event__(event, state, adj, sg_ratio=None, random_seed=None)
- event = Event = the event that was chosen to be performed
- state = dict() = the dictionary mapping monomer indices to the monomer object and events that would be changed by a 
  change to the monomer
- adj = dok_matrix = adjacency matrix
- sg_ratio = float needed if and only if: a) there are S and G and only S and G, and b) new monomers will be added
- random_seed = int (positive val) needed if repeatable results are desired (for testing)

Updates the monomers and adjacency matrix to reflect the execution of the chosen event

__run_kmc__(rate_dict, initial_state, initial_events, n_max=10, t_max=10, dynamics=False, random_seed=None)
- rate_dict:  dict -- the rate of each of the possible events as a 3-d dictionary mapping bond type, monomers sizes,  
                      and monomer types to a rate in units consistent with your final time definition
- initial_state: dict  -- The dictionary mapping the index of each monomer to a dictionary with the monomer
-      and the set of events that a change to this monomer would impact
- initial_events: dictionary -- The dictionary mapping event hash values to those events
- n_max: int   -- The maximum number of monomers in the simulation
- t_max: float -- The final simulation time (units depend on units of rates)
- dynamics: boolean -- if True, will keep values for every time step
- random_seed: None or hashable value to aid testing
- return: dict with the simulation times, adjacency matrix, and list of monomers at the end of the simulation

Runs the Gillespie algorithm on the situation specified by the parameters. This is the workhorse of the code, where the 
monomers are changed and linked to simulate the growth of lignin *in planta*.

__generate_mol__(adj, node_list)
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

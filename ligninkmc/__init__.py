from ligninkmc.kmc_functions import run_kmc
from ligninkmc.analysis import analyze_adj_matrix
from ligninkmc.event import Event
from ligninkmc.monomer import Monomer

try:
    from ligninkmc.visualization import (generate_mol)
except Exception as e:
    print(e)
    print("\nRDKit is likely not installed correctly. Visualization methods will be unavailable.")

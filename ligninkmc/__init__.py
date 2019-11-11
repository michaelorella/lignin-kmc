"""
tools for simulating lignin biosynthesis using kinetic Monte Carlo
"""

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__author__ = 'Heather B Mayes'
__email__ = 'hmayes@hmayes.com'

try:
    from ligninkmc.visualization import (generate_mol)
except Exception as e:
    print(e)
    print("\nRDKit is likely not installed correctly. Visualization methods will be unavailable.")

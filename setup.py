# -*- coding: utf-8 -*-
"""
ligninkmc
Kinetic Monte Carlo implementation for creating realistic lignin topologies as described in
    https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534
"""

from setuptools import setup
import versioneer

DOCLINES = __doc__.split("\n")

setup(name='ligninkmc',
      author='Michael Orella, Terry Gani, and Heather Mayes',
      # description="Kinetic Monte Carlo implementation for creating realistic lignin topologies.",
      description=DOCLINES[0],
      # long_description="\n".join(DOCLINES[2:]),
      url='https://github.com/michaelorella/lignin-kmc',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='MIT',
      packages=['ligninkmc'],
      entry_points={'console_scripts': ['create_lignin = ligninkmc.create_lignin:main',
                                        ],
                    },     package_dir={'ligninkmc': 'ligninkmc'},
      python_requires=">3.6",  # Required for f-strings support
      test_suite='tests',
      # rdkit isn't installable via pip, so not included on list below
      install_requires=['scipy', 'numpy', 'matplotlib', 'joblib', 'common_wrangler'],
      # zip_safe=False,
      )

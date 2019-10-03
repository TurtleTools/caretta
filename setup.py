from distutils.core import setup

setup(name='caretta',
      version='1.0',
      authors=["Janani Durairaj", "Mehmet Akdel"],
      packages=["caretta"],
      install_requires=["numpy", "numba", "prody", "biopython"])

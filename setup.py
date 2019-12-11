from distutils.core import setup

setup(name='caretta',
      version='1.0',
      authors=["Janani Durairaj", "Mehmet Akdel"],
      packages=["caretta"],
      install_requires=["numpy", "numba", "scipy", "prody", "biopython", "fire", "pyparsing"],
      extras_require={
          'GUI': ["dash==1.3.1", "dash-bio==0.1.4", "cryptography",
                  "dash-core-components==1.2.1", "dash-html-components==1.0.1", "dash-renderer==1.1.0",
                  "dash-table==4.3.0", "plotly==3.7.1", "flask"]},
      scripts=["bin/caretta-app", "bin/caretta-cli"]
      )

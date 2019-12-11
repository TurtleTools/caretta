from distutils.core import setup

setup(name='caretta',
      version='1.0',
      authors=["Janani Durairaj", "Mehmet Akdel"],
      packages=["caretta"],
      install_requires=["numpy==1.16.2", "numba==0.43.0", "scipy==1.3.1", "prody==1.10.10", "biopython==1.74", "fire==0.2.1", "pyparsing==2.4.2"],
      extras_require={
          'GUI': ["dash==1.3.1", "dash-bio==0.1.4", "cryptography",
                  "dash-core-components==1.2.1", "dash-html-components==1.0.1", "dash-renderer==1.1.0",
                  "dash-table==4.3.0", "plotly==3.7.1", "flask"]},
      scripts=["bin/caretta-app", "bin/caretta-cli"]
      )

from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DISTNAME = "caretta"
DESCRIPTION = "Fast multiple protein structure alignment"
LONG_DESCRIPTION = long_description
MAINTAINER = "Janani Durairaj, Mehmet Akdel"
MAINTAINER_EMAIL = "janani.durairaj@gmail.com"
URL = "https://github.com/TurtleTools/caretta"
LICENSE = "MIT License"
DOWNLOAD_URL = "https://github.com/TurtleTools/caretta"
VERSION = "0.1.0"
INST_DEPENDENCIES = ["numpy",
                     "numba",
                     "scipy",
                     "prody",
                     "biopython",
                     "typer",
                     "pyparsing",
                     "geometricus",
                     "arviz", ]

EXTRA_DEPENDENCIES = {
                         "GUI": [
                             "dash==1.3.1",
                             "dash-bio==0.1.4",
                             "cryptography",
                             "dash-core-components==1.2.1",
                             "dash-html-components==1.0.1",
                             "dash-renderer==1.1.0",
                             "plotly==3.7.1",
                             "flask",
                         ]
                     }

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        packages=["caretta", "caretta/app"],
        package_data={},
        install_requires=INST_DEPENDENCIES,
        extras_require=EXTRA_DEPENDENCIES,
        scripts=["bin/caretta-app", "bin/caretta-cli"],
        long_description_content_type='text/markdown',
    )

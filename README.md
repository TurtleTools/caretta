<p align="center"><img src="caretta_logo.png" width="300" title="Caretta Logo"></p>

# Caretta – A multiple protein structure alignment and feature extraction suite

Caretta is a software-suite to perform multiple protein structure alignment and structure feature extraction.

Visit the [demo server](http://bioinformatics.nl/caretta) to see caretta's capabilities. The server only allows alignment of up to 50 proteins at once.
The command-line tool and self-hosted web application do not have this restriction.

## Installation

### Requirements
#### Operating system support
1. Linux and Mac
* All capabilities are supported
2. Windows
* The external tool **msms** is not available in Windows. Due to this:
    * Feature extraction is not available.
    * `features` argument in caretta-cli must always be run with `--only-dssp`. 
    * `caretta-app` is not available.

#### Software
Caretta works with Python 3.7+
Run the following commands to install required external dependencies (Mac and Linux only):
```bash
conda install -c salilab dssp
conda install -c bioconda msms
```

### Download caretta
```bash
git clone https://github.com/TurtleTools/caretta.git
cd caretta
```

### Install both the command-line interface and the web-application (Mac and Linux only):
```bash
pip install -e ".[GUI]"
```

### Install only the command-line interface:
```bash
pip install .
```

### Environment variables:
```bash
export OMP_NUM_THREADS=1 # this should always be 1
export NUMBA_NUM_THREADS=20 # change to required number of threads
```

## Usage

### Command-line Usage

```bash
caretta-cli input_pdb_folder
# e.g. caretta-cli test_data  
# caretta-cli --help for more options
```

### Web-application Usage (Mac and Linux only)

```bash
caretta-app <host-ip> <port> 
# e.g. caretta-app localhost 8091
```
Then go to localhost:8091/caretta in a browser window.

## Publications
Janani Durairaj, Mehmet Akdel, Dick de Ridder, Aalt DJ can Dijk. "Fast and adaptive protein structure representations for machine learning."  Poster presented at the [Machine Learning for Structural Biology Workshop](mlsb.io), NeurIPS 2020 (https://www.mlsb.io/papers/MLSB2020_Fast_and_adaptive_protein.pdf)

![MLSB2020.png](MLSB2020.png)


Akdel, Mehmet, Janani Durairaj, Dick de Ridder, and Aalt DJ van Dijk. "Caretta-A Multiple Protein Structure Alignment and Feature Extraction Suite." Computational and Structural Biotechnology Journal (2020). (https://doi.org/10.1016/j.csbj.2020.03.011)

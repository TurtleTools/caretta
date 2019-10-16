# Caretta

Caretta is a software-suite to perform multiple protein structure alignment and structure feature extraction.

Visit the [demo server](http://bioinformatics.nl/caretta) to see caretta's capabilities. The server only allows alignment of up to 50 proteins at once.
The command-line tool and self-hosted web application do not have this restriction.

## Installation

### Requirements
Caretta works with Python 3.7+
Run the following commands to install required external dependencies:
```bash
conda install -c salilab dssp
conda install -c bioconda msms
```

### Download caretta
```bash
git clone https://git.wur.nl/durai001/caretta.git
cd caretta
```

### Installing both the command-line interface and the web-application:
```bash
pip install -e .[GUI]
cd bin
chmod +x caretta-cli
chmod +x caretta-app
```

### Installing only the command-line interface:
```bash
pip install -e .
cd bin
chmod +x caretta-cli
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
# caretta -h for more options
```

### Web-application Usage

```bash
caretta-app <host-ip> <port> 
# e.g. caretta-app localhost 8091
```

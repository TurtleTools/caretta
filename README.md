# Caretta

## Installation

```bash
conda install -c salilab dssp
conda install -c bioconda msms
git clone https://git.wur.nl/durai001/caretta.git
cd caretta
pip install .
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=20 # change to required number of threads

cd bin
chmod +x caretta-cli
chmod +x carettaapp
```

## Command-line Usage

```bash
caretta-cli input_pdb_folder
# caretta -h for more options
```

## Using Carettapp (web-app)

```bash
carettapp <host-ip> <port> 
```

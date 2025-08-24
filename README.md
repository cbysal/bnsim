# BN-Sim - Blockchain Network Simulator

This is a blockchain network simulator for ECCB [1].

## Project Structure
| File | Description |
| :-: | - |
| main.go | main logic of BN-Sim |
| go.mod, go.sum | go module dependencies |
| network.py | logic to generate network topology, which is embedded in the executable binary |
| graph.py | logic to plot figures, which is embedded in the executable binary |
| blocks.rlp.zst | blocks from heights 19,130,000 to 19,140,000, which is encoded in RLP and compressed with ZSTD |
| eccb{1,2}.log.zst | hitting transactions recorded by two neighboring nodes, each line in the form **[[block height],[transaction hash]*]**, compressed with ZSTD |

## Hardware Dependencies
### Minimal
- CPU: any x86-64 CPU with any number of cores
- RAM: 2 GB per CPU core
- Storage: 32 GB

### Recommended
- CPU: 16 cores
- RAM: 32 GB
- Storage: 64 GB

It is recommended to run the simulation on an [AliCloud](https://www.aliyun.com) server of type **ecs.sn1.3xlarge**.

### Software Dependencies
### Operating System
Debian sid and Ubuntu 24.04 are tested and recommended.

### APT Packages
- build-essential
- git-lfs
- golang (v1.22)
- python-is-python3
- python3
- python3-venv

### Python Packages
Listed in [requirements.txt](https://github.com/cbysal/eccbae/blob/master/requirements.txt).

## Setup
Install the APT packages:
```bash
cd ~ && apt update
apt install build-essential git-lfs golang python-is-python3 python3 python3-venv
```

Set up a Python virtual environment and install the Python packages:
```bash
python -m venv .venv
source .venv/bin/activate
wget https://raw.githubusercontent.com/cbysal/eccbae/refs/heads/master/requirements.txt
pip install -r requirements.txt
```

Clone this repository:
```bash
git clone https://github.com/cbysal/bnsim.git
```

## Build
```bash
make
```

## Run
### Preparation
```bash
./bnsim preprocess blocks.rlp.zst eccb1.log.zst eccb2.log.zst
```
This command converts the dataset into workloads: **result/block-info-[0-4].json**.

For additional usage: if you pass 2 parameters to the command, e.g.,
```bash
./bnsim preprocess eccb1.log.zst eccb2.log.zst
```
BN‑Sim reads blocks from the path specified by the **--datadir** option. The default path for **--datadir** is **~/.ethereum/geth/chaindata**. The logs can be in the form of raw, or compressed as \*.gz, \*.xz, or \*.zst. To produce your own eccb*.log files, modify the standard Geth client to record hitting transactions when a block arrives.

### Correction Factor
```bash
./bnsim simulate-correction-factor
```
This command performs the correction factor simulation described in ECCB [1]. Results are written to **result/simulate-correction-factor.csv** and **images/simulate-correction-factor.pdf**.

### Network Scale
```bash
./bnsim simulate-scalability
```

This command performs the network scale simulation in ECCB [1]. Results are written to **result/simulate-scalability-*.csv** and **images/simulate-scalability-*.pdf**.

### Bandwidth
```bash
./bnsim simulate-bandwidth
```
This command performs the bandwidth simulation described in ECCB [1]. Results are written to **result/simulate-bandwidth-*.csv** and **images/simulate-bandwidth-*.pdf**.

### Hitting Transaction Ratio
```bash
./bnsim simulate-similarity
```
This command performs the hitting transaction ratio simulation described in ECCB [1]. Results are written to **result/simulate-similarity-*.csv** and **images/simulate-similarity-*.pdf**.

## Troubleshooting
1. Why do I fail to build BN-Sim?

Please install the [software dependencies](#software-dependencies) and check if the Go version is 1.22.

2. Why do I fail to run the subcommands (simulate-*) or plot figures?

Please ensure the Python virtual environment is activated. If not, run the following command:
```bash
source ~/eccbae/.venv/bin/activate
```

## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

## References
[1] Bingyi Cai, Shenggang Wan, and Hong Jiang. *ECCB: Boosting Block Propagation of Blockchain with Erasure-Coded Compact Block*. EuroSys '26, ACM, 2026.

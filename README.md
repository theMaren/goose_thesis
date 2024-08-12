# <span style="font-weight:normal">**GOOSE**: **G**raphs **O**ptimised f**O**r **S**earch **E**valuation</span>

This repository hosts the code for the Master Thesis titled **"Graph Neural Networks for Learning Domain-Dependent Heuristics in Automated Planning: Enhancing the GOOSE Framework."** This work was conducted at the University of Padova as part of the Erasmus Mundus Master Program in Big Data Management and Analytics.

Given that the thesis focuses on enhancing the GOOSE framework, a significant portion of the code is derived from the original GOOSE framework. You can find the original GOOSE framework [here](https://github.com/DillonZChen/goose).

## Table of contents
- [**GOOSE**: **G**raphs **O**ptimised f**O**r **S**earch **E**valuation](#goose-graphs-optimised-for-search-evaluation)
  - [Table of contents](#table-of-contents)
  - [Setup](#setup)
  - [Training](#training)
  - [Heuristic Search](#heuristic-search)
  - [Retraining](#retraining)

## Setup
Use the commands below to make a virtual environment or a conda environment, activate it, install packages, and build cpp components.
The setup has been tested with python versions 3.10.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sh setup.sh
```

Setup with anaconda
```
conda create --name goose python=3.10.4
conda activate goose
pip install -r requirements.txt
sh setup.sh
```

## Training
- The training configurations for various problem domains, including the domain, task, and plans directory for each domain, are available in the `.toml` files located in `experiments/ipc23-learning`.
- The training configurations for the models, which define the model type, number of layers, graph type, and hidden units, can be found in the `.toml` files within `experiments/models`.
- Below are example commands for training a domain-dependent Blocksworld model. To train a model for a different domain, the path to the corresponding configuration file must be updated. The `save_file` parameter is optional but required if the model needs to be saved for later testing.
- If saving models for the RGAT models, it is important that the filename includes the term `rgat` to ensure they can be correctly executed later.


### Example for WL models (GOOSE_WL-GPR)
```
python3 train.py experiments/models/wl_ilg_gpr.toml experiments/ipc23-learning/blocksworld.toml --save-file blocksworld_wl.model
```

### Example for RGNN models (Goose_standard)
```
python3 train.py experiments/models/gnn_mean_ilg.toml experiments/ipc23-learning/blocksworld.toml --save-file blocksworld_gnn.model
```

### Example for RGAT models (Goose_gat)
```
python3 train.py experiments/models/rgat_max_ilg.toml experiments/ipc23-learning/blocksworld.toml --save-file blocksworld_rgat.model
```


## Heuristic Search
- Test the models by performing heuristic search.
- Example commands are provided for solving a single problem.
- There is a different run script for each heuristic: see `run_blind.py`, `run_hff.py`, `run_multi.py`, `run_gnn.py`, and `run_wl.py`.
- `run_blind.py` and `run_hff.py` do not require any previously trained models.
- `run_multi.py` is implemented to always use the provided model along with the HFF heuristic as a second heuristic.
- `run_gnn.py` is used for running both the GOOSE standard RGNN models and the RGAT models. It will automatically attempt to use the GPU if available.


### Example for blind search
```
python3 run_blind.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl
```

### Example for hff 
```
python3 run_hff.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl
```

### Example for RGNN models (GOOSE_standard)
```
python3 run_gnn.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl blocksworld_gnn.model
```

### Example for RGAT models (GOOSE_gat)
```
python3 run_gnn.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl blocksworld_rgat.model
```

### Example for multiheuristic search
```
python3 run_multi.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl blocksworld_gnn.model
```

### Example for WL models
```
python3 run_wl.py benchmarks/ipc23-learning/blocksworld/domain.pddl benchmarks/ipc23-learning/blocksworld/testing/easy/p01.pddl blocksworld_wl.model
```

## Retraining


- retraining of the models for the heuristics xyu and cyxz
- the solved problems for the retraining need to be saved in retraining/solutions/{difficulty}/{domain} in .plan files
- the retraining scripts require the following parameters
  - --model: Path to the model that should be retrained
  - --domain: Problem domain for which the initial model was trained
  - --difficulty: Difficulty level of the problems on which the model should be retrained possible values are "medium" and "hard"
- Retraining will automatically attempt to use the GPU if available

### Seperate test dataset
- the seperate dataset that was generate for the retraining can be found under retraining/dataset_new
- the generation can be replicated using the generate_all.py scripts that can be found in benchmarks/ipc23-learning/{domain}. Setting the random seeds in the scripts to the following values
```
seeds = [2, 2008, 2008, 2008]
```

### Example for rgnn retraining 
```
python3 retrain_gnn.py --model blocksworld_gnn.model  --domain blocksworld --difficulty medium
```

### Example for rgat retraining
```
python3 retrain_rgat.py --model blocksworld_rgat.model  --domain blocksworld --difficulty medium
```


## References


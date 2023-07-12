# 2023_PDADS
This contains the supplementary code, notebooks, and data associated with our 
2023 PDADS publications

## Directories

* `config/` -- YAML config files used for the runs
* `notebooks/` -- Jupyter notebooks used for paper analytics
* `scripts/` -- Summit batch submission and visualization scripts
* `templates/` -- JSON file for DeepMD input used to set hyperparameters for 
  evaluations

## Files

* `decoder.py` -- defines LEAP `Decoder` subclass `DeepMDDecoder` from mapping 
  gene values into a phenome for evaluation
* `deepmd-tuner.py` -- main executable for experiments
* `individual.py` -- `DeepMDIndividual` is a subclass of LEAP 
  `DistributeIndividual` that overrides `evaluate()`
* `phenotype.py` -- defines the genes and their respective ranges for initial 
  random individuals
* `problem.py` -- LEAP `Problem` subclass used to implement fitness 
  evaluation for `DeepMDIndividual`s
* `reporting.py` -- defines `log_pop` and `log_ind` functions for capturing 
  run output
* `representation.py` -- defines a LEAP representation that dictates how 
  initial random populations are created, that `DeepMDIndividual` is to be 
  used, and `DeepMDDecoder` is the decoder to be used
* `README.md` -- this file

# Optimizing deepmd hyperparameters on Summit


## Directories

* `config/` -- YAML config files used for the runs
* `notebooks/` -- Jupyter notebooks used for paper analytics
* `scripts/` -- Summit batch submission and visualization scripts
* `templates/` -- JSON file for DeepMD input used to set hyperparameters for 
  evaluations

## Files

* `decoder.py` -- This defines `DeepMDDecoder`, which decodes the "genomes" of real-valued numbers into "phenomes" of DeePMD hyperparameters.
* `deepmd-tuner.py` -- The main script that drives the evolutionary algorithm.
* `individual.py` -- Defines `DeepMDIndividual`, which is a subclass of LEAP's `DistributedIndividual`. We do that to 
   override `DistributedIndividual`'s default behavior of assigning NaNs as 
  fitness for broken individuals; we assign MAXINT, instead. ("Broken" means 
  that there was an 
  exception thrown during evaluation, or some other problem.  This is likely 
  due to a weird combination of hyperparameters or a hardware error.)  We 
  needed to make this substitution to allow sorting of individuals work, 
  which is paramount for NSGA-II to work.  I.e., sorting individuals with 
  NaNs as fitnesses leads to undefined behavior.
* `phenotype.py` -- Defines what the individuals genes mean, and the valid 
  ranges for initializing them when starting with a random population.
* `problem.py` -- Defines `DeepMDProblem` that implements the mechanism of 
  calling DeePMD to evaluate an individual.
* `reporting.py` -- Defines logging functions for writing run results to CSV 
  files.
* `representation.py` -- Defines `DeepMDRepresentation` that just connects 
  how to initialize individuals, decode them, and what base class for 
  `Individual` to use.

# YAML config files for deepmd-tuner.py

These contain config files that control how experiments using `deepmd-tuner.
py` run. These configuration files can be "stacked" in that multiple YAML 
files can be given as command line arguments to `deepmd-tuner.py`; during a 
run, all the configuration parameters are merged together.

E.g., `deepmd-tuner.py general.yaml summit.yaml` may set up a run for Summit.
Adding `debug.yaml` will set up for a very short test run.

* `debug.yaml` -- sets the number of generations to two for debugging
* `general.yaml` -- contains parameters common to all runs
* `local.yaml` -- for running on local machine
* `summit.yaml` -- for Summit runs
* `test.yaml` -- sets `test` to true; this means returning two random values 
  for a fitness instead of doing the the actual DeepMD training, and is 
  used to shake out general problems with the EA.

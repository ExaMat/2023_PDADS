#!/usr/bin/env python3
"""
    Hyperparameter tuning for deepmd-kit via an evolutionary algorithm
    implemented with LEAP.
"""
import argparse
import logging
import os
import sys
import random
import numpy as np
from time import time

from toolz import pipe

from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, \
    ConfigAttributeError
from omegaconf import OmegaConf

from rich.console import Console
from rich.logging import RichHandler

# Create unique logger for this namespace
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True)
logging.basicConfig(level='DEBUG', format='%(message)s',
                    datefmt="[%Y/%m/%d %H:%M:%S]",
                    handlers=[rich_handler])
logger = logging.getLogger(__name__)

from rich.table import Table
from rich import print
from rich import pretty

pretty.install()

from rich.traceback import install

install()

import dask
from distributed import Client, LocalCluster

import leap_ec.ops as ops
import leap_ec.util as util


from leap_ec.ops import context
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.global_vars import context
from leap_ec.distrib import synchronous
from leap_ec.multiobjective.ops import rank_ordinal_sort, \
    crowding_distance_calc

from leap_ec.distrib import asynchronous
from leap_ec.distrib.synchronous import eval_pool
from leap_ec.distrib.logger import WorkerLoggerPlugin

from representation import DeepMDRepresentation
from problem import DeepMDProblem
from reporting import log_pop, log_worker_location


DESCRIPTION = """
Hyperparameter tuning for deepmd-kit via an evolutionary algorithm 
implemented with LEAP.
"""

CLIENT_NAME = __name__


def read_config_files(config_files):
    """  Read one or more YAML files containing configuration options.

    The notion is that you can have a set of YAML files for controlling the
    configuration, such as having a set of default global settings that are
    overridden or extended by subsequent configuration files.

    E.g.,

    deepmd-turner.py general.yaml summit_config.yaml this_run.yaml

    :param config_files: command line arguments
    :return: config object of current config
    """
    serial_configs = [OmegaConf.load(x) for x in config_files]
    config = OmegaConf.merge(*serial_configs)

    return config


def setup_dask_client(config):
    """ Set up dask client for either localhost or on a cluster
    :param config: run-time configuration parameters
    """
    if 'distributed' in config and 'scheduler_file' in config.distributed:
        # We're wanting to submit workers onto other nodes, and *not* run
        # them locally because we went through the trouble of specifying
        # a scheduler file that the scheduler and workers will use to
        # coordinate with one another.
        logger.info('Using a remote distributed model')
        logger.info(f'Using scheduler file {config.distributed.scheduler_file}')

        client = Client(scheduler_file=config.distributed.scheduler_file,
                        timeout=config.distributed.scheduler_timeout,
                        name=CLIENT_NAME)
    else:
        logger.info('Using a local distributed model')
        if 'distributed' in config and 'num_workers' in config.distributed:
            logger.info('Using config file specified number of workers')
            workers = config.distributed.num_workers
        else:
            logger.info('Using number of cores for workers')
            workers = os.cpu_count()

        logger.info(f'Using {workers} dask workers')

        cluster = LocalCluster(n_workers=workers,
                               processes=True,
                               threads_per_worker=1,
                               silence_logs=logger.level)
        logger.info("Cluster: %s", cluster)
        client = Client(cluster,
                        timeout=config.get('distributed.scheduler_timeout', 30))

    return client


def get_num_workers(client):
    """
    :param client: active dask client
    :return: the number of workers registered to the scheduler
    """
    scheduler_info = client.scheduler_info()

    return len(scheduler_info['workers'].keys())


def wait_for_workers(config):
    """ Optionally wait for a certain number of workers before proceeding

    :param config: configuration parameters
    """
    # Optionally wait for the number of workers to spin up before starting
    try:
        if not OmegaConf.is_missing(config,
                                    'distributed.optional_num_wait_for_workers'):
            logger.info(
                f'Waiting for {config.distributed.optional_num_wait_for_workers}'
                f' workers before proceeding')
            start_wait_for_workers = time()
            client.wait_for_workers(
                int(config.distributed.optional_num_wait_for_workers))
            logger.info(f'Took {(time() - start_wait_for_workers) / 60} minutes'
                        f' for workers to spin up.')
        else:
            logger.info(f'Not waiting on any workers, so going right in!')
    except (ConfigKeyError, ConfigAttributeError, MissingMandatoryValue):
        # No worries, optional_num_wait_for_workers not defined so we're just
        # going to keep on trucking.
        logger.info(f'Not waiting on any workers, so going right in!')


def run_ea(config, representation, problem, max_generations, context, client):
    """ Run the EA to optimize and train a deepmd model

        This uses NSGA-II to optimize for minimizing the energy and forces
        for a given deepmd model.

        :param config: the run-time configuration parameters
        :param representation: for each individual
        :param problem: for which we are trying to optimize
        :param max_generations: how many generations to run to?
        :param context: global context object to get current generation
        :param client: to an active Dask client
        :returns: Last generation of solutions (deepmd networks)
    """
    # Initialize a population of pop_size individuals of the same type as
    # individual_cls
    parents = representation.create_population(int(config.ea.pop_size),
                                               problem=problem)

    logger.debug(f'Creating initial random population')

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(context=context)

    logger.debug(f'About to evaluate initial random population')

    # Scatter the initial parents to dask workers for evaluation
    parents = synchronous.eval_population(parents, client=client)

    logger.debug(f'Finished evaluating initial random population')

    # Reporting setup
    # For taking snapshots of the population
    pop_probe_stream = open(config.ea.pop_csv_file, 'w')
    pop_probe = log_pop(job=config.job_id,
                        context=context,
                        stream=pop_probe_stream)

    # For taking snapshots of the offspring including initial population; i.e.,
    # *everyone* and not just the best
    evaluated_probe_stream = open(config.ea.ind_csv_file, 'w')
    evaluated_probe = log_worker_location(
        job=config.job_id,
        stream=evaluated_probe_stream)

    pop_probe(parents) # report on generation zero
    evaluated_probe(parents)

    context['std'] = np.array([0.001,  # start_lr
                               0.0001, # stop_lr
                               0.0625, # rcut
                               0.0625, # rcut smth
                               0.0625, # scale by worker
                               0.0625, # des activ func
                               0.0625, # fitting activ func
                               ])

    try:
        while generation_counter.generation() < max_generations:
            # Force flushing on Summit to see files
            pop_probe_stream.flush()
            evaluated_probe_stream.flush()

            generation_counter()  # Increment to the next generation

            offspring = pipe(parents,
                             # pipeline for user defined selection, cloning,
                             # mutation, and maybe crossover
                             ops.random_selection,
                             ops.clone,
                             mutate_gaussian(
                                 std=context['std'],
                                 expected_num_mutations='isotropic', # zap all genes
                                 hard_bounds=DeepMDRepresentation.bounds),
                             eval_pool(client=client, size=len(parents)),
                             evaluated_probe,
                             rank_ordinal_sort(parents=parents),
                             crowding_distance_calc,
                             ops.truncation_selection(size=len(parents),
                                                      key=lambda x: (-x.rank,
                                                                     x.distance)),
                             pop_probe
                             )

            parents = offspring  # Make offspring new parents for next generation

            print(f'Finished generation '
                        f'{generation_counter.generation()!s}')
            logger.info(f'Finished generation '
                        f'{generation_counter.generation()!s}')


            # .85 was an original annealing step size used by Hans-Paul
            # Schwefel, though this was in the context of the 1/5 success
            # rule, which we've not implemented here. Handbook of EC, B1.3:2
            context['std'] *= .85
            logger.info(f"New stds: {context['std']}")
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        # evaluated_probe_stream.close()
        pop_probe_stream.close()

    return parents




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config_files', nargs='+',
                        help='One or more YAML config files')

    args = parser.parse_args()

    config = read_config_files(args.config_files)

    if 'verbose' in config and config.verbose:
        # print(dask.config.config, '\n\n')
        pretty.pprint(OmegaConf.to_container(config, resolve=True))

    # By default we're not testing the EA
    test_mode = False

    if 'test' in config and config.test:
        test_mode = True

    if 'seed' in config.ea:
        logger.info(f'Setting seed to {config.ea.seed}.')
        random.seed(int(config.ea.seed))

    # Ensure all output files and dask droppings are created in the run_dir by
    # setting our working directory there first thing.
    os.chdir(config.run_dir)

    client = setup_dask_client(config)
    client.register_worker_plugin(WorkerLoggerPlugin(verbose=True))

    # Wait for a certain number of dask workers to spin up before proceeding
    wait_for_workers(config)

    logger.info(f'Starting with {get_num_workers(client)} dask workers')

    # Use NSGA-II to optimize deepmd models for minimizing energies and forces
    final_pop = run_ea(config,
                       DeepMDRepresentation(),
                       DeepMDProblem(config.run_dir,
                                     config.input_template,
                                     timeout=config.ea.training_timeout,
                                     verbose=config.verbose,
                                     test=test_mode),
                       config.ea.max_generations,
                       context,
                       client)

    logger.info(f'Finished with {get_num_workers(client)} dask workers')

    print(f'Final pop:')
    pretty.pprint(final_pop)

    print('End')

#!/usr/bin/env python3
"""
    This is where we define how decode each individual to then pass the
    corresponding hyperparameter settings to the evaluator to determine the
    fitness.

    Note that WaterProblem and CopperProblem differed in that the latter did
    not have stop_lr.  However, updating the input.json template from what the
    Cu folks gave us brought that back in, thus making the two representations
    identical.  But I'm keeping the two representations the same for now given
    that these two can potentially greatly diverge as we get a deeper
    understanding on how their software works.
"""
import os
import random
import subprocess

import sys
from pathlib import Path
from string import Template
from subprocess import CalledProcessError

import numpy as np
from numpy import nan, isnan
from rich import pretty
from rich import print

pretty.install()

from rich.traceback import install

install()

from distributed import get_worker

# from leap_ec.problem import ScalarProblem
from leap_ec.multiobjective.problems import MultiObjectiveProblem

class DeepMDProblem(MultiObjectiveProblem):
    """
        deepmd-kit hyperparameter tuning for the water example
    """

    BAD_FITNESS = np.iinfo(np.int32).max # how we flag bad fitness values

    COMMAND_STR = ['jsrun', '--smpiargs="-gpu"',
                   '-e', 'individual', '--stdio_stdout=worker_out.%j.%h.%p',
                   '--stdio_stderr=worker_error.%j.%h.%p',
                   '-n', '1',
                   '-a', '6',
                   '-c', '40',
                   '-g', '6',
                   '-b', 'none', '${BIND}',
                   '--latency_priority', 'gpu-cpu',
                   'env', 'OMP_NUM_THREADS=1', # 'TF_NUM_INTRAOP_THREADS=1',
                   # 'TF_NUM_INTEROP_THREADS=7',
                   'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5',
                   'dp', 'train', '--skip-neighbor-stat', 'input.json']

    def __init__(self, run_dir, template, timeout=None, verbose=True, test=False):
        """
        :param run_dir: top-level directory in which the main process/script is
            running
        :param template: deepmd-kit input JSON file template
        :param timeout: how long, in minutes, do we allow the training
            subprocess to run before we give up on it?
        :param verbose: boolean for chatty run-time behavior during eval
        :param test: if true, don't actually invoke dp, but return random fitnesses to test
            the overall EA process
        """
        # This is a _minimization_ problem in that we're minimizing the
        # error loss.
        super().__init__(maximize=[False, False])

        self.run_dir = run_dir
        self.template = template
        self.timeout = timeout
        self.verbose = verbose
        self.test = test

    def check_phenome(self, phenome):
        """ Semantic checking for phenome.
        """
        assert phenome.start_lr > phenome.stop_lr
        assert phenome.rcut_smth < phenome.rcut

    def create_input_json(self, phenome):
        """ Create input.json based on phenome.
        Intended to be overriden by subclasses """
        with open(self.template, 'r') as template_file:
            template = Template(template_file.read())
            out_str = template.substitute(
                start_lr=phenome.start_lr,
                stop_lr=phenome.stop_lr,
                rcut=phenome.rcut,
                rcut_smth=phenome.rcut_smth,
                scale_by_worker=phenome.scale_by_worker,
                desc_activ_func=phenome.desc_activ_func,
                fitting_activ_func=phenome.fitting_activ_func,
                seed=random.randrange(sys.maxsize))
        return out_str

    def evaluate(self, phenome, uuid):
        """
        Evaluate the given individual's phenome by running deepmd-kit with those
        parameters substituted in a corresponding input.json file.  The uuid is
        necessary to create a sub-dir named after the uuid to create all the
        input and output files.

        Phenome composition:
        gene 0, float, start learning rate
        gene 1, float, stop learning rate
        gene 2, float, decay steps

        Note that this a loose-coupling approach in that we shell out to
        invoke `dp` and then after it's done, we read and slurp in the
        curve output file and return the

        :param phenome: [learning rate]
        :param uuid: UUID bound the individual
        :return: force rmse for last batch training value
        """
        if phenome is None:
            # More than likely a decoder error occurred
            raise ValueError('phenome was none likely due to decoder error')

        if self.test:
            # Return two random fitnesses so that we can exercise the overall EA process to shake out bugs
            return np.random.uniform(size=(2,))

        fitness = np.array((DeepMDProblem.BAD_FITNESS, DeepMDProblem.BAD_FITNESS))

        worker = get_worker()
        worker.logger.info(f'Starting evaluate() for {uuid} with '
                           f'genome {phenome!s}')

        # First change to the top-level run-dir to ensure that we're in
        # the correct starting directory location.  If we didn't do this
        # there's a chance that all the UUID subdirs would be nested within
        # one another.
        os.chdir(self.run_dir)

        # The starting learning rate should always be higher than the stopping;
        # if this throws an exception, then this individual is flagged by LEAP
        # as being invalid. This just means the individual still exists, but
        # will always lose against a valid individual for selection; if
        # compared against another invalid individual, then selection will just
        # randomly pick one.
        worker.logger.debug('Before check phenome')
        self.check_phenome(phenome)

        # Create subdir in which we'll write all output; the name will
        # be the UUID for this individual so that we can later cross-
        # reference the EA CSV output that has records of all individuals
        # with their respective output directories.  We should *NEVER* write
        # to an existing UUID; doing so indicates a possible error, hence
        # exist_ok=False.
        cwd = Path('.').absolute()
        new_subdir = cwd / str(uuid)
        new_subdir.mkdir(parents=True, exist_ok=False)

        # Now change into that directory so that everything we do is
        # written there.
        os.chdir(new_subdir)

        worker.logger.debug(f'Now in cwd: {os.getcwd()}')

        # Read and update the JSON input template with the hyperparameter
        # values associated with this individual.
        out_str = self.create_input_json(phenome)

        with open('input.json', 'w') as input_json:
            input_json.write(out_str)

        worker.logger.debug('Wrote input.json')

        # Then shell out and run `dp` pointing it to the input JSON file
        # we generated from the template.
        worker.logger.info(f'About to run for UUID {uuid}')
        completed_process = subprocess.run(' '.join(DeepMDProblem.COMMAND_STR),
                                           shell=True,
                                           capture_output=True,
                                           # convert to seconds
                                           timeout=int(self.timeout) * 60,
                                           check=False)
        worker.logger.info(f'Finished run for UUID {uuid}')

        if hasattr(completed_process, 'stdout'):
            print(completed_process.stdout, file=sys.stdout, flush=True)
            print(completed_process.stderr, file=sys.stderr, flush=True)

            worker.logger.info(completed_process.stdout)
            worker.logger.info(completed_process.stderr)

        if hasattr(completed_process, 'returncode') and \
                completed_process.returncode != 0:
            worker.logger.warning(f'Training failed.  Return '
                                  f'code {completed_process.returncode}')
        else:
            # If `dp` ran successfully, slurp and and return the force rmse as
            # the fitness; if it didn't run correctly, throw an exception so
            # that LEAP will flag this as an "invalid" individual.  An invalid
            # individual has no corresponding fitness, and indicates that
            # something went wrong such that the individual couldn't be
            # evaluated.  This usually means that there was something wrong
            # with the posed configuration, such as a deep-learner constraint
            # violation.  (And which is actually OK, especially at the beginning
            # of a run when all the individuals are just random.  You're likely
            # to have weird configurations that pytorch/tensorflow will puke on,
            # and that's fine.  Eventually evolution will cull those constraint
            # violations.)

            # If all ran ok, then deepmd-kit should have written all the data
            # to lcurve.out.
            if not Path('lcurve.out').exists():
                # Sadly, for some reason, deepmd will just wedge and not run at all
                # on Summit, thus producing no lcurve.out.
                worker.logger.error('No lcurve file.')
                print(f'lcurve.out does not exist. cwd: {os.getcwd()}',
                      file=sys.stderr, flush=True)
            else:
                data = np.genfromtxt("lcurve.out", names=True)

                # return the last validation data point for the force error as the
                # fitness
                last_rmse_e_val = data['rmse_e_val'][-1]
                last_rmse_f_val = data['rmse_f_val'][-1]
                fitness = np.array((last_rmse_e_val, last_rmse_f_val))
                worker.logger.info(f'fitness is {fitness!s}')

        os.chdir(cwd)  # change back to rundir
        worker.logger.debug(f"Now cwd back to: {os.getcwd()}")

        if np.isnan(fitness).any() or np.equal(fitness, DeepMDProblem.BAD_FITNESS).any():
            # Raise an exception so that LEAP can make this individual
            # formally not viable to force creating a new offspring in its
            # place.
            raise ValueError(f'Unable to evaluate individual {uuid} with fitness {fitness}')

        return fitness

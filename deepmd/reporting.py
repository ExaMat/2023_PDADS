#!/usr/bin/env python3
"""
    Functions for reporting evaluating individuals and populations
"""
import csv
import sys
import traceback
import traceback

from leap_ec.global_vars import context

from representation import DeepMDRepresentation
from phenotype import PhenotypeBounds
from individual import DeepMDIndividual


# TODO convert to by-generation
def log_pop(job, context, stream=sys.stdout, header=True):
    """ Log the population to a CSV file for a given interval.

    (Lifted from leap_ec.distributed.log and hacked to add scenario column.)

    :param job: which job is this in a set of jobs?
    :param context: from which to get the current generation
    :param stream: open stream to which to write rows
    :param header: True if we want a header for the CSV file
    :return: a function for saving regular population snapshots
    """
    job = job
    stream = stream

    # We just want to splice in the phenotypic file names in the middle of the
    # CSV file.  Doing it this way allows us to gradually add new phenotypic
    # fields in one place and have them automatically show up elsewhere.
    fieldnames = ['job', 'generation', 'uuid', 'birth_id']
    fieldnames.extend(PhenotypeBounds._fields)
    fieldnames.extend(['start_eval_time', 'stop_eval_time', 'energy_fitness',
                       'force_fitness'])

    writer = csv.DictWriter(stream, fieldnames=fieldnames)

    if header:
        writer.writeheader()
        stream.flush()

    def write_pop_update(population):
        """

        :param population: to be written to stream
        :return: None
        """
        nonlocal stream
        nonlocal writer

        for individual in population:

            try:
                phenome = individual.decoder.decode(individual.genome)
            except Exception as e:
                traceback.print_exc()

            if type(individual.fitness) == float:
                # Even though the individuals *should* be a tuple of floats, it
                # may end up just being a float, likely a NaN if an exception
                # was thrown during evaluation.  If this is the case, we then
                # correct the error here by reassigning that single value to
                # a tuple as originally intended.
                individual.fitness = (individual.fitness, individual.fitness)

            writer.writerow({'job'                  : job,
                             'generation'           : context['leap']
                                                             ['generation'],
                             'uuid'                 : individual.uuid,
                             'birth_id'             : individual.birth_id,
                             'start_lr'             : phenome.start_lr,
                             'stop_lr'              : phenome.stop_lr,
                             'rcut'                 : phenome.rcut,
                             'rcut_smth'            : phenome.rcut_smth,
                             'scale_by_worker'      : phenome.scale_by_worker,
                             'desc_activ_func'      : phenome.desc_activ_func,
                             'fitting_activ_func'   : phenome.fitting_activ_func,
                             'start_eval_time'      : individual.start_eval_time,
                             'stop_eval_time'       : individual.stop_eval_time,
                             'energy_fitness'       : individual.fitness[0],
                             'force_fitness'        : individual.fitness[1]
                             })

        # On some systems, such as Summit, we need to force a flush else there
        # will be no output until the very end of the job.
        stream.flush()

        return population

    return write_pop_update


def log_worker_location(job, stream=sys.stdout, header=True):
    """
    When debugging dask distribution configurations, this function can be used
    to track what machine and process was used to evaluate a given
    individual

    Suitable for being passed as the `evaluated_probe` argument for
    leap_ec.distributed.asynchronous.steady_state().

    :param job: which job is this in a set of jobs?
    :param stream: to which we want to write the machine details
    :param header: True if we want a header for the CSV file
    :return: a function for recording where individuals are evaluated
    """
    job = job
    stream = stream
    # We just want to splice in the phenotypic file names in the middle of the
    # CSV file.  Doing it this way allows us to gradually add new phenotypic
    # fields in one place and have them automatically show up elsewhere.
    fieldnames = ['job', 'hostname', 'pid', 'uuid', 'birth_id']
    fieldnames.extend(PhenotypeBounds._fields)
    fieldnames.extend(['start_eval_time', 'stop_eval_time', 'energy_fitness',
                       'force_fitness'])

    writer = csv.DictWriter(stream, fieldnames=fieldnames)

    if header:
        writer.writeheader()
        stream.flush()

    def write_records(population):
        """ This writes a row to the CSV for the given individual

        evaluate() will tack on the hostname and pid for the individual.  The
        uuid should also be part of the distributed.Individual, too.

        :param population: to be written to stream
        :return: None
        """
        nonlocal stream
        nonlocal writer

        for individual in population:

            # The first element of the returned phenome after decoding is the
            # scenario number for the corresponding scenario dictionary stored
            # in the dask-worker that was initially read from a scenario YAML
            # configuration file.
            try:
                phenome = individual.decoder.decode(individual.genome)
            except Exception as e:
                traceback.print_exc()

            if type(individual.fitness) == float:
                # Even though the individuals *should* be a tuple of floats, it
                # may end up just being a float, likely a NaN if an exception
                # was thrown during evaluation.  If this is the case, we then
                # correct the error here by reassigning that single value to
                # a tuple as originally intended.
                individual.fitness = (individual.fitness, individual.fitness)

            writer.writerow({'job'                  : job,
                             'hostname'             : individual.hostname,
                             'pid'                  : individual.pid,
                             'uuid'                 : individual.uuid,
                             'birth_id'             : individual.birth_id,
                             'start_eval_time'      : individual.start_eval_time,
                             'stop_eval_time'       : individual.stop_eval_time,
                             'start_lr'             : phenome.start_lr,
                             'stop_lr'              : phenome.stop_lr,
                             'rcut'                 : phenome.rcut,
                             'rcut_smth'            : phenome.rcut_smth,
                             'scale_by_worker'      : phenome.scale_by_worker,
                             'desc_activ_func'      : phenome.desc_activ_func,
                             'fitting_activ_func'   : phenome.fitting_activ_func,
                             'energy_fitness'       : individual.fitness[0],
                             'force_fitness'        : individual.fitness[1]})
            # On some systems, such as Summit, we need to force a flush else there
            # will be no output until the very end of the job.
            stream.flush()

        return population

    return write_records

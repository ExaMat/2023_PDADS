#!/usr/bin/env python3
"""
    DistributedIndividual subclass to allow for using UUIDs to create sub-
    directories to save output for the given individual
"""
import numpy as np
from leap_ec.distrib.individual import DistributedIndividual

from problem import DeepMDProblem


class DeepMDIndividual(DistributedIndividual):

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)
        self.fitness = (None, None) # After eval: (rmse_e_val, rmse_f_val)

    def evaluate_imp(self):
        """ We override Individual.evaluate_imp() to pass in the UUID
        """
        return self.problem.evaluate(self.decode(), uuid=self.uuid)

    def evaluate(self):
        """ determine this individual's fitness

        Over-riding RobustIndividual.evaluate() to ensure that fitness is set
        to (self.problem.BAD_FITNESS, self.problem.BAD_FITNESS) instead of
        NaN to maintain that the fitness is now a tuple.  We also use
        self.problem.BAD_FITNESS to ensure that NSGA-II rank sorting works
        because we're not sure how NaNs sort.

        :return: the calculated fitness
        """
        try:
            self.fitness = self.evaluate_imp()
            self.is_viable = True  # we were able to evaluate
        except Exception as e:
            self.fitness = np.array((DeepMDProblem.BAD_FITNESS, DeepMDProblem.BAD_FITNESS))
            self.exception = e
            self.is_viable = False  # we could not complete an eval

        # Even though we've already *set* the fitness, it may be useful to also
        # *return* it to give more options to the programmer for using the
        # newly evaluated fitness.
        return self.fitness

    def __str__(self):
        phenome = self.decode()
        if hasattr(self, 'exception'):
            e = str(self.exception)
        else:
            e = ''
        return f'{self.birth_id}, {self.uuid}, "{phenome}", {self.fitness!s}, e'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    pass

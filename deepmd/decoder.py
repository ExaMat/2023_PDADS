#!/usr/bin/env python3
""" For decoding the genes into phenomes (or values that mean something)
"""
from math import floor
from phenotype import PhenotypeBounds

from leap_ec.decoder import Decoder



class DeepMDDecoder(Decoder):
    """ se_e2_a decoder """

    def __init__(self):
        super().__init__()

    @classmethod
    def wrap(cls, min, max, value):
        """ Ensure that `value` is within [min,max]

            Instead of clamping to bounds by truncating to the nearest bound,
            use MOD operator to "wrap" values, thus allowing mutation to freely
            move without introducing a boundary bias.

            >>> [MNISTDecoder.wrap(1,6, x) for x in range(16)]
            [6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3]


            TODO: consider adding to leap_ec.utils.
        """
        return (value - min) % (1 + max - min) + min


    @classmethod
    def _map_to_scale_by_worker(cls, gene):
        """ map gene value to ["linear", "sqrt", "none"] """
        values = ["linear", "sqrt", "none"]
        i = floor(gene) % len(values) # use % to wrap values > 2 to valid index
        return values[i]

    @classmethod
    def _map_to_activ_func(cls, gene):
        """ map gene value to [“relu”, “relu6”, “softplus”, “sigmoid”,
            “tanh”, “gelu”, “gelu_tf”]
        """
        # remove gelu since that appears to cause problems
        # values = ['relu', 'relu6', 'softplus', 'sigmoid',  'tanh', 'gelu', 'gelu_tf']
        values = ['relu', 'relu6', 'softplus', 'sigmoid',  'tanh']
        i = floor(gene) % len(values) # use % to wrap values > 2 to valid index
        return values[i]


    def decode(self, genome, *args, **kwargs):
        """ decode the given individual

        :param genome: gene values for an individual
        :returns: named tuple of phenotypic traits
        """
        i = 0
        start_lr = float(genome[i])
        i += 1
        stop_lr = float(genome[i])
        i += 1
        rcut_smth = float(genome[i])
        i += 1
        rcut = float(genome[i])
        i += 1
        scale_by_worker = DeepMDDecoder._map_to_scale_by_worker(float(genome[i]))
        i += 1
        desc_activ_func = DeepMDDecoder._map_to_activ_func(float(genome[i]))
        i += 1
        fitting_activ_func = DeepMDDecoder._map_to_activ_func(float(genome[i]))

        phenome = PhenotypeBounds(start_lr=start_lr,
                                  stop_lr=stop_lr,
                                  rcut=rcut,
                                  rcut_smth=rcut_smth,
                                  scale_by_worker=scale_by_worker,
                                  desc_activ_func=desc_activ_func,
                                  fitting_activ_func=fitting_activ_func,
                                  )
        return phenome

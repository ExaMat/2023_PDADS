#!/usr/bin/env python3
"""
    Defines the representation for individuals used by deepmd-tuner.py
"""
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.representation import Representation

from decoder import DeepMDDecoder
from individual import DeepMDIndividual
from phenotype import PhenotypeBounds


class DeepMDRepresentation(Representation):
    """ Encapsulates deepmd-kit internals
    """

    bounds = PhenotypeBounds()

    def __init__(self):
        super().__init__(
            initialize=create_real_vector(DeepMDRepresentation.bounds),
            decoder=DeepMDDecoder(),
            individual_cls=DeepMDIndividual)

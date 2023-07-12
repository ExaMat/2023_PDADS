#!/usr/bin/env python3
""" Describes the phenotype for the representation.

    That is, this describes each gene and their respective bounds.
"""
from collections import namedtuple

__GENES = {
    "start_lr": (3.51e-8, 0.01),  # start learning rate
    "stop_lr": (3.51e-8, 0.0001),  # stop
    "rcut_smth": (2.0, 6.0),
    "rcut": (6.0, 12.0),
    "scale_by_worker": (0.0, 3.0), # maps to [0,1,2] ->  ["linear", "sqrt", "none"]
    "desc_activ_func": (0.0, 5.0), # maps to [0-6] -> [“relu”, “relu6”, “softplus”, “sigmoid”, “tanh”]
    "fitting_activ_func": (0.0, 5.0), # maps to [0-6] -> [“relu”, “relu6”, “softplus”, “sigmoid”, “tanh”]
}

PhenotypeBounds = namedtuple(
    "Phenotype", list(__GENES.keys()), defaults=list(__GENES.values())
)

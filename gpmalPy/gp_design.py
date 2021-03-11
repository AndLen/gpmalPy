import itertools

import numpy as np
from deap import gp
from deap.creator import _numpy_array

from gptools.gp_util import np_many_add, np_sigmoid, np_relu, erc_array, np_if


def get_pset_weights(num_features, rundata):
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(_numpy_array, num_features), _numpy_array, "f")

    pset.addPrimitive(np.add, [_numpy_array, _numpy_array], _numpy_array, name="add")
    pset.addPrimitive(np.multiply, [_numpy_array, _numpy_array], _numpy_array, name="mul")
    pset.addPrimitive(np_many_add, [_numpy_array, _numpy_array, _numpy_array, _numpy_array, _numpy_array], _numpy_array,
                      name="many_add")

    pset.addPrimitive(np_sigmoid, [_numpy_array], _numpy_array, name="sigmoid")
    pset.addPrimitive(np_relu, [_numpy_array], _numpy_array, name="relu")

    pset.addPrimitive(np.maximum, [_numpy_array, _numpy_array], _numpy_array, name="max")
    pset.addPrimitive(np.minimum, [_numpy_array, _numpy_array], _numpy_array, name="min")
    pset.addPrimitive(np_if, [_numpy_array, _numpy_array, _numpy_array], _numpy_array, name="np_if")

    # deap you muppet
    pset.context["array"] = np.array
    num_ercs = num_features
    # so we get as many as we do terms...
    if rundata.use_ercs:
        print("Using {:d} ERCS".format(num_ercs))
        for i in range(num_ercs):  # range(num_features):
            pset.addEphemeralConstant("rand", erc_array, _numpy_array)

    return pset

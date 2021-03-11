import numpy as np
from jax.interpreters.xla import DeviceArray
from numba import jit, njit

# A global dictionary storing the variables passed from the initializer.
from scipy.stats import norm

var_dict = {}

STD_DEV_FOR_PENALTY = 20.


def init_worker(num_trees, data_T_flat, data_T_shape, ordering, identity_ordering):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['num_trees'] = num_trees
    var_dict['data_T'] = np.frombuffer(data_T_flat).reshape(data_T_shape)
    var_dict['ordering'] = ordering
    var_dict['identity_ordering'] = identity_ordering
    nD = norm(loc=0, scale=STD_DEV_FOR_PENALTY)
    n_instances = var_dict['data_T'].shape[1]
    gaussian_penalties = np.ndarray((n_instances,))
    for i in range(n_instances):
        # print(i)
        bounds = i + 1
        gaussian_penalties[i] = nD.cdf(bounds) - nD.cdf(-bounds)
    var_dict['gaussian_penalties'] = gaussian_penalties
    # print(gaussian_penalties)


def worker_func(func_str, tree_compiler):
    dat_array = evalTrees(var_dict['data_T'], func_str, tree_compiler, var_dict['num_trees'])
    identity_ordering = var_dict['identity_ordering']

    cost, ratio_uniques, ordering, ordering_dists = eval_similarity_gaussian(var_dict['ordering'], identity_ordering,
                                                                             dat_array, var_dict['gaussian_penalties'])
    return cost, ordering, ordering_dists, dat_array


def evalTrees(data_t, func_str, tree_compiler, num_trees):
    num_instances = data_t.shape[1]

    result = np.zeros(shape=(num_trees, num_instances))
    # TODO can this be matrixy?
    for i, f in enumerate(func_str):
        # Transform the tree expression in a callable function
        func = tree_compiler(expr=f)
        comp = func(*data_t)
        if (not isinstance(comp, (np.ndarray, DeviceArray))) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)

        result[i] = comp
        # result.append(comp)

    dat_array = result.T
    return dat_array


@njit(fastmath=True)
def weight_discrepancy(diff, gaussian_penalties):
    if diff == 0:
        return 1
    elif diff > len(gaussian_penalties):
        return 0
    else:
        return 1 - gaussian_penalties[diff - 1]


@njit(fastmath=True)
def eval_similarity_gaussian(orderings, identity_ordering, dat_array, gaussian_penalties):
    sum = 0.
    n_instances = orderings.shape[0]
    n_neighbours = orderings.shape[1]

    # arr = deepcopy(dat_array)
    num_uniques = 0
    instance_pair_dists = np.empty((n_instances, n_neighbours), dtype=np.double)
    achieved_orderings = np.empty((n_instances, n_neighbours), dtype=np.int32)
    for i in range(n_instances):
        array_a = dat_array[i]
        array_b = dat_array[orderings[i]]
        for j in range(n_neighbours):
            instance_pair_dists[i][j] = np.linalg.norm(array_a - array_b[j])
        distincts = len(np.unique(instance_pair_dists[i]))
        num_uniques = num_uniques + (distincts / n_neighbours)

        achieved_ordering = np.argsort(instance_pair_dists[i])

        discrepencies = np.abs(identity_ordering - achieved_ordering)
        for i in range(n_neighbours):
            sum = sum + weight_discrepancy(discrepencies[i], gaussian_penalties)

        achieved_orderings[i] = achieved_ordering

    return (sum / (n_instances * n_neighbours)), num_uniques / n_instances, achieved_orderings, instance_pair_dists

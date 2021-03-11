import math
import time
from multiprocessing import RawArray

from deap import base, tools
from deap import creator
from deap.gp import genHalfAndHalf, genFull
from deap.tools import HallOfFame
from sklearn.metrics import pairwise_distances

from gpmalV2 import my_ea
from gpmalV2.eval_independent import init_worker
from gpmalV2.gp_design import get_pset_weights
from gpmalV2.rundata import rd
from gptools.ParallelToolbox import ParallelToolbox
from gptools.gp_util import *
from gptools.multitree import *
from gptools.util import init_data, final_output


def main():
    pop = toolbox.population(n=rd.pop_size)
    stats_cost = tools.Statistics(lambda ind: ind.fitness.values[0])
    mstats = tools.MultiStatistics(cost=stats_cost)
    mstats.register("min", np.min, axis=0)
    mstats.register("median", np.median, axis=0)
    mstats.register("max", np.max, axis=0)
    hof = HallOfFame(1)
    pop, logbook = my_ea.ea(pop, toolbox, rd.cxpb, rd.mutpb, rd.elitism, rd.gens, stats=mstats,
                            halloffame=hof, verbose=True)
    return pop, mstats, hof, logbook


def pick_nns(rd, step_length=10):
    i = 0
    indicies = []
    # this can probably just be a for loop if I derive the no. iterations instead of being lazy
    while True:
        base = step_length * ((2 ** i) - 1)
        step_multiplier = 2 ** i
        for j in range(step_length):
            next = base + (step_multiplier * j)
            # print(next)
            ##yeah yeah, it's easier...
            if next >= rd.num_instances:
                return indicies
            if next != 0:
                indicies.append(next)
        i += 1


def make_ind(toolbox, creator, num_trees):
    return creator.Individual([toolbox.tree() for _ in range(num_trees)])


if __name__ == "__main__":
    init_data(rd)
    pset = get_pset_weights(rd.num_features, rd)
    rd.pset = pset
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox = ParallelToolbox()  #

    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=0, max_=rd.max_depth)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", make_ind, toolbox, creator, rd.num_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", lim_xmate)

    toolbox.register("expr_mut", genFull, min_=0, max_=rd.max_depth)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)

    # GPMAL stuff.
    rd.pairwise_distances = pairwise_distances(rd.data)
    rd.ordered_neighbours = np.argsort(rd.pairwise_distances, axis=1)

    # get a list of indicies to use
    rd.neighbours = np.array(pick_nns(rd, step_length=10))
    rd.identity_ordering = np.array([x for x in range(len(rd.neighbours))])
    rd.all_orderings = rd.ordered_neighbours[:, rd.neighbours]
    print(rd.neighbours)
    assert math.isclose(rd.cxpb + rd.mutpb, 1), "Probabilities of operators should sum to ~1."

    print(rd)

    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes
    from multiprocessing import Pool

    threads = os.cpu_count()
    print("Using " + str(threads) + " threads")
    raw_trees = rd.num_trees
    raw_data_T_shape = rd.data_t.shape
    raw_data_T = RawArray('d', raw_data_T_shape[0] * raw_data_T_shape[1])
    raw_data_T_np = np.frombuffer(raw_data_T, dtype=np.float64).reshape(raw_data_T_shape)
    np.copyto(raw_data_T_np, rd.data_t)

    raw_orderings = rd.all_orderings
    raw_identity_ordering = rd.identity_ordering
    with Pool(processes=threads, initializer=init_worker,
              initargs=(raw_trees, raw_data_T, raw_data_T_shape, raw_orderings, raw_identity_ordering)) as rd.pool:
        toolbox.register("map", rd.pool.map)
        starttime = time.time()
        pop, stats, hof, logbook = main()
        endtime = time.time()
        print('Main Thread Complete , Total Time Taken = {}'.format(endtime - starttime))

    final_output(hof, toolbox, logbook, pop, rd)

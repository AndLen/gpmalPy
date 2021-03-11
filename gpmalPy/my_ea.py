from copy import deepcopy
from operator import attrgetter

from deap import tools
from deap.algorithms import varOr
from functools import partial

from gpmalV2.eval_independent import worker_func
from gptools.multitree import str_ind


def ea(population, toolbox, cxpb, mutpb, elitism, ngen, stats=None,
       halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(1, ngen + 1):
        population = eval_population(population, toolbox)
        sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)
        direct_elite = sorted_elite[:elitism]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        offspring = toolbox.select(deepcopy(population), len(population) - elitism)
        # Vary the pool of individuals
        offspring = varOr(offspring, toolbox, len(offspring), cxpb, mutpb)

        population[:elitism] = direct_elite
        population[elitism:] = offspring

    population = eval_population(population, toolbox)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=ngen, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    return population, logbook

def eval_population(population, toolbox):
    inds = [ind for ind in population]
    strs = [str_ind(ind) for ind in inds]
    for i, ind in enumerate(inds):
        ind.str = strs[i]
    results = toolbox.map(partial(worker_func, tree_compiler=toolbox.compile), strs)
    for ind, res in zip(inds, results):
        ind.fitness.values = res[0],
        ind.ordering = res[1]
        ind.ordering_dists = res[2]
        ind.output = res[3]

    inds = sorted(inds, key=attrgetter("fitness"),reverse=True)
    return inds


# https://pastebin.com/QKMhafRq
import copy
import random

from deap import gp

from gpmalV2.rundata import rd as rundata


def maxheight(v):
    return max(i.height for i in v)

# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > rundata.max_height:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def xmate(ind1, ind2):
    # if (random.random() < cxpb):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmate_st(ind1, ind2):
    # if (random.random() < cxpb):
    i1 = random.randrange(min(len(ind1), len(ind2)))
    # deepcopy needed? duplicates?
    ind1[i1], ind2[i1] = gp.cxOnePoint(copy.deepcopy(ind1[i1]), copy.deepcopy(ind2[i1]))
    return ind1, ind2


def lim_xmate_st(ind1, ind2):
    return wrap(xmate_st, ind1, ind2)


def xmate_addtrees(max_no, ind1, ind2):
    ind1_size = len(ind1)
    i1 = random.randrange(ind1_size)
    ind2_size = len(ind2)
    i2 = random.randrange(ind2_size)

    if ind1_size < max_no:
        ind1.append(copy.deepcopy(ind2[i2]))
    if ind2_size < max_no:
        ind2.append(copy.deepcopy(ind1[i1]))

    return ind1, ind2


def lim_xmate_aic(ind1, ind2):
    """
    Basically, keep only changes that obey max depth constraint on a tree-wise (NOT individual-wise) level.
    :param ind1:
    :param ind2:
    :return:
    """
    keep_inds = [copy.deepcopy(ind1), copy.deepcopy(ind2)]
    new_inds = list(xmate_aic(ind1, ind2))
    for i, ind in enumerate(new_inds):
        for j, tree in enumerate(ind):
            if tree.height > rundata.max_height:
                new_inds[i][j] = keep_inds[i][j]
    return new_inds


def xmate_aic(ind1, ind2):
    min_size = min(len(ind1), len(ind2))
    for i in range(min_size):
        ind1[i], ind2[i] = gp.cxOnePoint(copy.deepcopy(ind1[i]), copy.deepcopy(ind2[i]))
    return ind1, ind2


def xmate_maxt(ind1, ind2):
    max_size = max(len(ind1), len(ind2))
    i1 = random.randrange(max_size)
    i2 = random.randrange(max_size)

    if i1 >= len(ind1):
        # add one!
        ind1.append(copy.deepcopy(ind2[i2]))
    elif i2 >= len(ind2):
        # add one!
        ind2.append(copy.deepcopy(ind1[i1]))
    else:
        # normal crossover.
        ind1[i1], ind2[i2] = gp.cxOnePoint(copy.deepcopy(ind1[i1]), copy.deepcopy(ind2[i2]))

    return ind1, ind2


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


DO_ALL_ERCS = True

def str_ind(ind):
    return tuple(str(i) for i in ind)
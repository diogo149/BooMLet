import random
import numpy as np

from boomlet.utils.heap import ConstSizeHeap
from boomlet.utils.collection import grouper
from boomlet.utils.estimators import quick_score
from boomlet.parallel import pmap


def bits_to_char(bits):
    """ converts an iterable of 8 bits into a character
    """
    return chr(sum(bit * (1 << i) for i, bit in enumerate(bits)))


def bits_from_char(char):
    """ converts a character into 8 bits
    """
    num = ord(char)
    return [1 if num & (1 << i) else 0 for i in range(8)]


def bitmask_to_string(bitmask):
    """ converts a bitmask into a compressed string representation
    """
    chrs = map(bits_to_char, grouper(8, bitmask, 0))
    return "".join(chrs)


def bitmask_from_string(string):
    """ converts a string into a bitmask
    """
    bits = map(bits_from_char, string)
    return [bit for bit8 in bits for bit in bit8]


def bitmask_child(bitmask1, bitmask2):
    assert len(bitmask1) == len(bitmask2)
    return [bit1 if random.random() > 0.5 else bit2 for bit1, bit2 in zip(bitmask1, bitmask2)]


def bitmask_mutant(bitmask, avg_mutations=100):
    mutation_rate = float(avg_mutations) / len(bitmask)
    return [bit if random.random() > mutation_rate else 1 - bit for bit in bitmask]


def bitmask_evolve(bitmasks, avg_mutations=100, child_rate=0.5):
    if random.random() <= child_rate:
        return bitmask_child(*random.sample(bitmasks, 2))
    else:
        return bitmask_mutant(*random.sample(bitmasks, 1), avg_mutations=avg_mutations)


def random_bitmask(bits):
    return [random.randint(0, 1) for _ in range(bits)]


def bitmask_generation(bitmasks, history, bits, size=32, avg_mutations=100, child_rate=0.5):
    generation = []
    while len(generation) < size:
        if bitmasks:
            new_bitmask = bitmask_evolve(bitmasks, avg_mutations, child_rate)
        else:
            new_bitmask = random_bitmask(bits)
        bitmask_str = bitmask_to_string(new_bitmask)
        if bitmask_str not in history:
            generation.append(new_bitmask)
            history.add(bitmask_str)
    return generation


def bitmask_genetic_algorithm(bits, score_func, history=None, gene_pool=None, epochs=100, population_size=32, avg_mutations=100, child_rate=0.5, verbose=False, random_state=None):
    """ genetic algorithm that MAXIMIZES the value of the input scoring function
    """
    if random_state is not None:
        random.seed(random_state)
    if history is None:
        history = set()
    if gene_pool is None:
        gene_pool = ConstSizeHeap(population_size)
    for i in range(epochs):
        bitmasks = [parent[1] for parent in gene_pool.to_list()]
        generation = bitmask_generation(bitmasks, history, bits, population_size, avg_mutations, child_rate)
        scores = pmap(score_func, generation)
        for item in zip(scores, generation):
            gene_pool.push(item)
        if verbose:
            print "Completed epoch: {}\t Best score: {}".format(i, max([x[0] for x in gene_pool.to_list()]))
    return gene_pool


def feature_bitmask(X, bitmask):
    """ keeps only columns of a 2D numpy array corresponding to the input bitmask
    """
    return X[:, np.where(bitmask)[0]]


def bitmask_score_func(clf,
                       X,
                       y,
                       score_func,
                       X_valid=None,
                       y_valid=None,
                       n_iter=3,
                       test_size=0.1,
                       random_state=None):
    """ returns a function that takes in a bitmask and returns it's score
    """
    def wrapped(bitmask):
        new_X = feature_bitmask(X, bitmask)
        if X_valid is not None:
            new_X_valid = feature_bitmask(X_valid, bitmask)
        else:
            new_X_valid = None
        return quick_score(clf=clf,
                           X=X,
                           y=y,
                           score_func=score_func,
                           X_valid=new_X_valid,
                           y_valid=y_valid,
                           n_iter=n_iter,
                           test_size=test_size,
                           random_state=random_state)
    return wrapped

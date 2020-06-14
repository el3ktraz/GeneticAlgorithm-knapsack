import numpy as np
import random

# Hyper-parameters
INVENTORY = (
    ("map", 9, 150), ("compass", 13, 35), ("water", 153, 200), ("sandwich", 50, 160),
    ("glucose", 15, 60), ("tin", 68, 45), ("banana", 27, 60), ("apple", 39, 40),
    ("cheese", 23, 30), ("beer", 52, 10), ("suntan cream", 11, 70), ("camera", 32, 30),
    ("t-shirt", 24, 15), ("trousers", 48, 10), ("umbrella", 73, 40),
    ("waterproof trousers", 42, 70), ("waterproof overclothes", 43, 75),
    ("note-case", 22, 80), ("sunglasses", 7, 20), ("towel", 18, 12),
    ("socks", 4, 50), ("book", 30, 10),
)
BAG_CAPACITY = 400
ELITISM = 4
POPULATION_SIZE = 200
MUTATION = 10
ITERATIONS = 3000


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.value = 0


def InitPopulation(population_size, number_of_objects):
    """
    Init the binary population
    :param population_size: size of population
    :param number_of_objects: number of objects, this will be used to compose the chromosome
    :return: the population with random solutions
    """
    p = 0.5
    population = []
    for idx in range(population_size):
        population.append(Chromosome(np.random.choice(a=[False, True], size=(1, number_of_objects), p=[p, 1 - p])[0]))

    return population


def EvaluateSingleChromosome(genes):
    """
    Computes the value of a chromosome.
    :param genes:
    :return: the value of the chromosome
    """
    global INVENTORY
    global BAG_CAPACITY

    total_weight = 0
    total_value = 0
    for idx, presence in enumerate(genes):
        if presence:
            total_value += INVENTORY[idx][2]
            total_weight += INVENTORY[idx][1]
            if total_weight > BAG_CAPACITY:
                return 0
    return total_value


def EvaluatePopulation(chrm_list):
    """
    Assess the population according to the evaluation function
    :param chrm_list:
    :return:
    """
    for chrm in chrm_list:
        chrm.value = EvaluateSingleChromosome(chrm.genes)


def Select(population):
    """
    Using Roulette Wheel
    :param population: The population from which I might choose an individual
    :return: the index of the selected one
    """
    chosen = random.randrange(sum([chrm.value for chrm in population]) + 1)
    roulette = 0
    for idx, chrm in enumerate(population):
        roulette += chrm.value
        if roulette >= chosen:
            return idx
    return len(population)


def Crossover(parent1, parent2):
    """
    This returns only 1 chromossome resulting from the crossover of parent1 and parent2
    :param parent1: a Chromossome from previous generation
    :param parent2: a Chromossome from previous generation
    :return:
    """
    cutting_point = random.randrange(len(parent1.genes))
    return Chromosome(list(parent1.genes[:cutting_point]) + list(parent2.genes[cutting_point:]))


def Mutate(chromosome):
    """
    Flip the boolean according to some probability
    :param chromosome: the chromosome to be mutated
    :return: nothing
    """
    global MUTATION
    if random.randrange(100) <= MUTATION:
        gene = random.randrange(len(chromosome.genes))
        chromosome.genes[gene] = not chromosome.genes[gene]


def NextGeneration(population):
    """
    Creates the next population based on the current one
    :param population:
    :return:
    """
    global POPULATION_SIZE
    global ELITISM

    new_population = []

    # Performing elitism with an absolute value - ELITISM
    new_population.extend(population[0:ELITISM])
    for count in range(POPULATION_SIZE - len(new_population)):
        # Selection phase
        idx_p1 = Select(population)
        idx_p2 = Select(population)

        # This introduces a little bit of randomness to the algorithm
        if idx_p1 == idx_p2:
            idx_p2 = random.randrange(len(population))

        chrm = Crossover(population[idx_p1], population[idx_p2])
        Mutate(chrm)
        new_population.append(chrm)

    return new_population


def main():
    global ITERATIONS
    global POPULATION_SIZE
    global INVENTORY

    pop = InitPopulation(POPULATION_SIZE, len(INVENTORY))
    i = 0
    while i < ITERATIONS:
        EvaluatePopulation(pop)
        pop.sort(key=lambda x: x.value, reverse=True)
        pop = NextGeneration(pop)
        print("ITERATION[" + str(i) + "] Best result -> " + str(pop[0].value))
        i += 1


if __name__ == "__main__":
    main()

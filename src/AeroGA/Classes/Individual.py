
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.genes)
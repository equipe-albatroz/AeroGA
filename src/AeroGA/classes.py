from loguru import logger

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.genes)

class ErrorType:
    type: str
    message: str

    def __init__(self, type, message):
        self.type = type
        self.message = message
    
    def __str__(self) -> str:
        return f'Message: {self.message}\nType: {self.type}'

class Log(object):
    def __init__(self, log_file):
        self.log_file = log_file
        logger.add(log_file)
        
    def info(self, message):
        logger.info(message)

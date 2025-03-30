from .. import settings


def check_bounds(max_values: list = None, min_values: list = None):
    """
    Checking if num_varialbes matches the lb and ub sizes
    """

    if len(max_values) != len(min_values):
        settings.log.critical("There is an inconsistency in the bounds: Between the size of lower and upper bounds")
        return dict(error = 1)
    
    for i in range(len(max_values)):
        if max_values[i] < min_values[i]:
            settings.log.critical("There is an inconsistency in the bounds: Upper bound - {max_values[i]} is lower then the lower bounds - {min_values[i]}")
            return dict(error = 1)
        
    pass
    
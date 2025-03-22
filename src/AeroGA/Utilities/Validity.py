from .. import settings


def check_bounds(max_values: list = None, min_values: list = None):
    """
    Checking if num_varialbes matches the lb and ub sizes
    """

    if len(max_values) != len(min_values):
        settings.log.critical("There is an inconsistency between the size of lower and upper bounds")
        return dict(error = 1)
    
    pass
    
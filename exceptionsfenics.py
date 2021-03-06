# Custom exceptions for Fenics:

# Exceptions for plotfenics
class NoOutdirError(Exception):
    """Forgot to define an output directory for plots"""
    pass

class NoVarnameError(Exception):
    """Forgot to define varname for plot"""
    pass

class NoIndicesError(Exception):
    """No index list to gather plots"""
    pass

# Exceptions for datafenics/datamisfit
class WrongInstanceError(Exception):
    """Input is from the wrong instance"""
    pass

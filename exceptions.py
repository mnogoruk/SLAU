class SLAUException(Exception):
    INDEFINITE = "System is indefinite."
    INCONSISTENT = 'System is inconsistent'
    INCORRECT = 'System is incorrect'
    PREMATURELY_BACKWARD = "Can not backward before forward."

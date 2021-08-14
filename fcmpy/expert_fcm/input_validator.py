# Taken and modified from Simon Pirschel at
# https://aboutsimon.com/blog/2018/04/04/Python3-Type-Checking-And-Data-Validation-With-Type-Hints.html

from functools import wraps
from typing import get_type_hints, get_args
from inspect import getfullargspec


def validate_input(obj, **kwargs):
    """
    Check whether the passed arguments are of the same type
    as the pre-specified typehints.    
    """
    hints = get_type_hints(obj)

    # iterate all type hints
    for attr_name, attr_type in hints.items():
        if attr_name == 'return':
            continue
        if attr_name not in kwargs.keys(): # for default arguments (self)
            continue
        else:
            try:
                if not isinstance(kwargs[attr_name], attr_type):
                    raise TypeError(
                        'Argument %r is not of type %s' % (attr_name, attr_type)
                    )
            except: # for Union cases.
                if not isinstance(kwargs[attr_name], get_args(attr_type)):
                    raise TypeError(
                        'Argument %r is not of type %s' % (attr_name, attr_type)
                    )
                
def type_check(decorator):
    """
    Decorator for validating the typehints.
    """
    @wraps(decorator)
    def wrapped_decorator(*args, **kwargs):
        # translate *args into **kwargs
        func_args = getfullargspec(decorator)[0]
        kwargs.update(dict(zip(func_args, args)))

        validate_input(decorator, **kwargs)
        return decorator(**kwargs)

    return wrapped_decorator
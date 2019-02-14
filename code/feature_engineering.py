import numpy as np


def feature_engineer(x):
    x = handle_name(x)
    return x


def handle_name(x):
    """
    :param x:
    :return: X with added two features and removed Name feature:
    1. name_type: categorical
    values:
        0: no name 'nan'
        1: multiple names
        2: not a real name
        3: real name
    2. name length: scalar.
    """
    names = x.Name
    name_type = list(map(lambda x: 'NO_NAME' if type(x) == float else x, names))

    def cond(a):
        if a == 'NO_NAME':  # no name (nan)
            return 0
        elif '&' in a or 'and' in a.lower() or ',' in a:  # multiple names
            return 1
        elif ' ' in a:  # not actual name
            return 2
        else:  # real name
            return 3
    name_type = list(map(cond, name_type))
    name_length = list(map(lambda x: 0 if type(x) == float else len(x), names))
    x = x.drop('Name', axis=1)
    x['name_type'] = name_type
    x['name_length'] = name_length
    return x












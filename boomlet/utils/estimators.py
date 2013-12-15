import math


def flexible_int(size, in_val=None):
    """ allows for flexible input as a size
    """
    if in_val is None:
        return size
    elif isinstance(in_val, (float, int)):
        if isinstance(in_val, float):
            assert abs(in_val) <= 1.0, in_val
            in_val = int(round(in_val * size))
        if in_val < 0:
            in_val += size  # setting negative values as amount not taken
        return max(0, min(size, in_val))
    elif isinstance(in_val, str):
        if in_val == "sqrt":
            return int(round(math.sqrt(size)))
        elif in_val == "log2":
            return int(round(math.log(size) / math.log(2)))
        elif in_val == "auto":
            return size
    raise Exception("Improper flexible_int input: {}".format(in_val))

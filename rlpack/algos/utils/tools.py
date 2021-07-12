import torch
def get_aggregate(aggregate_way):
    def get_min(a, b):
        return torch.min(a, b)
    def get_avg(a, b):
        return a+b/2
    
    if aggregate_way == "min":
        return get_min
    elif aggregate_way == "avg":
        return get_avg
    else:
        raise NotImplementedError
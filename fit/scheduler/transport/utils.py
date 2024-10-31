import torch as th

class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []
    
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    
    return '\n'.join(result)






def get_flexible_mask_and_ratio(model_kwargs: dict, x: th.Tensor):
    '''
    sequential case (fit): 
        x: (B, N, C)
        model_kwargs: {y: (B,), mask: (B, N), grid: (B, 2, N)}
        mask: (B, N) -> (B, 1, N)
    spatial case (dit):
        x: (B, C, H, W)
        model_kwargs: {y: (B,)}
        mask: (B, C) -> (B, C, 1, 1)
    '''
    mask = model_kwargs.get('mask', th.ones(x.shape[:2]))    # (B, N) or (B, C)
    ratio = float(mask.shape[-1]) / th.count_nonzero(mask, dim=-1)  # (B,)
    if len(x.shape) == 3:               # sequential x: (B, N, C)
        mask = mask[..., None]         # (B, N) -> (B, N, 1)
    elif len(x.shape) == 4:             # spatial x: (B, C, H, W)
        mask = mask[..., None, None]    # (B, C) -> (B, C, 1, 1)
    else:
        raise NotImplementedError
    return mask.to(x), ratio.to(x)
    
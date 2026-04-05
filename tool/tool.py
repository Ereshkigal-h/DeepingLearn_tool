import numpy as np
import torch
import random
import os
def np_to_tensor(arrays,device='gpu',accuracy=torch.float32):
    tensor=torch.from_numpy(arrays).type(accuracy)
    return tensor.cuda() if device=='gpu' else tensor
def set_seed(seed):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
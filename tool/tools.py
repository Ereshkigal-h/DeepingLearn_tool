import numpy as np
import torch
import random
import os
def np_to_tensor(arrays,device='gpu',accuracy=torch.float32):
    tensor=torch.from_numpy(arrays).type(accuracy)
    return tensor.cuda() if device=='gpu' else tensor
def set_seed(seed):
        #基础环境随机性
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        #PyTorch 全局与 CPU 随机性
        torch.manual_seed(seed)

        #PyTorch GPU 随机性
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #torch.use_deterministic_algorithms(True)
        print(f"全局随机种子已固定为: {seed}")
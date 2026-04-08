import numpy as np
import torch
import random
import os
def np_to_tensor(arrays,device='gpu',accuracy=torch.float32):
    tensor=torch.from_numpy(arrays).type(accuracy)
    return tensor.cuda() if device=='gpu' else tensor
def set_seed(seed):
        #基础环境随机性
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止 Python 的 Hash 随机化
        random.seed(seed)  # 固定 Python 内置 random 模块
        np.random.seed(seed)  # 固定 Numpy 的随机数生成器

        #PyTorch 全局与 CPU 随机性
        torch.manual_seed(seed)  # 固定 PyTorch CPU 端的种子

        #PyTorch GPU 随机性
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 固定当前 GPU 的种子
            torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 的种子 (多卡训练必备)

        #底层 cuDNN 卷积算法的确定性
        torch.backends.cudnn.deterministic = True  # 强制使用确定性算法
        torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动寻找最快算法的机制

        #强制 PyTorch 报错如果使用了不可复现的算法
        torch.use_deterministic_algorithms(True)
        print(f"全局随机种子已固定为: {seed}")
import torch
from torch.utils.data import Dataset
from tool.check_data import load_data_txt
import ast

def load_cornell_dialogue(data_path, label_path):
    line_list = load_data_txt(data_path, sep=" +++$+++ ", encoding="iso-8859-1")
    label_list = load_data_txt(label_path, sep=" +++$+++ ", encoding="iso-8859-1")
    id_dict = dict(zip(line_list.iloc[:, 0], line_list.iloc[:, 4]))

    label_list['col_3'] = label_list['col_3'].apply(ast.literal_eval)
    qa_data = []

    for sequence in label_list["col_3"]:
        for i in range(len(sequence) - 1):
            q_id = sequence[i].strip()
            a_id = sequence[i + 1].strip()

            q_text = str(id_dict.get(q_id, "")).strip()
            a_text = str(id_dict.get(a_id, "")).strip()

            if q_text and a_text:
                qa_data.append((q_text, a_text))

    return qa_data


class NLPDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 仅仅返回字符串，不做任何转换
        # 数据会在 DataLoader 的 collate_fn 阶段交由 Tokenizer 处理
        item = self.data_list[index]
        return item
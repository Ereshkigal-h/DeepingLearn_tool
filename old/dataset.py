import ast
import os
from collections import Counter
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tool.check_data import load_data_txt
import tool
def read_vocab(text_list,lower=False,min_freq=2):
    counter = Counter()
    for text in text_list:
        if lower:
            text = str(text).lower()
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)

        tokens = text.strip().split()
        counter.update(tokens)
    vocab = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3
    }
    current_id = 4
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = current_id
            current_id += 1

    inv_vocab = {v: k for k, v in vocab.items()}

    print(f"词汇表构建完成！有效单词数量: {len(vocab)}")
    return vocab, inv_vocab
def load_cornell_dialogue(data_path,label_path):
    line_list = load_data_txt(data_path,sep=" +++$+++ ",encoding = "iso-8859-1")
    label_list= load_data_txt(label_path,sep=" +++$+++ ",encoding = "iso-8859-1")
    id_dict=dict(zip(line_list.iloc[:,0],line_list.iloc[:,4]))
    vocab,in_vocab = read_vocab(line_list["col_4"].tolist())
    label_list['col_3'] = label_list['col_3'].apply(ast.literal_eval)
    qa_data = []
    for sequence in label_list["col_3"]:
        for i in range(len(sequence)-1):
            q_id = sequence[i].strip()
            a_id = sequence[i + 1].strip()

            q_val = id_dict.get(q_id, "")
            a_val = id_dict.get(a_id, "")

            q_text = q_val.strip() if isinstance(q_val, str) else ""
            a_text = a_val.strip() if isinstance(a_val, str) else ""
            if q_text and a_text:
                qa_data.append([q_text,a_text])
    return qa_data,vocab,in_vocab

class General_Dataset(Dataset):
    def __init__(self, train_path,label_path, image_size=112, accuracy="float32"):
        self.train_path,self.label_path = train_path,label_path
        self.images_size = image_size

        self.vocab,self.in_vocab,self.samples,self.mask_samples=self.read_path(self.train_path,self.label_path)
        dtype_mapping = {
            "float16": (np.float16, torch.float16),
            "float32": (np.float32, torch.float32),
            "float64": (np.float64, torch.float64)
        }
        self.numpy_type,self.torch_type = dtype_mapping[accuracy]
    def read_path(self,train_path,label_path,max_length=20):
        samples,vocab,in_vocab = load_cornell_dialogue(train_path,label_path)
        result=[]
        mask_result=[]
        for sample1,sample2 in samples:
            sample1 = sample1.split(" ")
            sample2 = sample2.split(" ")
            pad_id,unk_id=vocab.get('<PAD>',0),vocab.get('<UNK>',3)
            sample1 = sample1[:max_length]
            sample2 = sample2[:max_length]
            valid_len1 = len(sample1)
            valid_len2 = len(sample2)
            sample1 = [vocab.get(sample,unk_id) for sample in sample1]
            sample2 = [vocab.get(sample,unk_id) for sample in sample2]
            padding1=max_length-valid_len1
            padding2=max_length-valid_len2
            sample1 = sample1+padding1*[pad_id]
            sample2 = sample2+padding2*[pad_id]
            mask_sample1=[1]*valid_len1+[0]*(max_length-valid_len1)
            mask_sample2=[1]*valid_len2+[0]*(max_length-valid_len2)
            result.append((sample1, sample2))
            mask_result.append((mask_sample1, mask_sample2))
        return vocab,in_vocab,result,mask_result
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        single_sample=self.samples[index]
        data,label=single_sample
        return torch.tensor(data,dtype=self.torch_type),torch.tensor(label,dtype=self.torch_type)
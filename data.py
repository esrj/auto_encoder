from mapping import  mapping_height,padding_height
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

# 將 100000 筆 input data 合併
# input_num = 100000
# dataset = []
#
# for i in range(input_num):
#     with open(f'Data/input{i}.txt','r',encoding='utf-8') as f:
#         line = f.readlines()
#         dataset.append(line)
#
#
# with open('Data/dataset.txt','w',encoding='utf-8') as f:
#     for line in dataset:
#         merged = " ".join(line).strip()
#         f.write(merged + "\n")


# 輸出成 2D tensor

# with open('data/dataset.txt', 'r', encoding='utf-8') as f:
#     lines = f.read().splitlines()

# for i,line in enumerate(lines):
#    params_data = [float(data) for data in line.split()]
#    params = params_data[:10]
#
#    H = mapping_height(params, resolution_x_mm=1.0, resolution_y_mm=1.0)
#    padding_H = padding_height(H)  # shape: (401, 401)
#
#    tensor_H = torch.tensor(padding_H, dtype=torch.float32)
#
#    torch.save(tensor_H, f'mapping_data/2D_{i}.pt')




class MappingPTDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(f"{folder}/*.pt"))  #只存檔案路徑

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])  # [400, 400]
        data = data.float() / 70.0          # 正規化到 [0, 1]
        data = data.unsqueeze(0)  # [1, 400, 400] (加 channel 維度)
        return data








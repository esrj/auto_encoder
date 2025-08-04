from data import MappingPTDataset
from torch.utils.data import Dataset, DataLoader
import torch
from model import Autoencoder
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')




dataset = MappingPTDataset('mapping_data')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

batch = next(iter(test_dataloader))

model = Autoencoder()
state_dict = torch.load("auto_encoder.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    out,_ = model(batch)

for i in range(100):
    ground_true = (batch[i][0])
    pred = (out[i][0])

    fig, ax = plt.subplots()
    ax.plot(range(400), pred[100])
    ax.grid(True)
    fig.savefig(f"img/pred_{i}.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(400), ground_true[100])
    ax.grid(True)
    fig.savefig(f"img/true_{i}.png")
    plt.close(fig)

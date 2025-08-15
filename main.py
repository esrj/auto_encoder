from data import MappingPTDataset
from torch.utils.data import Dataset, DataLoader
import torch
from model import Autoencoder
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = MappingPTDataset('mapping_data')

dataloader = DataLoader(dataset, batch_size=100, shuffle=True,num_workers=4)
# dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

model = Autoencoder().to(device)
myLoss = nn.MSELoss()
myOptim = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    myOptim,
    max_lr=1e-3,
    total_steps=50 * len(dataloader), # 總次數
    pct_start=0.1,   # 前面 0.1 提升 lr
    anneal_strategy='cos',
    div_factor=10,  # 一開始加 10 倍，從 0.0001 到 0.001
    final_div_factor=100  # 決定週期結束時的最小 learning rate 0.0001 * 0.01
)

for epoch in range(50):
    model.train()
    running_loss = 0
    for i,batch in enumerate(dataloader):

        batch = batch.to(device)
        myOptim.zero_grad()
        out, _ = model(batch)
        loss = myLoss(out, batch)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)

        myOptim.step()
        scheduler.step()

        myOptim.step()
        running_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch [{epoch}:{(i/3000)*100}%] , Loss:{loss.item()}")
    if epoch % 10 == 0 and epoch >= 30 :
        torch.save(model.state_dict(),f'{epoch}_auto_encoder.pt')
        print(f'{epoch}_auto_encoder.pt is create')
    print("============================================================")
    print(f"Epoch [{epoch}], Loss: {running_loss / len(dataloader):.4f}")
    print("============================================================")

torch.save(model.state_dict(), 'auto_encoder.pt')

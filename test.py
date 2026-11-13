import torch
from torch.utils.data import Dataset,DataLoader
torch.manual_seed(42)
t_x=torch.randn(4,3,dtype=torch.float32)
t_y=torch.arange(4)
class JointData(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
join_dataset=JointData(t_x,t_y)
loader=DataLoader(join_dataset,batch_size=2,shuffle=True)
for epoch in range(3):
    print(f"epoch: {epoch+1}\n")
    for i,batch in enumerate(loader,1):
        print(f"batch: {i} x: {batch[0]}, y: {batch[1]}\n")
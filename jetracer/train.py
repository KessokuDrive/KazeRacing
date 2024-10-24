import torch
import torchvision

from torch.utils.data import DataLoader
from DataLoader import XYDataset
import torchvision.transforms as transforms
device = torch.device('cuda')

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_eval(dataloader,model,is_training):
    try:
        if is_training:
            model = model.train()
        else:
            model = model.eval()
        for batch_idx, (images, xy) in enumerate(dataloader):
            print(f"Batch {batch_idx}")
            # send data to device
            images = images.to(device)
            xy = xy.to(device)

            if is_training:
                # zero gradients of parameters
                optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)

            # compute MSE loss over x, y coordinates for associated categories
            loss = torch.mean((outputs[batch_idx] - xy[batch_idx])**2)
            if batch_idx % 100 == 0:
                current = batch_idx * batch_size + len(images)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
            if is_training:
                # run backpropogation to accumulate gradients
                loss.backward()

                # step optimizer to adjust parameters
                optimizer.step()
                        # increment progress
    except Exception as e:
        print("Error" + str(e))
    model = model.eval()
    torch.save(model.state_dict(), f"model_{epoch}.pth")

#Set batch size, larger batch sizes will be train faster and stabilize learning
batch_size = 64

#Load datasets and create dataloaders
train_datasets = XYDataset("datasets/train.txt", TRANSFORMS, random_hflip=True)
valid_datasets = XYDataset("datasets/valid.txt", TRANSFORMS, random_hflip=True)
train_dataloader = DataLoader(train_datasets, batch_size, shuffle=True)
test_dataloader = DataLoader(valid_datasets, batch_size, shuffle=True)
#Load model and optimizer
model = torchvision.models.wide_resnet101_2(pretrained=True).to(device)
model.fc = torch.nn.Linear(2048, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#set number of epochs
epoch = 200

for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_eval(train_dataloader, model, True)
    train_eval(test_dataloader, model, False)
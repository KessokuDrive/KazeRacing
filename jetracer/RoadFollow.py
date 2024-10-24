
import torch
import torchvision
device = torch.device('cuda')

model = torchvision.models.wide_resnet101_2(pretrained=True).to(device)
model.fc = torch.nn.Linear(2048, 2).to(device)
model = model.to(device)
model.load_state_dict(torch.load("model_10.pth"))

#TODO: Put our follow code here, or just run the notebook code.
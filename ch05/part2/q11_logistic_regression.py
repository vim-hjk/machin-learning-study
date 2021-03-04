import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import argparse

parser = argparse.ArgumentParser(description='loss_type')
parser.add_argument('--loss_type', type=str, help='loss function type', default='log')
args = parser.parse_args()

data = load_breast_cancer()
x = data['data']
y = data['target']
print("shape of x: {}\n shape of y: {}".format(x.shape, y.shape))

sc = StandardScaler()
x = sc.fit_transform(x)

class dataset(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
  def __len__(self):
    return self.length
trainset = dataset(x, y)

trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

class LogisticRegression(nn.Module):
  def __init__(self, input_shape):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_shape, 1)
  def forward(self, x):
    output = torch.sigmoid(self.linear(x))
    return output, self.linear(x).mean()

learning_rate = 0.001
epochs = 700

model = LogisticRegression(input_shape=x.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
if args.loss_type == 'mse':
  loss_fn = nn.MSELoss()
if args.loss_type == 'log':
  loss_fn = nn.BCELoss()


ws = []
losses = []
accur = []
for i in range(epochs):
  for j, (x_train, y_train) in enumerate(trainloader):
    
    #calculate output
    output, w = model(x_train)
 
    #calculate loss
    loss = loss_fn(output, y_train.reshape(-1, 1))
 
    #accuracy
    predicted, _ = model(torch.tensor(x, dtype=torch.float32))
    acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i % 20 == 0:
    ws.append(w)
    losses.append(loss)
    accur.append(acc)
    print("epoch {}\t loss : {}\t accuracy : {}".format(i, loss, acc))

plt.plot(losses)
plt.title('Epoch-Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()

plt.plot(accur)
plt.title('Epoch-Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(ws, losses)
plt.title('Weight-Loss')
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.show()
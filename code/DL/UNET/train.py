#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--samples', help='npz archive with the inputs and desired masks', type=str, default='caras.npz')
parser.add_argument('--model', help="name of model to optimize", type=str, default=None)
parser.add_argument('--epochs', help="number of epochs to run", type=int, default=20)
parser.add_argument('--loss', help="target loss to stop", type=float, default=0)
args = parser.parse_args()

samples = np.load(args.samples)
Xr,Y = (samples[s] for s in ['x','y'])

sdev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(sdev)
device = torch.device(sdev)

from myUNET import *

if args.model is None:
    model = UNet(in_channels=1, out_channels=1)
    model.to(device)
    modelname = 'model.torch'
else:
    modelname = args.model
    model = torch.load(modelname, map_location=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(
    torch.from_numpy(Xr.astype(np.float32)[:,:,:,[1]].transpose(0,3,1,2)).to(device),
    torch.from_numpy(Y.astype(np.float32).reshape(-1,128,128,1).transpose(0,3,1,2)).to(device)
)

dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

target_loss = args.loss
num_epochs = args.epochs
history = []

for epoch in range(num_epochs):
    model.train()  # Poner el modelo en modo entrenamiento
    running_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Limpiar los gradientes
        outputs = model(inputs)  # Obtener las predicciones del modelo
        loss = criterion(outputs, labels)  # Calcular la pérdida
        #print(loss)
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizar los pesos

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    history.append(epoch_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    if epoch_loss < target_loss:
        break

plt.plot(history)
plt.xlabel('epoch'); plt.ylabel('loss')
plt.show()

torch.save(model, modelname)


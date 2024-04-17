#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--samples', help='npz archive with the inputs and desired masks', type=str, default='caras.npz')
parser.add_argument('--model', help="name of model to check", type=str, default='caras.torch')
args = parser.parse_args()

samples = np.load(args.samples)
Xr,Y = (samples[s] for s in ['x','y'])
samples = np.random.choice(len(Xr), 13, replace=False)

sdev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(sdev)
device = torch.device(sdev)

from myUNET import *
model = torch.load(args.model, map_location=device)

xs = torch.from_numpy(Xr[samples].astype(np.float32)[:,:,:,[1]].transpose(0,3,1,2)).to(device)
ys = model(xs).detach().cpu().numpy().reshape(-1,128,128)

fig, axs = plt.subplots(3,1, figsize=(8,3))
axs[0].imshow(np.hstack(Xr[samples]))
axs[0].set_axis_off()
axs[1].imshow(np.hstack(Y[samples]))
axs[1].set_axis_off()
axs[2].imshow(np.hstack(np.clip(ys,0,255)))
axs[2].set_axis_off()
plt.show()

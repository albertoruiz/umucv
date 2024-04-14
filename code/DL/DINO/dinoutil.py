import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

# Load the model from the file...
dinov2_vitb14_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# ... and move to GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_vitb14_model.to(device)

# Define the transforms to be applied to the input image:
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
transform_fun = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN,std=STD)
])

def DINOv2_from_image(image):
    # Apply the transforms to the input image
    input_tensor = transform_fun(Image.fromarray(image)).unsqueeze(0).to(device)
    # Use the model to perform inference on the input image
    output = dinov2_vitb14_model.forward_features(input_tensor)

    # Convert the output tensor to a numpy array
    output_cls = output['x_norm_clstoken'].detach().cpu().numpy()
    output_patches = output['x_norm_patchtokens'].detach().cpu().numpy()
    h, w, _ = image.shape
    pw = w//14
    ph = h//14
    return output_cls[0,:], output_patches[0,:].reshape(ph,pw,-1)


def resize_for_dino(rgb):
    h,w,_ = rgb.shape
    pw = w//14
    ph = h//14
    return rgb[:ph*14,:pw*14]



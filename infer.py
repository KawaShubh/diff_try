import torch
import yaml
from PIL import Image
import os
import numpy as np
from models.vae import VAE
from models.unet import WCESuperResUNet

# Load config
with open("configs/inference.yaml") as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device(config["device"])
os.makedirs("outputs", exist_ok=True)

# Load models
vae = VAE().to(device)
unet = WCESuperResUNet().to(device)
unet.load_state_dict(torch.load(config["model"]["checkpoint"]))
unet.eval()

# Process images
for filename in os.listdir(config["data"]["lr_dir"]):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Load image
    lr_path = os.path.join(config["data"]["lr_dir"], filename)
    lr_img = Image.open(lr_path).convert("RGB").resize(
        (config["data"]["lr_size"], config["data"]["lr_size"])
    )
    
    # Convert to tensor [-1, 1] range
    lr_tensor = (torch.tensor(np.array(lr_img)).float() / 127.5 - 1
                ).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run through model
    with torch.no_grad():
        # Encode to latent space
        lr_latent = vae.encode(lr_tensor)
        
        # Super-resolve in latent space
        hr_latent = unet(lr_latent)
        
        # Decode to image space
        hr_tensor = vae.decode(hr_latent)
    
    # Convert and save
    hr_img = (hr_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5
    hr_img = Image.fromarray(hr_img.astype(np.uint8)).resize(
        (config["model"]["output_size"], config["model"]["output_size"])
    )
    hr_img.save(f"outputs/sr_{filename}")
    print(f"Processed: {filename} → outputs/sr_{filename}")
from diffusers import AutoencoderKL

class VAE:
    def __init__(self, pretrained=True, device=None):
        # Store the device
        self.device = device
        
        # Load the pretrained model and move it to the specified device
        self.model = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(self.device) if pretrained else None
    
    def encode(self, x):
        return self.model.encode(x).latent_dist.sample() * 0.18215
    
    def decode(self, x):
        return self.model.decode(x / 0.18215).sample

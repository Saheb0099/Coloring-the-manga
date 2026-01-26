import torch
from torchvision.transforms import ToTensor
import numpy as np

from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad

class MangaColorizator:
    def __init__(self, device, generator_path = 'networks/generator.zip', extractor_path = 'networks/extractor.pth'):
        self.colorizer = Colorizer().to(device)
        self.colorizer.generator.load_state_dict(torch.load(generator_path, map_location = device))
        self.colorizer = self.colorizer.eval()
        
        self.denoiser = FFDNetDenoiser(device)
        
        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        
        self.device = device
        
    def set_image(self, image, size = 576, apply_denoise = True, denoise_sigma = 25, transform = ToTensor()):
        # --- MEMORY HYGIENE ---
        # Explicitly clear old tensors before starting a new page.
        # This keeps the 16GB RAM pool open for the next image.
        self.current_image = None
        self.current_hint = None
        if self.device == 'mps':
            torch.mps.empty_cache()

        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        # Denoiser stays in float32 to prevent "Input type and bias type" crashes
        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        
        image, self.current_pad = resize_pad(image, size)
        
        # Using standard .float() (32-bit) for 100% stability on M4
        self.current_image = transform(image).unsqueeze(0).to(self.device).float()
        
        self.current_hint = torch.zeros(
            1, 4, self.current_image.shape[2], self.current_image.shape[3]
        ).float().to(self.device)
    
    def update_hint(self, hint, mask):
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255
            
        hint = (hint - 0.5) / 0.5
        
        # Standard 32-bit tensors
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        # Using torch.no_grad() is essential for memory during 200+ page batches
        with torch.no_grad():
            # Standard 32-bit concatenation
            input_tensor = torch.cat([self.current_image, self.current_hint], 1)
            
            fake_color, _ = self.colorizer(input_tensor)
            fake_color = fake_color.detach()

        # Move to CPU immediately to free up GPU/MPS space in the 16GB pool
        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if self.current_pad[0] != 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] != 0:
            result = result[:, :-self.current_pad[1]]
            
        # --- FINAL CLEANUP ---
        # Shred the references so Python clears the RAM immediately
        self.current_image = None
        self.current_hint = None
            
        return result.numpy()
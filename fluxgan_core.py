# fluxgan_core.py

import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_dim = 10

class Generator(nn.Module):
    def __init__(self, noise_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 3, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, z, conditions):
        x = torch.cat([z, conditions], dim=1)
        return self.net(x)

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator = Generator(noise_dim=noise_dim).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    return generator, {
        'enrichment_center': checkpoint['enrichment_scaler']['center'],
        'enrichment_scale': checkpoint['enrichment_scaler']['scale'],
        'flux_burnup_min': checkpoint['flux_burnup_min'],
        'flux_burnup_max': checkpoint['flux_burnup_max'],
        'scaler': {'feature_range': (0, 1)}
    }

def predict_flux_burnup(generator, checkpoint_info, enrichment_value):
    # Normalize input
    enrich_center = checkpoint_info['enrichment_center'][0]
    enrich_scale = checkpoint_info['enrichment_scale'][0]
    norm_enrich = (enrichment_value - enrich_center) / enrich_scale

    # Dummy conditions (or improve later)
    flux_cond = 0.0
    burnup_cond = 0.0
    input_conditions = np.array([norm_enrich, flux_cond, burnup_cond])
    z = torch.randn(1, noise_dim, device=device)
    cond_tensor = torch.tensor(input_conditions, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        output = generator(z, cond_tensor).cpu().numpy()[0]

    # De-normalize Flux and Burnup
    flux_min, burnup_min = checkpoint_info['flux_burnup_min']
    flux_max, burnup_max = checkpoint_info['flux_burnup_max']
    feature_range = (0, 1)

    def denormalize(val, idx):
        min_val = [flux_min, burnup_min][idx]
        max_val = [flux_max, burnup_max][idx]
        return np.expm1(((val - feature_range[0]) / (feature_range[1] - feature_range[0])) * (max_val - min_val) + min_val)

    flux = denormalize(output[1], 0)
    burnup = denormalize(output[2], 1)

    return flux, burnup

def predict_flux_burnup(generator, checkpoint_info, enrichment_value):
    # Extract normalization values
    enrich_center = checkpoint_info['enrichment_center'][0]
    enrich_scale = checkpoint_info['enrichment_scale'][0]
    flux_min, burnup_min = checkpoint_info['flux_burnup_min']
    flux_max, burnup_max = checkpoint_info['flux_burnup_max']
    feature_range = (0, 1)
    noise_dim = 10

    # Normalize enrichment input
    norm_enrich = (enrichment_value - enrich_center) / enrich_scale

    # Dummy conditional values (0) for flux and burnup input
    flux_cond = 0.0
    burnup_cond = 0.0
    condition_input = np.array([norm_enrich, flux_cond, burnup_cond])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.randn(1, noise_dim, device=device)
    cond_tensor = torch.tensor(condition_input, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        output = generator(z, cond_tensor).cpu().numpy()[0]

    # Denormalize Flux and Burnup
    def denormalize(val, idx):
        min_val = [flux_min, burnup_min][idx]
        max_val = [flux_max, burnup_max][idx]
        return np.expm1(((val - feature_range[0]) / (feature_range[1] - feature_range[0])) * (max_val - min_val) + min_val)

    flux = denormalize(output[1], 0)
    burnup = denormalize(output[2], 1)

    return flux, burnup

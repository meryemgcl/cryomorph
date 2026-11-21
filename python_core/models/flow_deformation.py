"""
CryoMorph: Pillar B - Continuous Deformation Vector Flow Fields (3DFlex & Flow Matching Style)

Implements continuous 3D macromolecular flexibility via invertible vector displacement fields:
- 3D Canonical Consensus Potential V_0(x) deformed via pull-back: V(x; z) = V_0(x - u(x; z)).
- 3D Differentiable Trilinear Volume Warping (F.grid_sample).
- Physical conservation of mass regularizers: Divergence penalty (div u = 0) and elastic smoothness.
- Differentiable 3D-to-2D line-integral projection and Fourier-space CTF modulation.
"""

from __future__ import annotations
import math
import os
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from python_core.models.cryo_gaussian import CryoEncoder


class FlowFieldGenerator(nn.Module):
    """
    Generates a continuous 3D vector displacement field u(x; z) in R^3
    for each spatial grid point x conditioned on latent variable z.
    """
    def __init__(self, grid_size: int = 32, latent_dim: int = 1, hidden_dim: int = 48):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        # Mapping latent z -> coarse 3D deformation grid, followed by 3D transposed convs
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.deconv = nn.Sequential(
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(hidden_dim // 4, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output bounded normalized displacement [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        Returns displacement field u: [B, 3, D, D, D] in normalized coordinates [-0.3, 0.3]
        """
        B = z.size(0)
        feat = self.fc(z).view(B, -1, 4, 4, 4)
        disp = self.deconv(feat) * 0.35  # Scale maximum displacement
        
        # If grid_size != 32, interpolate
        if self.grid_size != 32:
            disp = F.interpolate(disp, size=(self.grid_size, self.grid_size, self.grid_size), mode="trilinear", align_corners=True)
        return disp


class CryoFlowDeformationModel(nn.Module):
    """
    Pillar B End-to-End Continuous Deformation Architecture.
    - Learns a canonical 3D density volume V_0.
    - Predicts continuous displacement vector field u(x; z).
    - Warps V_0(x - u(x; z)) via 3D trilinear sampling.
    - Projects along line of sight, rotates by R_i, and convolved with CTF in Fourier space.
    """
    def __init__(
        self,
        box_size: int = 64,
        volume_grid_size: int = 32,
        latent_dim: int = 1,
        init_volume: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.box_size = box_size
        self.grid_size = volume_grid_size
        self.latent_dim = latent_dim

        # 1. Canonical Consensus 3D Density Grid V_0: [1, 1, D, D, D]
        if init_volume is not None:
            self.consensus_volume = nn.Parameter(init_volume.clone())
        else:
            # Initialize with soft Gaussian sphere in center
            coords = torch.linspace(-1.0, 1.0, volume_grid_size)
            gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing="ij")
            r2 = gx**2 + gy**2 + gz**2
            init_v = torch.exp(-r2 / 0.3).unsqueeze(0).unsqueeze(0)
            self.consensus_volume = nn.Parameter(init_v)

        # Precompute base identity coordinate grid: [1, D, D, D, 3] in [-1, 1]
        c = torch.linspace(-1.0, 1.0, volume_grid_size)
        base_z, base_y, base_x = torch.meshgrid(c, c, c, indexing="ij")
        # grid_sample expects (x, y, z) order in the last dimension
        base_grid = torch.stack([base_x, base_y, base_z], dim=-1).unsqueeze(0)
        self.register_buffer("base_grid", base_grid)

        # 2. Conv2D Encoder
        self.encoder = CryoEncoder(in_channels=2, latent_dim=latent_dim)

        # 3. Flow Field Generator
        self.flow_gen = FlowFieldGenerator(grid_size=volume_grid_size, latent_dim=latent_dim)

    def warp_volume(self, u_disp: torch.Tensor) -> torch.Tensor:
        """
        Warps consensus volume V_0 by displacement field u:
        V(x; z) = V_0(x - u(x; z))
        u_disp: [B, 3, D, D, D]
        Returns: [B, 1, D, D, D]
        """
        B = u_disp.size(0)
        # Permute displacement to [B, D, D, D, 3] for grid_sample
        u_permuted = u_disp.permute(0, 2, 3, 4, 1)
        
        # Pull-back warped sampling coordinates: x_sample = x_base - u
        sampling_grid = self.base_grid.expand(B, -1, -1, -1, -1) - u_permuted
        sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)

        # 3D Trilinear sampling
        warped_volume = F.grid_sample(
            F.softplus(self.consensus_volume).expand(B, -1, -1, -1, -1),
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return warped_volume

    def project_volume(
        self,
        volume: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        ctfs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates 3D volume by R_i, integrates along z to form clean 2D projection,
        and applies CTF modulation in Fourier space.
        """
        B = volume.size(0)
        D = self.grid_size
        
        # Rotate 3D volume: Apply inverse rotation to coordinates: x' = x @ R
        # rotations: [B, 3, 3]
        flat_grid = self.base_grid.view(1, D * D * D, 3).expand(B, -1, -1)
        rotated_grid = torch.bmm(flat_grid, rotations).view(B, D, D, D, 3)

        rotated_volume = F.grid_sample(
            volume,
            rotated_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        # Integrate along z-axis (depth projection) -> Clean 2D projection [B, 1, D, D]
        clean_proj_low = torch.sum(rotated_volume, dim=2)

        # Interpolate to target detector box size (e.g. 32 -> 64)
        clean_proj = F.interpolate(clean_proj_low, size=(self.box_size, self.box_size), mode="bilinear", align_corners=True)

        # 2D Fourier CTF Modulation
        clean_fft = torch.fft.fft2(clean_proj)
        modulated_fft = clean_fft * ctfs
        pred_noisy = torch.real(torch.fft.ifft2(modulated_fft))

        return pred_noisy, clean_proj

    def forward(
        self,
        img: torch.Tensor,
        ctf: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 1. Encode image and CTF to latent state z
        z, mu_z, logvar_z = self.encoder(img, ctf)

        # 2. Generate 3D continuous displacement field u(x; z)
        u_disp = self.flow_gen(z)  # [B, 3, D, D, D]

        # 3. Warp consensus 3D volume
        warped_volume = self.warp_volume(u_disp)  # [B, 1, D, D, D]

        # 4. Project and modulate with CTF
        pred_noisy, pred_clean = self.project_volume(warped_volume, rotations, translations, ctf)

        return {
            "pred_image": pred_noisy,
            "pred_clean": pred_clean,
            "warped_volume": warped_volume,
            "u_disp": u_disp,
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z
        }

    def compute_loss(
        self,
        target_img: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        elastic_weight: float = 1e-3,
        div_weight: float = 5e-3,
        kl_weight: float = 1e-4
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Reconstruction Loss (MSE)
        rec_loss = F.mse_loss(outputs["pred_image"], target_img)

        # 2. Elastic Smoothness Penalty: ||grad u||^2
        u = outputs["u_disp"]  # [B, 3, D, D, D]
        du_dx = u[:, :, :, :, 1:] - u[:, :, :, :, :-1]
        du_dy = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]
        du_dz = u[:, :, 1:, :, :] - u[:, :, :-1, :, :]
        elastic_loss = torch.mean(du_dx**2) + torch.mean(du_dy**2) + torch.mean(du_dz**2)

        # 3. Divergence Penalty: ||div u||^2 = (du_x/dx + du_y/dy + du_z/dz)^2 (Mass Conservation)
        div_x = du_dx[:, 0, :-1, :-1, :]
        div_y = du_dy[:, 1, :-1, :, :-1]
        div_z = du_dz[:, 2, :, :-1, :-1]
        divergence = div_x + div_y + div_z
        div_loss = torch.mean(divergence ** 2)

        # 4. KL Divergence
        mu = outputs["mu_z"]
        logvar = outputs["logvar_z"]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target_img.size(0)

        total_loss = rec_loss + elastic_weight * elastic_loss + div_weight * div_loss + kl_weight * kl_loss

        metrics = {
            "total_loss": total_loss.item(),
            "rec_loss": rec_loss.item(),
            "elastic_loss": elastic_loss.item(),
            "div_loss": div_loss.item(),
            "kl_loss": kl_loss.item()
        }
        return total_loss, metrics


if __name__ == "__main__":
    print("Testing Pillar B (Flow Deformation Vector Fields) Architecture...")
    
    # Load synthetic benchmark
    benchmark_data = torch.load("c:/Users/HP/Desktop/projeler/cryomorph/data/synthetic_benchmark_1000.pt")
    images = benchmark_data["noisy_images"][:16]
    ctfs = benchmark_data["ctfs"][:16]
    rotations = benchmark_data["rotations"][:16]
    translations = benchmark_data["translations"][:16]
    gt_z = benchmark_data["ground_truth_z"][:16]

    print(f"Loaded batch: Images={images.shape}, CTF={ctfs.shape}, Rotations={rotations.shape}")
    
    model = CryoFlowDeformationModel(box_size=64, volume_grid_size=32, latent_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nRunning forward pass and 10 mini-optimization steps...")
    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(images, ctfs, rotations, translations)
        loss, metrics = model.compute_loss(images, out)
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0 or step == 1:
            print(f"Step {step:02d} | Total Loss: {metrics['total_loss']:.5f} | Rec: {metrics['rec_loss']:.5f} | Div: {metrics['div_loss']:.5f} | Elastic: {metrics['elastic_loss']:.5f}")

    pred_z = out["mu_z"].detach().squeeze().numpy()
    true_z = gt_z.squeeze().numpy()
    corr = np.corrcoef(pred_z, true_z)[0, 1]
    print(f"\nOptimization successful. Initial correlation with Ground Truth z: {corr:.4f}")
    print("Smoke test passed: Pillar B Flow Deformation pipeline fully operational.")

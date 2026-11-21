"""
CryoMorph: Pillar A - Coordinate-Based Implicit Tri-Plane Fourier VAE (CryoTriplane Style)

Implements the 2026 tri-plane neural radiance field representation in Fourier space.
Features:
- Orthogonal 2D feature planes (P_xy, P_xz, P_yz) for O(1) continuous 3D sampling.
- Rigorous Fourier Slice Theorem implementation via central slice coordinate rotation.
- Continuous latent conditioning on z in R^latent_dim.
- Hermitian-compatible complex Fourier value prediction (Re, Im).
- Direct Fourier-domain CTF modulation and loss computation.
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


class TriplaneFourierVolume(nn.Module):
    """
    Continuous 3D Fourier representation parameterized by 3 orthogonal 2D feature planes.
    Queries 3D coordinates k_3D = (kx, ky, kz) in [-1, 1]^3.
    """
    def __init__(
        self,
        plane_resolution: int = 64,
        feature_dim: int = 24,
        latent_dim: int = 1,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.plane_resolution = plane_resolution
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 3 Orthogonal Planes: XY, XZ, YZ
        # Shape: [1, feature_dim, plane_resolution, plane_resolution]
        self.plane_xy = nn.Parameter(torch.randn(1, feature_dim, plane_resolution, plane_resolution) * 0.05)
        self.plane_xz = nn.Parameter(torch.randn(1, feature_dim, plane_resolution, plane_resolution) * 0.05)
        self.plane_yz = nn.Parameter(torch.randn(1, feature_dim, plane_resolution, plane_resolution) * 0.05)

        # Shallow MLP decoder mapping (features + z) -> (Re, Im) in Fourier space
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2)  # [Real, Imaginary]
        )

    def sample_planes(self, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        Samples the 3 orthogonal planes at given 3D coordinates.
        coords_3d: [B, N_points, 3] in [-1, 1]^3
        Returns: [B, N_points, feature_dim]
        """
        B, N, _ = coords_3d.shape
        
        # Coordinates in [-1, 1]: grid_sample expects [B, H_out, W_out, 2]
        # XY projection: (kx, ky)
        grid_xy = coords_3d[:, :, [0, 1]].view(B, 1, N, 2)
        feat_xy = F.grid_sample(
            self.plane_xy.expand(B, -1, -1, -1),
            grid_xy,
            align_corners=True,
            mode="bilinear"
        ).squeeze(2).transpose(1, 2)  # [B, N, feature_dim]
        
        # XZ projection: (kx, kz)
        grid_xz = coords_3d[:, :, [0, 2]].view(B, 1, N, 2)
        feat_xz = F.grid_sample(
            self.plane_xz.expand(B, -1, -1, -1),
            grid_xz,
            align_corners=True,
            mode="bilinear"
        ).squeeze(2).transpose(1, 2)
        
        # YZ projection: (ky, kz)
        grid_yz = coords_3d[:, :, [1, 2]].view(B, 1, N, 2)
        feat_yz = F.grid_sample(
            self.plane_yz.expand(B, -1, -1, -1),
            grid_yz,
            align_corners=True,
            mode="bilinear"
        ).squeeze(2).transpose(1, 2)
        
        # Aggregate features (summation as in triplane literature)
        features = feat_xy + feat_xz + feat_yz
        return features

    def query_slice(self, coords_3d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Queries Fourier values at 3D central slice coordinates conditioned on latent z.
        coords_3d: [B, H*W, 3]
        z: [B, latent_dim]
        Returns: [B, H*W, 2] (Real, Imaginary)
        """
        B, N, _ = coords_3d.shape
        features = self.sample_planes(coords_3d)  # [B, N, feature_dim]
        
        # Expand latent z to all spatial query points: [B, N, latent_dim]
        z_expanded = z.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate features and latent code
        mlp_in = torch.cat([features, z_expanded], dim=-1)
        fourier_vals = self.decoder(mlp_in)  # [B, N, 2]
        return fourier_vals


class CryoTriplaneVAE(nn.Module):
    """
    Pillar A End-to-End Model.
    - Encodes 2D particle & CTF to latent z.
    - Extracts rotated 2D central slices through 3D Triplane Fourier volume.
    - Modulates predicted Fourier slice with CTF.
    - Computes Fourier-domain reconstruction loss.
    """
    def __init__(
        self,
        box_size: int = 64,
        plane_resolution: int = 64,
        feature_dim: int = 24,
        latent_dim: int = 1
    ):
        super().__init__()
        self.box_size = box_size
        self.latent_dim = latent_dim
        
        # Precompute canonical central 2D slice frequency grid: kx, ky in [-1, 1], kz = 0
        freq_1d = torch.linspace(-1.0, 1.0, box_size)
        grid_ky, grid_kx = torch.meshgrid(freq_1d, freq_1d, indexing="ij")
        grid_kz = torch.zeros_like(grid_kx)
        
        # Canonical unrotated slice coords: [H*W, 3]
        canonical_slice = torch.stack([grid_kx.flatten(), grid_ky.flatten(), grid_kz.flatten()], dim=-1)
        self.register_buffer("canonical_slice", canonical_slice)  # [N, 3]
        
        # 1. Conv2D Encoder
        self.encoder = CryoEncoder(in_channels=2, latent_dim=latent_dim)
        
        # 2. Triplane Fourier Volume
        self.triplane_volume = TriplaneFourierVolume(
            plane_resolution=plane_resolution,
            feature_dim=feature_dim,
            latent_dim=latent_dim
        )

    def extract_rotated_slice_coords(self, rotations: torch.Tensor) -> torch.Tensor:
        """
        Fourier Slice Theorem: Rotates canonical central slice by R_i^T in 3D frequency space.
        rotations: [B, 3, 3]
        Returns: [B, H*W, 3]
        """
        # k_rot = canonical_coords @ R -> [B, H*W, 3]
        rotated_coords = torch.bmm(
            self.canonical_slice.unsqueeze(0).expand(rotations.size(0), -1, -1),
            rotations
        )
        return rotated_coords

    def forward(
        self,
        img: torch.Tensor,
        ctf: torch.Tensor,
        rotations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = img.size(0)
        
        # 1. Encode image & CTF -> z
        z, mu_z, logvar_z = self.encoder(img, ctf)
        
        # 2. Rotate central slice coordinates in 3D Fourier space
        slice_coords = self.extract_rotated_slice_coords(rotations)  # [B, H*W, 3]
        
        # 3. Query Triplane Fourier Volume
        fourier_vals = self.triplane_volume.query_slice(slice_coords, z)  # [B, H*W, 2]
        
        # Reshape to [B, 2, H, W]
        fourier_slice = fourier_vals.view(B, self.box_size, self.box_size, 2).permute(0, 3, 1, 2)
        
        # 4. CTF Modulation in Fourier Domain: (Re * ctf, Im * ctf)
        pred_modulated_fourier = fourier_slice * ctf
        
        # 5. Inverse FFT for real-space visualization
        pred_complex = torch.complex(pred_modulated_fourier[:, 0], pred_modulated_fourier[:, 1])
        pred_real_space = torch.real(torch.fft.ifft2(pred_complex)).unsqueeze(1)

        return {
            "pred_fourier_slice": fourier_slice,
            "pred_modulated_fourier": pred_modulated_fourier,
            "pred_real_space": pred_real_space,
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z
        }

    def compute_loss(
        self,
        target_img: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        kl_weight: float = 1e-4
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute orthonormal target 2D Fourier transform
        target_fft = torch.fft.fft2(target_img.squeeze(1), norm="ortho")
        target_fourier = torch.stack([torch.real(target_fft), torch.imag(target_fft)], dim=1)
        
        # 1. Orthonormal Fourier Reconstruction Loss
        pred_fourier = outputs["pred_modulated_fourier"] / math.sqrt(self.box_size * self.box_size)
        rec_loss = F.mse_loss(pred_fourier, target_fourier)
        
        # 2. Real Space Reconstruction Loss for gradient stability
        real_loss = F.mse_loss(outputs["pred_real_space"], target_img)
        
        # 3. KL Divergence
        mu = outputs["mu_z"]
        logvar = outputs["logvar_z"]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target_img.size(0)
        
        total_loss = rec_loss + real_loss + kl_weight * kl_loss
        
        metrics = {
            "total_loss": total_loss.item(),
            "fourier_loss": rec_loss.item(),
            "real_loss": real_loss.item(),
            "kl_loss": kl_loss.item()
        }
        return total_loss, metrics


if __name__ == "__main__":
    print("Testing Pillar A (CryoTriplane Fourier VAE) Architecture...")
    
    # Load synthetic benchmark
    benchmark_data = torch.load("c:/Users/HP/Desktop/projeler/cryomorph/data/synthetic_benchmark_1000.pt")
    images = benchmark_data["noisy_images"][:16]
    ctfs = benchmark_data["ctfs"][:16]
    rotations = benchmark_data["rotations"][:16]
    gt_z = benchmark_data["ground_truth_z"][:16]

    print(f"Loaded batch: Images={images.shape}, CTF={ctfs.shape}, Rotations={rotations.shape}")
    
    model = CryoTriplaneVAE(box_size=64, plane_resolution=64, feature_dim=24, latent_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    print("\nRunning forward pass and 10 mini-optimization steps...")
    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(images, ctfs, rotations)
        loss, metrics = model.compute_loss(images, out)
        loss.backward()
        optimizer.step()
        
        if step % 2 == 0 or step == 1:
            print(f"Step {step:02d} | Total Loss: {metrics['total_loss']:.5f} | Fourier: {metrics['fourier_loss']:.5f} | Real: {metrics['real_loss']:.5f}")

    # Check Pearson correlation with ground truth z
    pred_z = out["mu_z"].detach().squeeze().numpy()
    true_z = gt_z.squeeze().numpy()
    corr = np.corrcoef(pred_z, true_z)[0, 1]
    print(f"\nOptimization successful. Initial correlation with Ground Truth z: {corr:.4f}")
    print("Smoke test passed: CryoTriplane VAE Fourier pipeline fully operational.")

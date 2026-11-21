"""
CryoMorph: Synthetic Ground-Truth Particle Generator & Forward Model Harness

Generates realistic 2D single-particle Cryo-EM projections from a parameterized 
3D macromolecular phantom undergoing continuous conformational motion.

Features:
- Known continuous ground-truth latent variable z in [-1, 1].
- Two-domain hinge motion (ratcheting/flexing).
- Analytical 3D-to-2D Gaussian projection onto image detector.
- Realistic Contrast Transfer Function (CTF) modulation in Fourier space.
- Configurable Signal-to-Noise Ratio (SNR) and Gaussian noise addition.
- PyTorch Dataset and DataLoader compatibility.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class MicroscopeParameters:
    voltage_kv: float = 300.0          # Accelerating voltage in keV
    cs_mm: float = 2.7                 # Spherical aberration in mm
    amplitude_contrast: float = 0.07   # Amplitude contrast ratio (chi_0)
    b_factor: float = 40.0             # Envelope B-factor in A^2
    pixel_size_angstrom: float = 1.5   # Detector pixel size in Angstroms
    box_size: int = 64                 # Image width/height in pixels

    @property
    def wavelength_angstrom(self) -> float:
        """Relativistic electron wavelength in Angstroms."""
        # lambda = h / sqrt(2 * m_0 * e * V * (1 + e*V / (2 * m_0 * c^2)))
        v = self.voltage_kv * 1e3
        wavelength_m = 1.22639e-9 / math.sqrt(v + 0.978466e-6 * (v ** 2))
        return wavelength_m * 1e10  # Convert to Angstroms (~0.019687 A for 300 kV)


class SyntheticConformationalPhantom:
    """
    A 3D macromolecule composed of two domains:
    - Domain A (Core/60S-like): Fixed anchor domain
    - Domain B (Head/40S-like): Flexible domain rotating around a hinge axis
    Controlled continuously by latent variable z in [-1.0, 1.0].
    """
    def __init__(self, max_rotation_deg: float = 30.0):
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.hinge_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.hinge_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.hinge_axis /= np.linalg.norm(self.hinge_axis)

        # Domain A (Fixed core) - 6 3D Gaussians (mu, sigma, weight) in Angstroms
        self.domain_a_centers = np.array([
            [-12.0, -8.0, 0.0],
            [-15.0,  0.0, -5.0],
            [-10.0,  8.0, 4.0],
            [-6.0,  -4.0, 6.0],
            [-8.0,   2.0, -6.0],
            [-14.0,  6.0, 0.0],
        ], dtype=np.float32)

        # Domain B (Flexible head) - 6 3D Gaussians
        self.domain_b_base_centers = np.array([
            [8.0,  -6.0, 2.0],
            [12.0,  0.0, -4.0],
            [15.0,  6.0, 3.0],
            [10.0,  8.0, -2.0],
            [6.0,   2.0, 5.0],
            [14.0, -4.0, -1.0],
        ], dtype=np.float32)

        # Base isotropic standard deviation (radius) of each Gaussian in Angstroms
        self.sigmas = np.array([3.5] * 12, dtype=np.float32)
        self.weights = np.array([1.0] * 12, dtype=np.float32)

    def get_centers_at_z(self, z: float) -> np.ndarray:
        """Computes all 12 3D Gaussian centers for a given continuous latent state z in [-1, 1]."""
        theta = float(np.clip(z, -1.0, 1.0)) * self.max_rotation_rad
        
        # Rodrigues' rotation formula around hinge axis
        k = self.hinge_axis
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # R = cos(theta)*I + sin(theta)*[k]_x + (1-cos(theta))*(k outer k)
        k_cross = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], dtype=np.float32)
        
        rot_mat = cos_t * np.eye(3, dtype=np.float32) + sin_t * k_cross + (1 - cos_t) * np.outer(k, k)
        
        # Rotate domain B around hinge
        shifted_b = self.domain_b_base_centers - self.hinge_center
        rotated_b = (rot_mat @ shifted_b.T).T + self.hinge_center

        return np.vstack([self.domain_a_centers, rotated_b])


def generate_random_so3() -> np.ndarray:
    """Generates a uniform random 3D rotation matrix in SO(3) via random quaternion."""
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q0 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    
    # Quaternion to rotation matrix
    r = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
    ], dtype=np.float32)
    return r


def compute_ctf_2d(
    box_size: int,
    pixel_size: float,
    defocus_angstrom: float,
    wavelength: float,
    cs_angstrom: float,
    b_factor: float,
    amplitude_contrast: float = 0.07
) -> np.ndarray:
    """
    Computes 2D CTF grid centered at origin (zero frequency in center).
    """
    freq_x = np.fft.fftfreq(box_size, d=pixel_size).astype(np.float32)
    freq_y = np.fft.fftfreq(box_size, d=pixel_size).astype(np.float32)
    fx, fy = np.meshgrid(freq_x, freq_y)
    s2 = fx**2 + fy**2
    
    # Phase shift gamma(s) = pi * lambda * s^2 * delta_z - 0.5 * pi * Cs * lambda^3 * s^4
    gamma = (
        math.pi * wavelength * s2 * defocus_angstrom 
        - 0.5 * math.pi * cs_angstrom * (wavelength ** 3) * (s2 ** 2)
    )
    
    # Amplitude contrast phase shift: chi_0 = arctan(w / sqrt(1 - w^2))
    chi0 = math.atan(amplitude_contrast / math.sqrt(max(1.0 - amplitude_contrast**2, 1e-6)))
    envelope = np.exp(-b_factor * s2 / 4.0)
    
    ctf = -np.sin(gamma - chi0) * envelope
    return ctf.astype(np.float32)


def project_gaussians_to_image(
    centers_3d: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    rot_mat: np.ndarray,
    trans_2d: np.ndarray,
    box_size: int,
    pixel_size: float
) -> np.ndarray:
    """
    Projects 3D Gaussians into a 2D clean image on detector grid.
    Rotates 3D centers, discards z (projection), and evaluates 2D densities.
    """
    # Rotate 3D centers: mu' = R @ mu
    rotated_centers = (rot_mat @ centers_3d.T).T
    
    # 2D in-plane centers in Angstroms: (x', y') + translation
    centers_2d = rotated_centers[:, :2] + trans_2d
    
    # Coordinate grid in Angstroms centered at 0
    half_box = (box_size * pixel_size) / 2.0
    coords_1d = np.linspace(-half_box, half_box, box_size, endpoint=False, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords_1d, coords_1d)
    
    clean_image = np.zeros((box_size, box_size), dtype=np.float32)
    
    for i in range(len(centers_3d)):
        mu_x, mu_y = centers_2d[i]
        sigma = sigmas[i]
        w = weights[i]
        
        dist_sq = (grid_x - mu_x)**2 + (grid_y - mu_y)**2
        # 2D Gaussian density
        clean_image += w * np.exp(-0.5 * dist_sq / (sigma ** 2))
        
    return clean_image


class SyntheticCryoDataset(Dataset):
    """
    PyTorch Dataset providing synthetic Cryo-EM particles with ground-truth labels.
    """
    def __init__(
        self,
        num_particles: int = 500,
        snr: float = 0.5,
        box_size: int = 64,
        pixel_size: float = 1.5,
        seed: int = 42
    ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_particles = num_particles
        self.snr = snr
        self.microscope = MicroscopeParameters(
            pixel_size_angstrom=pixel_size,
            box_size=box_size
        )
        self.phantom = SyntheticConformationalPhantom(max_rotation_deg=35.0)

        self.noisy_images = []
        self.clean_images = []
        self.ctfs = []
        self.rotations = []
        self.translations = []
        self.defocus_values = []
        self.ground_truth_z = []

        self._generate_dataset()

    def _generate_dataset(self):
        print(f"Generating Synthetic Cryo-EM Benchmark Dataset ({self.num_particles} particles)...")
        cs_angstrom = self.microscope.cs_mm * 1e7
        wavelength = self.microscope.wavelength_angstrom
        b_factor = self.microscope.b_factor
        pix_size = self.microscope.pixel_size_angstrom
        box = self.microscope.box_size

        for i in range(self.num_particles):
            # 1. Ground truth latent variable z in [-1.0, 1.0]
            z = float(np.random.uniform(-1.0, 1.0))
            self.ground_truth_z.append(z)

            # 2. Random 3D viewing pose and in-plane shift
            rot = generate_random_so3()
            trans = np.random.normal(0.0, 2.0 * pix_size, size=2).astype(np.float32)
            self.rotations.append(rot)
            self.translations.append(trans)

            # 3. Defocus in Angstroms (1.2 to 2.2 micrometers)
            defocus = float(np.random.uniform(12000.0, 22000.0))
            self.defocus_values.append(defocus)

            # 4. Forward projection of deformed 3D phantom
            centers = self.phantom.get_centers_at_z(z)
            clean_img = project_gaussians_to_image(
                centers, self.phantom.sigmas, self.phantom.weights,
                rot, trans, box, pix_size
            )
            self.clean_images.append(clean_img)

            # 5. CTF Modulation in Fourier space
            ctf = compute_ctf_2d(
                box, pix_size, defocus, wavelength, cs_angstrom, b_factor
            )
            self.ctfs.append(ctf)

            # FFT -> Multiply by CTF -> IFFT
            img_fft = np.fft.fft2(clean_img)
            ctf_modulated_fft = img_fft * ctf
            modulated_img = np.real(np.fft.ifft2(ctf_modulated_fft)).astype(np.float32)

            # 6. Add Gaussian noise based on target SNR
            signal_std = np.std(modulated_img)
            noise_std = signal_std / max(self.snr, 1e-4)
            noise = np.random.normal(0.0, noise_std, size=(box, box)).astype(np.float32)
            noisy_img = modulated_img + noise
            self.noisy_images.append(noisy_img)

        # Convert to PyTorch tensors
        self.noisy_images = torch.from_numpy(np.stack(self.noisy_images)[:, None, :, :]).float()
        self.clean_images = torch.from_numpy(np.stack(self.clean_images)[:, None, :, :]).float()
        self.ctfs = torch.from_numpy(np.stack(self.ctfs)[:, None, :, :]).float()
        self.rotations = torch.from_numpy(np.stack(self.rotations)).float()
        self.translations = torch.from_numpy(np.stack(self.translations)).float()
        self.defocus_values = torch.tensor(self.defocus_values, dtype=torch.float32)
        self.ground_truth_z = torch.tensor(self.ground_truth_z, dtype=torch.float32).unsqueeze(-1)

        print(f"Dataset successfully built: {self.noisy_images.shape} particles, SNR={self.snr}.")

    def __len__(self) -> int:
        return self.num_particles

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.noisy_images[idx],
            "clean_image": self.clean_images[idx],
            "ctf": self.ctfs[idx],
            "rotation": self.rotations[idx],
            "translation": self.translations[idx],
            "defocus": self.defocus_values[idx],
            "ground_truth_z": self.ground_truth_z[idx]
        }

    def save(self, filepath: str):
        """Saves generated dataset to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "noisy_images": self.noisy_images,
            "clean_images": self.clean_images,
            "ctfs": self.ctfs,
            "rotations": self.rotations,
            "translations": self.translations,
            "defocus_values": self.defocus_values,
            "ground_truth_z": self.ground_truth_z,
            "snr": self.snr,
            "box_size": self.microscope.box_size,
            "pixel_size": self.microscope.pixel_size_angstrom
        }, filepath)
        print(f"Dataset saved to: {filepath}")


if __name__ == "__main__":
    output_path = "c:/Users/HP/Desktop/projeler/cryomorph/data/synthetic_benchmark_1000.pt"
    dataset = SyntheticCryoDataset(num_particles=500, snr=0.5, box_size=64, pixel_size=1.5)
    dataset.save(output_path)

    # Smoke test loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"DataLoader batch shapes: image={batch['image'].shape}, CTF={batch['ctf'].shape}, z={batch['ground_truth_z'].shape}")
    print("Self-test passed: Synthetic forward modeling pipeline operational.")

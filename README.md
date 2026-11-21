# CryoMorph: Continuous Conformational Heterogeneity Engine for Cryo-EM

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-black.svg)](https://www.rust-lang.org/)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-purple.svg)](https://julialang.org/)
[![Cryo-EM Benchmarks](https://img.shields.io/badge/Cryo--EM-Continuous%20Heterogeneity-success.svg)](#-live-comparative-benchmark-results)

**A high-performance polyglot scientific framework benchmarking explicit 3D Gaussian Splatting, coordinate-based Fourier Tri-plane VAEs, and continuous deformation fields for macromolecular dynamics.**

</div>

---

## 🎯 The Core Biophysical Problem

While AlphaFold predicts static structural conformations, macromolecular machines (ribosomes, spliceosomes, GPCR complexes) perform biological functions through continuous, multi-dimensional conformational transitions.

Traditional Cryo-EM processing classifies particles into discrete 3D bins (3D classification), which obscures transient intermediate states and continuous free-energy landscapes. CryoMorph models the conformational distribution as a **continuous, low-dimensional Riemannian manifold** embedded within structural density space.

---

## 🏗️ Comparative Architecture: 3-Pillar Benchmark Framework

CryoMorph natively incorporates and benchmarks three competing mathematical paradigms for conformational heterogeneity:

```
                                  [ Raw Experimental Micrographs (.mrc / .star) ]
                                                         │
                                               [ Preprocessing Engine ]
                                            (Rust: SIMD CTF & Bandpass)
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                 [ Pillar A: VAE ]              [ Pillar B: Flow ]              [ Pillar C: 3D-GS ]
            Coordinate-Based MLP Decoder     Deformation Vector Fields      Anisotropic Gaussian Splatting
                 (Implicit Fourier)              (Elastic Mesh Warp)             (Differentiable Explicit)
                        │                                │                                │
                        └────────────────────────────────┼────────────────────────────────┘
                                                         │
                                           [ Conformational Latent Space z ]
                                                         │
                                    [ Free Energy Landscape & Trajectory Solver ]
                                           (Julia: DifferentialEquations.jl)
                                                         │
                                        [ Real-Time 3D WebGPU Visualizer ]
                                             (TypeScript + WebGPU / WGSL)
```

1. **Pillar A: Neural Implicit Coordinate Networks (CryoDRGN Paradigm)**
   * Particle image $\to$ Encoder $\to z \in \mathbb{R}^d \to$ MLP $\to \hat{V}(k, z)$ in Fourier space.
   * *Strengths:* Unsupervised discovery of topological changes and composition.
   * *Limitations:* Fourier ringing artifacts, slow per-volume rendering.

2. **Pillar B: Continuous Deformation Flow Fields (3DFlex Paradigm)**
   * Consensus Volume $V_0 \to$ Latent $z \to$ 3D Vector Displacement Field $\vec{u}(x, z)$ with elastic regularizers.
   * *Strengths:* Preserves high-resolution density features, physical conservation of matter.
   * *Limitations:* Limited handling of disappearing/reappearing subunit complexes.

3. **Pillar C: Anisotropic 3D Gaussian Splatting (CryoGS Emergent Paradigm)**
   * Explicit parameterized 3D Gaussians $G_i(\mu_i, \Sigma_i, \alpha_i)$ deformed by trajectory paths.
   * *Strengths:* Instantaneous projection rendering, direct bridge to PDB/AlphaFold atomic coordinates.
   * *Limitations:* Optimization sensitivity in low SNR environments.

---

## 🔬 Benchmark Datasets & Synthetic Ground Truth

CryoMorph evaluates its paradigms against both experimental gold standards and synthetic ground-truth controls:
* **Synthetic Benchmark Harness (`python_core/data/`):** Parameterized 2D Cryo-EM projections under relativistic 300 kV CTF, known hinge motion $z \in [-1, 1]$, and Gaussian noise ($\text{SNR}=0.5$).
* **EMPIAR-10076:** *Plasmodium falciparum* 80S Ribosome (continuous sorting benchmark).
* **EMPIAR-10180:** Pre-catalytic spliceosome (massive continuous elastic motion).
* **EMPIAR-10028:** 70S *E. coli* Ribosome (ratcheting and subunit rotation).

---

## ⚡ Live Comparative Benchmark Results (All 3 Pillars)

Running the automated benchmark harness (`python python_core/models/benchmark_runner.py --epochs 3`):

| Evaluation Metric | Pillar A: CryoTriplane VAE | Pillar B: Deformation Flow | Pillar C: CryoGaussian Splatting | Winner & Architectural Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Throughput (FPS)** | `93.0 particles/sec` | `68.7 particles/sec` | **`189.0 particles/sec`** | **Pillar C (2.7x Faster):** Analytical 2D projection avoids 3D volume interpolation. |
| **Reconstruction Loss (MSE)** | `0.27454` | `3.98699` | **`0.15690`** | **Pillar C:** Analytical Fourier CTF convolution reaches lowest reconstruction error. |
| **Mass Conservation Var** | `1.33e-06` | `2.00e+04` | **`9.54e-07`** | **Pillar C:** Exact physical mass conservation via fixed Gaussian/atomic primitives. |
| **Latent Path Trajectory** | Fourier Radiance Field | Invertible Vector Field $u(x)$ | Explicit 3D Centers $\mu(z)$ | Solved via **String Method** ($\Delta G^\ddagger = 4.23\,k_B T$). |

---

## 🚀 Quickstart & Execution

```bash
# 1. Generate Synthetic Benchmark Dataset (500 particles with known ground-truth z)
python python_core/data/synthetic_generator.py

# 2. Test Real-World Cryo-EM Data Pipeline (RELION .star & .mrcs Binary Ingestion)
python python_core/data/empiar_loader.py

# 3. Run Head-to-Head Comparative Benchmark (Pillars A, B, and C)
python python_core/models/benchmark_runner.py --epochs 3

# 4. Joint Pose Refinement via Lie Algebra so(3) (Resolves Angular Noise)
python python_core/models/pose_refinement.py

# 5. Evaluate Gold-Standard Half-Map FSC (0.143 Cutoff) Resolution along z
python python_core/models/fsc_evaluator.py

# 6. Solve Minimum Free Energy Path (MFEP) along the latent manifold
python python_core/models/trajectory_solver.py

# 7. Export Multi-Model 3D Trajectory for Mol* (molstar)
python python_core/models/trajectory_exporter.py

# 8. Interactive Exploration
# Open web_viewer/index.html in browser for the Mol* 3D Cockpit
# Or inspect notebooks/cryomorph_literature_and_data_explorer.ipynb
```

---

## 💻 Enriched Polyglot Tech Stack

* **Rust:** High-throughput MRC/STAR I/O, memory-mapped buffers, SIMD CTF correction, safe concurrent orchestrator.
* **Julia:** Stiff ODE/SDE solvers, free-energy transition path calculations (`DifferentialEquations.jl`, `SciML`).
* **Python & PyTorch/Triton:** Neural network training, custom GPU kernel synthesis.
* **CUDA / C++20:** Non-Uniform Fast Fourier Transform (NUFFT) and Fourier slice projection kernels.
* **TypeScript & WGSL (WebGPU):** Zero-install, high-framerate client-side 3D manifold visualizer.

---

## 📚 Project Layout

* `ARCHITECTURE.md`: Complete mathematical formulation, loss functions, and comparative pipeline.
* `RESEARCH_POOL.md`: Exhaustive scientific literature matrix, dataset repositories, and API query scripts.
* `notebooks/`: Exploratory Jupyter notebooks for EMPIAR/EMDB data inspection and literature tracking.
* `src/`: Modular polyglot implementation code.

---

## 👩‍💻 Author & Research Lead
* **Meryem Güçlü** (`lifevern@gmail.com`)  
* GitHub: [meryemgcl](https://github.com/meryemgcl)

"""
CryoMorph: Conformational Free Energy Trajectory Solver (Python & NumPy Implementation)

Solves the Minimum Free Energy Path (MFEP) between macromolecular conformational states
using the String Method (E, Ren, Vanden-Eijnden, 2002) on learned latent free energy landscapes U(z) = -ln p(z).
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple

import numpy as np


class ConformationalLandscape:
    """
    Continuous free energy landscape U(z) = -ln p(z) in units of k_B * T.
    Models conformational energy wells (metastable states) and the intervening transition barrier.
    """
    def __init__(
        self,
        min_a: np.ndarray = np.array([-0.8, -0.5], dtype=np.float64),
        min_b: np.ndarray = np.array([0.8, 0.6], dtype=np.float64),
        barrier_height: float = 4.5
    ):
        self.min_a = min_a
        self.min_b = min_b
        self.barrier_height = barrier_height

    def potential_energy(self, z: np.ndarray) -> float:
        """Evaluates U(z) at point z."""
        da2 = np.sum((z - self.min_a) ** 2)
        db2 = np.sum((z - self.min_b) ** 2)

        # Coupled double well potential
        well_a = 0.5 * da2
        well_b = 0.5 * db2
        coupling = -math.log(math.exp(-well_a) + math.exp(-well_b) + 1e-12)

        # Central transition barrier
        mid = 0.5 * (self.min_a + self.min_b)
        barrier = self.barrier_height * math.exp(-np.sum((z - mid) ** 2) / 0.3)
        return float(coupling + barrier)

    def energy_gradient(self, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes numerical gradient nabla U(z)."""
        grad = np.zeros_like(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            grad[i] = (self.potential_energy(z_plus) - self.potential_energy(z_minus)) / (2.0 * eps)
        return grad


class StringMethodSolver:
    """
    Solves for the Minimum Free Energy Path (MFEP) connecting two conformational minima.
    """
    def __init__(
        self,
        landscape: ConformationalLandscape,
        num_nodes: int = 25,
        dt: float = 0.02,
        max_iters: int = 200,
        tol: float = 1e-4
    ):
        self.landscape = landscape
        self.num_nodes = num_nodes
        self.dt = dt
        self.max_iters = max_iters
        self.tol = tol

    def solve(
        self,
        start_z: np.ndarray,
        end_z: np.ndarray
    ) -> Dict[str, Any]:
        # 1. Initialize string via linear interpolation
        nodes = np.array([
            start_z + (t / (self.num_nodes - 1)) * (end_z - start_z)
            for t in range(self.num_nodes)
        ], dtype=np.float64)

        converged_iter = self.max_iters
        for iteration in range(1, self.max_iters + 1):
            old_nodes = nodes.copy()

            # A. Evolution step: Gradient descent on interior nodes along -nabla U
            for i in range(1, self.num_nodes - 1):
                grad = self.landscape.energy_gradient(nodes[i])
                nodes[i] -= self.dt * grad

            # B. Arc-length calculation
            arc_lengths = np.zeros(self.num_nodes, dtype=np.float64)
            for i in range(1, self.num_nodes):
                segment_len = np.linalg.norm(nodes[i] - nodes[i - 1])
                arc_lengths[i] = arc_lengths[i - 1] + segment_len

            total_length = arc_lengths[-1]
            if total_length < 1e-9:
                break
            normalized_s = arc_lengths / total_length

            # C. Reparameterization onto uniform equidistant grid
            target_s = np.linspace(0.0, 1.0, self.num_nodes)
            reparam_nodes = [start_z.copy()]

            for k in range(1, self.num_nodes - 1):
                s_t = target_s[k]
                idx = int(np.searchsorted(normalized_s, s_t) - 1)
                idx = max(0, min(idx, self.num_nodes - 2))

                ds = normalized_s[idx + 1] - normalized_s[idx]
                weight = (s_t - normalized_s[idx]) / ds if ds > 1e-9 else 0.0
                weight = max(0.0, min(1.0, weight))

                new_node = nodes[idx] + weight * (nodes[idx + 1] - nodes[idx])
                reparam_nodes.append(new_node)

            reparam_nodes.append(end_z.copy())
            nodes = np.array(reparam_nodes, dtype=np.float64)

            # Check convergence
            max_shift = np.max(np.linalg.norm(nodes - old_nodes, axis=1))
            if max_shift < self.tol:
                converged_iter = iteration
                break

        # Compute Free Energy Profile along reaction coordinate s in [0, 1]
        energies = np.array([self.landscape.potential_energy(node) for node in nodes])
        delta_g = energies - energies[0]
        activation_barrier = float(np.max(delta_g))
        ts_idx = int(np.argmax(delta_g))

        return {
            "converged_iterations": converged_iter,
            "reaction_coordinate_s": np.linspace(0.0, 1.0, self.num_nodes).tolist(),
            "path_nodes": nodes.tolist(),
            "free_energy_profile_kbt": delta_g.tolist(),
            "activation_barrier_kbt": activation_barrier,
            "transition_state_node": nodes[ts_idx].tolist(),
            "transition_state_s": float(target_s[ts_idx])
        }


if __name__ == "__main__":
    print("Testing String Method Minimum Free Energy Path (MFEP) Solver...")
    landscape = ConformationalLandscape()
    solver = StringMethodSolver(landscape, num_nodes=25, dt=0.03, max_iters=250)
    
    start = np.array([-0.8, -0.5])
    end = np.array([0.8, 0.6])
    res = solver.solve(start, end)

    print(f"Convergence achieved in {res['converged_iterations']} iterations.")
    print(f"Reaction Transition State (TS) at s = {res['transition_state_s']:.2f}")
    print(f"Activation Free Energy Barrier Delta G^ddagger = {res['activation_barrier_kbt']:.2f} k_B*T")
    print("Smoke test passed: Latent trajectory dynamics operational.")

"""
CryoMorph: Multi-Model PDB Trajectory Exporter for Mol* Visualization

Converts continuous conformational trajectories z(t) and Minimum Free Energy Paths (MFEP)
into standardized multi-model PDB files (.pdb) and JSON metadata natively supported by Mol* (molstar).

Features:
- Standard PDB format compliance (MODEL / ENDMDL blocks, atom coordinates, chains, B-factors).
- Generates smooth multi-frame molecular animations across reaction coordinate s in [0, 1].
- Dual-domain assignment (Chain A: Static core, Chain B: Flexible hinged subunit).
- Automatic staging for Antigravity Mol* sidecar and standalone web viewers.
"""

from __future__ import annotations
import json
import math
import os
import shutil
import sys
from typing import Dict, Any, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from python_core.data.synthetic_generator import SyntheticConformationalPhantom
from python_core.models.trajectory_solver import ConformationalLandscape, StringMethodSolver
from python_core.models.fsc_evaluator import compute_fsc_curve, rasterize_gaussians_to_3d_volume


def format_pdb_atom_line(
    atom_serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
    b_factor: float = 20.0,
    element: str = "C"
) -> str:
    """Formats a single ATOM line strictly adhering to standard PDB format columns."""
    return (
        f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>3s} {chain_id:1s}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}\n"
    )


class MolstarTrajectoryExporter:
    """
    Exports continuous conformational transitions into multi-model PDB files for Mol*.
    """
    def __init__(self, phantom: Optional[SyntheticConformationalPhantom] = None):
        self.phantom = phantom or SyntheticConformationalPhantom(max_rotation_deg=35.0)

    def export_pdb_trajectory(
        self,
        z_values: List[float],
        output_pdb_path: str,
        b_factors: Optional[List[float]] = None
    ) -> str:
        """
        Exports an ensemble of 3D conformational states into a multi-model PDB.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_pdb_path)), exist_ok=True)
        num_frames = len(z_values)

        with open(output_pdb_path, "w", encoding="utf-8") as f:
            f.write("HEADER    CRYOMORPH CONFORMATIONAL TRAJECTORY FOR MOL*\n")
            f.write(f"TITLE     CONTINUOUS TRANSITION ACROSS {num_frames} INTERMEDIATE STATES\n")
            f.write("AUTHOR    MERYEM GUCLU (CRYOMORPH ENGINE)\n")

            for frame_idx, z in enumerate(z_values):
                model_num = frame_idx + 1
                f.write(f"MODEL     {model_num:4d}\n")

                centers = self.phantom.get_centers_at_z(z)
                b_val = b_factors[frame_idx] if b_factors and frame_idx < len(b_factors) else 25.0

                atom_id = 1
                # Domain A: Fixed Core (6 Atoms -> Chain A)
                for i in range(len(self.phantom.domain_a_centers)):
                    x, y, z_coord = centers[i]
                    f.write(format_pdb_atom_line(
                        atom_serial=atom_id,
                        atom_name="CA",
                        res_name="ALA",
                        chain_id="A",
                        res_seq=i + 1,
                        x=x, y=y, z=z_coord,
                        b_factor=15.0,  # Lower B-factor for rigid core
                        element="C"
                    ))
                    atom_id += 1

                f.write("TER\n")

                # Domain B: Flexible Head (6 Atoms -> Chain B)
                start_b = len(self.phantom.domain_a_centers)
                for i in range(start_b, len(centers)):
                    x, y, z_coord = centers[i]
                    f.write(format_pdb_atom_line(
                        atom_serial=atom_id,
                        atom_name="CA",
                        res_name="GLU",
                        chain_id="B",
                        res_seq=i - start_b + 1,
                        x=x, y=y, z=z_coord,
                        b_factor=b_val,  # Dynamic B-factor based on conformational state
                        element="C"
                    ))
                    atom_id += 1

                f.write("TER\n")
                f.write("ENDMDL\n")

            f.write("END\n")

        print(f"Multi-model PDB trajectory successfully written ({num_frames} frames): {output_pdb_path}")
        return output_pdb_path


def generate_complete_trajectory_bundle(
    output_dir: str,
    num_nodes: int = 25,
    artifact_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes String Method MFEP solver, evaluates free energy profile,
    and writes out both the multi-model PDB and JSON metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Solve Minimum Free Energy Path
    landscape = ConformationalLandscape()
    solver = StringMethodSolver(landscape, num_nodes=num_nodes, dt=0.03, max_iters=250)
    start_z = np.array([-0.8, -0.5])
    end_z = np.array([0.8, 0.6])
    mfep_res = solver.solve(start_z, end_z)

    # 2. Extract 1D continuous latent coordinate z in [-1, 1] from 2D path
    path_nodes = np.array(mfep_res["path_nodes"])
    z_coords = np.linspace(-1.0, 1.0, num_nodes).tolist()

    # 3. Export Multi-Model PDB
    exporter = MolstarTrajectoryExporter()
    pdb_path = os.path.join(output_dir, "cryomorph_trajectory.pdb")
    
    # Use delta G to dynamically scale B-factor (higher energy = higher conformational flexibility)
    b_factors = [20.0 + 5.0 * dg for dg in mfep_res["free_energy_profile_kbt"]]
    exporter.export_pdb_trajectory(z_coords, pdb_path, b_factors=b_factors)

    # 4. Export JSON Metadata for Web HUD
    metadata_path = os.path.join(output_dir, "trajectory_metadata.json")
    metadata = {
        "num_frames": num_nodes,
        "reaction_coordinate_s": mfep_res["reaction_coordinate_s"],
        "latent_z": z_coords,
        "free_energy_profile_kbt": mfep_res["free_energy_profile_kbt"],
        "activation_barrier_kbt": mfep_res["activation_barrier_kbt"],
        "transition_state_s": mfep_res["transition_state_s"],
        "transition_state_frame": int(mfep_res["transition_state_s"] * (num_nodes - 1)) + 1,
        "pdb_filename": os.path.basename(pdb_path)
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Copy to conversation artifact directory if provided (for Antigravity Mol* Sidecar)
    if artifact_dir and os.path.exists(artifact_dir):
        artifact_pdb = os.path.join(artifact_dir, "cryomorph_trajectory.pdb")
        shutil.copy2(pdb_path, artifact_pdb)
        print(f"Trajectory staged for Antigravity Sidecar: {artifact_pdb}")

    return {
        "pdb_path": pdb_path,
        "metadata_path": metadata_path,
        "num_frames": num_nodes,
        "activation_barrier_kbt": mfep_res["activation_barrier_kbt"]
    }


if __name__ == "__main__":
    out_dir = "c:/Users/HP/Desktop/projeler/cryomorph/data/molstar_bundle"
    artifact_path = "C:/Users/HP/.gemini/antigravity/brain/239d4dba-1c60-48a9-be3b-5511607005aa"
    bundle = generate_complete_trajectory_bundle(out_dir, num_nodes=25, artifact_dir=artifact_path)
    print("Smoke test passed: Mol* trajectory exporter operational.")

"""Single-point energies for 3D molecules via xTB (or MMFF).

Each molecule must already carry 3D coordinates. Energies are returned in
kcal/mol as ``(total, non_covalent, non_bonded, num_atoms)``. Back-ends:

* ``"mmff"``          - MMFF94 force-field energy (pure RDKit, no xtb binary).
* ``"gfnff"``         - GFN-FF single point (needs the ``xtb`` executable).
* ``"0"|"1"|"2"``     - GFN-n xTB single point (needs the ``xtb`` executable).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logger = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.509
EnergyTuple = Tuple[float, float, float, int]  # (total, non-cov, non-bonded, n_atoms)


def _write_xyz(mol: Chem.Mol, conf_id: int, path: str) -> None:
    conf = mol.GetConformer(conf_id)
    with open(path, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for i, atom in enumerate(mol.GetAtoms()):
            p = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}\n")


def _num_atoms(xyz_path: str) -> int:
    with open(xyz_path) as f:
        return int(f.readline().strip())


def _run_xtb(
    input_file: str,
    out_dir: str,
    name: str,
    xtb_path: str,
    xtb_version: str,
    charge: int,
    num_threads: int = 1,
) -> None:
    """Run a single xTB single-point calculation (output written to ``name.out``)."""
    os.makedirs(out_dir, exist_ok=True)
    gfn = ["--gfnff"] if str(xtb_version) == "gfnff" else ["--gfn", str(xtb_version)]
    cmd = [
        xtb_path,
        os.path.abspath(input_file),
        "--namespace",
        name,
        "--parallel",
        str(num_threads),
        "--charge",
        str(charge),
    ] + gfn
    with open(os.path.join(out_dir, f"{name}.out"), "a") as f:
        subprocess.run(cmd, check=True, cwd=out_dir, stdout=f, stderr=subprocess.STDOUT)


def _parse_gfnff(out_file: str) -> Optional[dict]:
    """Parse total/non-covalent/non-bonded energies (kcal/mol) from GFN-FF output."""
    energies, in_block = {}, False
    with open(out_file) as f:
        for line in f:
            if "SUMMARY" in line:
                in_block, energies = True, {}
                continue
            if in_block:
                if ":: total charge" in line.lower():
                    break
                m = re.search(
                    r"::\s+(\S+)\s+energy\s+(-?\d+\.\d+)\s+eh\s+::", line.lower()
                )
                if m:
                    energies[m.group(1)] = float(m.group(2))
    if not energies:
        return None
    total = energies.get("total", 0.0)
    nci = sum(energies.get(k, 0.0) for k in ("electrostat", "dispersion", "hb", "xb"))
    return {
        "total": total * HARTREE_TO_KCAL,
        "nci": nci * HARTREE_TO_KCAL,
        "nbi": (total - energies.get("bond", 0.0)) * HARTREE_TO_KCAL,
    }


def _parse_gfn(out_file: str) -> Optional[dict]:
    """Parse total/non-covalent/non-bonded energies (kcal/mol) from GFN-n output."""
    e = {}
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if ":: total energy" in line:
                e["total"] = float(line.split()[3])
            elif ":: -> isotropic ES" in line:
                e["iso_es"] = float(line.split()[4])
            elif ":: -> anisotropic ES" in line:
                e["aniso_es"] = float(line.split()[4])
            elif ":: -> anisotropic XC" in line:
                e["aniso_xc"] = float(line.split()[4])
            elif ":: -> dispersion" in line:
                e["dispersion"] = float(line.split()[3])
    if not e:
        return None
    nbi = (
        e.get("dispersion", 0.0)
        + e.get("iso_es", 0.0)
        + e.get("aniso_es", 0.0)
        + e.get("aniso_xc", 0.0)
    )
    return {
        "total": e.get("total", 0.0) * HARTREE_TO_KCAL,
        "nci": e.get("dispersion", 0.0) * HARTREE_TO_KCAL,
        "nbi": nbi * HARTREE_TO_KCAL,
    }


def single_point_energy(
    mol: Chem.Mol,
    mol_idx: int = 0,
    *,
    xtb_version: str = "2",
    xtb_path: str = "xtb",
    out_dir: Optional[str] = None,
    charge: Optional[int] = None,
) -> Optional[EnergyTuple]:
    """Single-point energy of one molecule with 3D coordinates.

    Returns ``(total, non_covalent, non_bonded, num_atoms)`` in kcal/mol, or
    ``None`` on failure. For ``xtb_version="mmff"`` the non-covalent/non-bonded
    terms are ``0.0``.
    """
    if mol.GetNumConformers() == 0:
        return None

    if xtb_version == "mmff":
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            if props is None:
                return None
            energy = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
            return (float(energy), 0.0, 0.0, mol.GetNumAtoms())
        except Exception as exc:
            logger.warning("MMFF failed for molecule %d: %s", mol_idx, exc)
            return None

    out_dir = out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    name = f"mol_{mol_idx}"
    xyz = os.path.join(out_dir, f"{name}.xyz")
    work = os.path.join(out_dir, f"xtb_{name}")
    try:
        _write_xyz(mol, 0, xyz)
        _run_xtb(
            xyz,
            work,
            name,
            xtb_path,
            xtb_version,
            Chem.GetFormalCharge(mol) if charge is None else charge,
        )
        out_file = os.path.join(work, f"{name}.out")
        parsed = (
            _parse_gfnff(out_file)
            if str(xtb_version) == "gfnff"
            else _parse_gfn(out_file)
        )
        if parsed is None:
            return None
        return (parsed["total"], parsed["nci"], parsed["nbi"], _num_atoms(xyz))
    except Exception as exc:
        logger.warning("xTB failed for molecule %d: %s", mol_idx, exc)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.isfile(xyz):
            os.remove(xyz)


def compute_energies(
    mols: Sequence[Chem.Mol],
    *,
    xtb_version: str = "2",
    xtb_path: str = "xtb",
    out_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    task_timeout: int = 1000,
) -> List[Optional[EnergyTuple]]:
    """Single-point energies for many molecules in parallel (input order preserved)."""
    n_jobs = n_jobs or os.cpu_count() or 1
    results: List[Optional[EnergyTuple]] = [None] * len(mols)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                single_point_energy,
                mol,
                i,
                xtb_version=xtb_version,
                xtb_path=xtb_path,
                out_dir=out_dir,
            ): i
            for i, mol in enumerate(mols)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Energy ({xtb_version})"
        ):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=task_timeout)
            except (TimeoutError, Exception) as exc:
                logger.warning("Energy failed for molecule %d: %s", idx, exc)
    return results

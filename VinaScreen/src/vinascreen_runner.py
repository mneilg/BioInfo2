"""
VinaScreen High-Throughput Batch Docking Controller (Vina 1.2.x Compatible)
Removes deprecated --log CLI flag (which causes Vina 1.2.x parse errors)
and instead writes complete docking logs directly from stdout.
"""

import sys
import os
import re
import glob
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd

# Configure module logger
logger = logging.getLogger("vinascreen_runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def has_valid_coordinates(pdbqt_path: Union[str, Path]) -> bool:
    """
    Validates that PDBQT coordinates are neither empty nor collapsed to (0,0,0).
    """
    path = Path(pdbqt_path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    if len(line) >= 54:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                            return True
        return False
    except Exception:
        return False


def resolve_vina_binary(custom_path: Optional[str] = None) -> str:
    """
    Locates the AutoDock Vina binary across ./src/, project root, or system PATH.
    Grants execution permissions (+x) automatically.
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists() and p.is_file():
            try:
                p.chmod(p.stat().st_mode | 0o111)
            except Exception:
                pass
            return str(p)

    search_locations = [
        Path.cwd() / "src" / "vina_1.2.7_linux_x86_64",
        Path.cwd() / "src" / "vina_1.2.5_linux_x86_64",
        Path.cwd() / "src" / "vina",
        Path.cwd() / "vina_1.2.7_linux_x86_64",
        Path.cwd() / "vina_1.2.5_linux_x86_64",
        Path.cwd() / "vina",
        Path.cwd() / "bin" / "vina",
    ]

    for loc in search_locations:
        if loc.exists() and loc.is_file():
            try:
                loc.chmod(loc.stat().st_mode | 0o111)
            except Exception:
                pass
            return str(loc.resolve())

    for pattern in ["src/vina*", "./src/vina*", "vina*", "./vina*", "bin/vina*"]:
        for match in glob.glob(pattern):
            p = Path(match).resolve()
            if p.is_file() and not p.name.endswith((".py", ".ipynb", ".csv", ".txt", ".pdbqt", ".sdf", ".pqr")):
                try:
                    p.chmod(p.stat().st_mode | 0o111)
                    return str(p)
                except Exception:
                    return str(p)

    which_vina = shutil.which("vina")
    if which_vina:
        return which_vina

    raise FileNotFoundError("AutoDock Vina executable not found.")


def parse_vina_scores(pdbqt_file: Path) -> Dict[str, Any]:
    """
    Extracts top binding affinity (kcal/mol) and RMSD values from a docked Vina PDBQT file.
    """
    if not pdbqt_file.exists():
        return {"affinity": None, "rmsd_lb": None}

    try:
        with open(pdbqt_file, "r") as f:
            for line in f:
                if re.match(r"^\s*1\s+([-\d\.\+eE]+)\s+([-\d\.]+)", line):
                    parts = line.split()
                    return {
                        "affinity": float(parts[1]),
                        "rmsd_lb": float(parts[2])
                    }
                match = re.search(r"REMARK\s+VINA\s+RESULT:\s+([-\d\.\+eE]+)\s+([-\d\.]+)", line)
                if match:
                    return {
                        "affinity": float(match.group(1)),
                        "rmsd_lb": float(match.group(2))
                    }
    except Exception:
        pass

    return {"affinity": None, "rmsd_lb": None}


def run_single_docking(
    vina_bin: str,
    receptor_path: Path,
    ligand_path: Path,
    output_pose_path: Path,
    config: Dict[str, Any],
    log_file_path: Optional[Path] = None,
    timeout_seconds: int = 600
) -> Dict[str, Any]:
    """
    Executes a single AutoDock Vina docking calculation.
    Captures stdout directly into log_file_path to avoid the deprecated --log CLI error in Vina 1.2.x.
    """
    if not receptor_path.exists():
        return {"success": False, "affinity": None, "rmsd_lb": None, "error": f"Receptor not found: {receptor_path.name}"}
    if not ligand_path.exists():
        return {"success": False, "affinity": None, "rmsd_lb": None, "error": f"Ligand not found: {ligand_path.name}"}

    if not has_valid_coordinates(ligand_path):
        return {"success": False, "affinity": None, "rmsd_lb": None, "error": "Input ligand has invalid (0,0,0) coordinates."}

    output_pose_path.parent.mkdir(parents=True, exist_ok=True)
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Note: --log is removed because Vina 1.2.x deprecated it in favor of console redirection
    cmd = [
        vina_bin,
        "--receptor", str(receptor_path.resolve()),
        "--ligand", str(ligand_path.resolve()),
        "--out", str(output_pose_path.resolve()),
        "--center_x", str(config.get("center_x", 0.0)),
        "--center_y", str(config.get("center_y", 0.0)),
        "--center_z", str(config.get("center_z", 0.0)),
        "--size_x", str(config.get("size_x", 20.0)),
        "--size_y", str(config.get("size_y", 20.0)),
        "--size_z", str(config.get("size_z", 20.0)),
        "--exhaustiveness", str(config.get("exhaustiveness", 8)),
        "--num_modes", str(config.get("num_modes", 9)),
        "--energy_range", str(config.get("energy_range", 3)),
        "--cpu", str(config.get("cpu", 4))
    ]

    if "seed" in config and config["seed"] is not None:
        cmd.extend(["--seed", str(config["seed"])])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        combined_text = f"{proc.stdout}\n{proc.stderr}".strip()

        # Write the full console output to the .log file in Python
        if log_file_path:
            with open(log_file_path, "w") as f_log:
                f_log.write(combined_text)

        if proc.returncode == 0 and output_pose_path.exists() and output_pose_path.stat().st_size > 0:
            if not has_valid_coordinates(output_pose_path):
                output_pose_path.unlink(missing_ok=True)
                return {"success": False, "affinity": None, "rmsd_lb": None, "error": "Docked pose collapsed to (0,0,0) coordinates."}

            scores = parse_vina_scores(output_pose_path)
            return {
                "success": True,
                "affinity": scores["affinity"],
                "rmsd_lb": scores["rmsd_lb"],
                "error": ""
            }
        else:
            error_lines = [
                line.strip() for line in combined_text.splitlines()
                if any(kw in line.lower() for kw in ["error", "fatal", "cannot", "failed", "exception", "unrecognised option"])
            ]
            final_err = "; ".join(error_lines) if error_lines else (combined_text.splitlines()[-1] if combined_text else "Vina non-zero exit")
            return {
                "success": False,
                "affinity": None,
                "rmsd_lb": None,
                "error": final_err,
                "raw_output": combined_text
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "affinity": None, "rmsd_lb": None, "error": f"Docking timed out after {timeout_seconds}s"}
    except Exception as exc:
        return {"success": False, "affinity": None, "rmsd_lb": None, "error": f"Unexpected execution error: {exc}"}


import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

def _docking_worker_task(args):
    (
        vina_bin,
        rec_path,
        lig_path,
        out_dir,
        docking_config,
        timeout_per_job,
        job_id
    ) = args

    rec_name = rec_path.stem
    lig_name = lig_path.stem
    output_pose = out_dir / f"{rec_name}_{lig_name}_docked.pdbqt"
    log_file = out_dir / f"{rec_name}_{lig_name}.log"

    # Use the cpu value defined in docking_config (e.g., 4)
    per_job_config = dict(docking_config)

    dock_res = run_single_docking(
        vina_bin=vina_bin,
        receptor_path=rec_path,
        ligand_path=lig_path,
        output_pose_path=output_pose,
        config=per_job_config,
        log_file_path=log_file,
        timeout_seconds=timeout_per_job
    )

    return {
        "job_id": job_id,
        "receptor": rec_name,
        "ligand": lig_name,
        "affinity_kcal_mol": dock_res["affinity"],
        "rmsd_lb": dock_res["rmsd_lb"],
        "status": "Success" if dock_res["success"] else "Failed",
        "docked_pose_file": str(output_pose) if dock_res["success"] else "",
        "error_message": dock_res["error"]
    }


def run_vinascreen_batch(
    receptor_dir: Union[str, Path],
    ligand_dir: Union[str, Path],
    output_dir: Union[str, Path],
    docking_config: Dict[str, Any],
    results_csv: Optional[Union[str, Path]] = None,
    vina_executable: Optional[str] = None,
    timeout_per_job: int = 600,
    max_workers: int = 22  # Uses 22 of your 24 CPU cores
) -> pd.DataFrame:
    """
    High-throughput concurrent batch docking coordinator.
    Distributes jobs across a 22-process pool with 1 thread per Vina task.
    """
    rec_dir = Path(receptor_dir).resolve()
    lig_dir = Path(ligand_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rec_dir.exists():
        raise FileNotFoundError(f"Receptor directory not found: {rec_dir}")
    if not lig_dir.exists():
        raise FileNotFoundError(f"Ligand directory not found: {lig_dir}")

    vina_bin = resolve_vina_binary(vina_executable)
    logger.info(f"Using AutoDock Vina binary: '{vina_bin}'")

    receptor_files = sorted(list(rec_dir.glob("*.pdbqt")))
    ligand_files = sorted(list(lig_dir.glob("*.pdbqt")))

    if not receptor_files:
        raise ValueError(f"No receptor .pdbqt files found in '{rec_dir}'.")
    if not ligand_files:
        raise ValueError(f"No ligand .pdbqt files found in '{lig_dir}'.")

    # Build work queue
    tasks = []
    job_idx = 0
    for rec_path in receptor_files:
        for lig_path in ligand_files:
            job_idx += 1
            tasks.append((
                vina_bin,
                rec_path,
                lig_path,
                out_dir,
                docking_config,
                timeout_per_job,
                job_idx
            ))

    total_jobs = len(tasks)
    logger.info(
        f"Initialized VinaScreen: {len(receptor_files)} receptor(s) x {len(ligand_files)} ligand(s) "
        f"= {total_jobs} total docking jobs across {max_workers} parallel workers."
    )

    results = []
    completed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_docking_worker_task, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            res = future.result()
            results.append(res)

            if completed_count % 25 == 0 or completed_count == total_jobs:
                logger.info(f"Progress: [{completed_count}/{total_jobs}] jobs completed.")

    # Sort results back by job_id for clean presentation
    results = sorted(results, key=lambda x: x["job_id"])
    df_results = pd.DataFrame(results)

    if results_csv:
        csv_path = Path(results_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Results table successfully written to '{csv_path}'.")

    logger.info("=" * 45)
    logger.info("VINASCREEN DOCKING COMPLETED")
    logger.info("=" * 45)
    logger.info(f"Total jobs executed : {total_jobs}")
    logger.info(f"Successful runs     : {(df_results['status'] == 'Success').sum()}")
    logger.info(f"Failed runs         : {(df_results['status'] == 'Failed').sum()}")
    logger.info(f"Output directory    : {out_dir}")
    logger.info("=" * 45)

    return df_results

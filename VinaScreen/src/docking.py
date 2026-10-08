#!/usr/bin/env python3
"""
docking.py - Run AutoDock Vina
*BUGFIX*: Validates input and output PDBQT files to prevent 0,0,0 coordinate hits.
"""

import os
import subprocess
import re
import sys
from pathlib import Path

def has_valid_coordinates(pdbqt_path):
    """Safety check: Returns False if coordinates are completely flattened."""
    try:
        with open(pdbqt_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                        return True
        return False
    except Exception:
        return False

def generate_output_filename(receptor, ligand, output_dir="output"):
    rec_name = Path(receptor).stem
    lig_name = Path(ligand).stem
    base = f"{rec_name}_{lig_name}"
    filename = f"{base}.pdbqt"
    fullpath = os.path.join(output_dir, filename)
    i = 1
    while os.path.exists(fullpath):
        filename = f"{base}_{i}.pdbqt"
        fullpath = os.path.join(output_dir, filename)
        i += 1
    return fullpath

def run_vina(receptor, ligand, config="config.txt", vina_exec="vina_1.2.7_linux_x86_64", output_dir="output"):
    if not os.path.exists(vina_exec):
        return {"success": False, "error": "Vina executable not found"}
    if not os.path.exists(receptor) or not os.path.exists(ligand):
        return {"success": False, "error": "Receptor or Ligand file not found"}
        
    # PRE-DOCKING SAFETY CHECK
    if not has_valid_coordinates(ligand):
        return {"success": False, "error": "Input ligand has invalid (0,0,0) coordinates. Skipped."}

    os.makedirs(output_dir, exist_ok=True)
    output_file = generate_output_filename(receptor, ligand, output_dir)
    
    cmd = [f"./{vina_exec}", "--receptor", receptor, "--ligand", ligand, "--config", config, "--out", output_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"success": False, "error": "Vina crashed or returned non-zero exit code."}

        # POST-DOCKING SAFETY CHECK
        if not os.path.exists(output_file):
            return {"success": False, "error": "Vina did not generate an output file."}
        if not has_valid_coordinates(output_file):
            os.remove(output_file) # Clean up the garbage file
            return {"success": False, "error": "Vina docked pose collapsed to (0,0,0) coordinates. Discarded."}

        affinity, rmsd_lb = None, None
        for line in (result.stdout + result.stderr).split('\n'):
            if re.match(r'^\s*1\s+', line):
                parts = line.split()
                if len(parts) >= 4:
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    
        if affinity is None:
            return {"success": False, "error": "Could not parse affinity from Vina output"}
            
        return {"success": True, "output_file": output_file, "affinity": affinity, "rmsd_lb": rmsd_lb}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Vina execution timed out"}
    except Exception as exc:
        return {"success": False, "error": f"Unexpected error: {exc}"}

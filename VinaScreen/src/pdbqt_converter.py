#!/usr/bin/env python3
"""
Fixed PDBQT Converter
Converts 3D molecular libraries (SDF/MOL) to PDBQT format.
*BUGFIX*: Validates 3D coordinates to prevent silent 0,0,0 coordinate failures.
"""

import subprocess
import sys
import os
import tempfile
import argparse
from pathlib import Path
import time

TIMEOUT_SECS = 20

def has_valid_coordinates(pdbqt_path):
    """Returns False if all ATOM/HETATM coordinates are essentially zero."""
    try:
        with open(pdbqt_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    # PDBQT strict column formatting for x, y, z
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    # If any coordinate is meaningfully non-zero, it's valid
                    if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.001:
                        return True
        return False
    except Exception:
        return False

class PDBQTConverter:
    def __init__(self, timeout=TIMEOUT_SECS):
        self.timeout = timeout
        self.stats = {'total': 0, 'converted': 0, 'skipped': 0, 'errors': 0}

    def read_sdf_file(self, filepath):
        molecules = []
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            mol_blocks = content.split('$$$$')
            for i, mol_block in enumerate(mol_blocks):
                mol_block = mol_block.strip()
                if not mol_block: continue
                lines = mol_block.split('\n')
                mol_id = lines[0].strip() if lines[0].strip() else f"mol_{i+1:04d}"
                molecules.append({'mol_id': mol_id, 'mol_block': mol_block})
        except Exception as e:
            print(f"Error reading SDF file: {e}")
        return molecules

    def convert_molecule_to_pdbqt(self, molecule, output_path, tmpdir):
        mol_id = molecule['mol_id']
        mol_block = molecule['mol_block']
        
        temp_input = os.path.join(tmpdir, f"{mol_id}_input.mol")
        temp_pdbqt = os.path.join(tmpdir, f"{mol_id}_temp.pdbqt")

        try:
            with open(temp_input, 'w') as f:
                f.write(mol_block)
                
            # Run OpenBabel (Notice: NO --gen3d flag here anymore!)
            cmd = ["obabel", temp_input, "-O", temp_pdbqt, "-h"]
            result = subprocess.run(cmd, timeout=self.timeout, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_pdbqt):
                # THE CRITICAL FIX: Validate coordinates before accepting
                if has_valid_coordinates(temp_pdbqt):
                    with open(temp_pdbqt, 'r') as src, open(output_path, 'w') as dst:
                        dst.write(src.read())
                    return True
                else:
                    print(f"[{mol_id}] Error: OpenBabel flattened the molecule to 0,0,0 coordinates.")
                    return False
        except subprocess.TimeoutExpired:
            print(f"[{mol_id}] Timeout.")
        except Exception as e:
            print(f"[{mol_id}] Error: {e}")
        return False

    def convert_library(self, input_file, output_dir="ligand"):
        molecules = self.read_sdf_file(input_file)
        self.stats['total'] = len(molecules)
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, molecule in enumerate(molecules, 1):
                output_path = os.path.join(output_dir, f"{molecule['mol_id']}.pdbqt")
                if self.convert_molecule_to_pdbqt(molecule, output_path, tmpdir):
                    self.stats['converted'] += 1
                else:
                    self.stats['skipped'] += 1
                print(f"\rProcessed: {i}/{self.stats['total']} (Success: {self.stats['converted']}, Skipped: {self.stats['skipped']})", end='')

        print("\n\n=== CONVERSION SUMMARY ===")
        print(f"Total processed: {self.stats['total']}")
        print(f"Successfully converted (Valid 3D): {self.stats['converted']}")
        print(f"Skipped (Errors/Flattened): {self.stats['skipped']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3D SDF to PDBQT")
    parser.add_argument("input", help="Input .sdf file")
    parser.add_argument("-o", "--output", default="ligand", help="Output directory")
    args = parser.parse_args()
    
    PDBQTConverter().convert_library(args.input, args.output)

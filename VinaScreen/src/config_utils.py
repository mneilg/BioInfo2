#!/usr/bin/env python3
"""
config_utils.py - Directory and Configuration Checker for VinaScreen
Checks for required folders, files, and Vina executable before running screening.
"""

import os
import sys

def check_vina_executable():
    vina_exe = "vina_1.2.7_linux_x86_64"
    if not os.path.exists(vina_exe):
        print("vina_1.2.7_linux_x86_64 executable is not found in the current directory")
        return False
    if not os.access(vina_exe, os.X_OK):
        print("vina_1.2.7_linux_x86_64 is not executable. Please check permissions.")
        return False
    print(f"✓ Found Vina executable: {vina_exe}")
    return True

def check_ligand_folder():
    ligand_dir = "ligand"
    if not os.path.exists(ligand_dir):
        print("ligand folder is not found in the current directory")
        return False
    ligand_files = [f for f in os.listdir(ligand_dir) if f.endswith(('.pdbqt', '.mol2', '.sdf'))]
    if not ligand_files:
        print(f"WARNING: No ligand files found in {ligand_dir} folder")
        return False
    print(f"✓ Found ligand folder with {len(ligand_files)} ligand files")
    return True

def check_receptor_folder():
    receptor_dir = "receptor"
    if not os.path.exists(receptor_dir):
        print("receptor folder is not found in the current directory")
        return False
    receptor_files = [f for f in os.listdir(receptor_dir) if f.endswith('.pdbqt')]
    if not receptor_files:
        print(f"WARNING: No receptor files (.pdbqt) found in {receptor_dir} folder")
        return False
    print(f"✓ Found receptor folder with {len(receptor_files)} receptor files")
    return True

def check_config_file():
    config_file = "config.txt"
    if not os.path.exists(config_file):
        print("config.txt file is not found in the current directory")
        return False
    try:
        with open(config_file, 'r') as f:
            content = f.read().strip()
            if not content:
                print("ERROR: config.txt file is empty")
                return False
    except Exception as e:
        print(f"ERROR: Cannot read config.txt file: {e}")
        return False
    print(f"✓ Found and validated config.txt file")
    return True

def check_output_folder():
    output_dir = "output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"✓ Created output folder: {output_dir}")
        except Exception as e:
            print(f"ERROR: Cannot create output folder: {e}")
            return False
    else:
        print(f"✓ Found output folder: {output_dir}")
    return True

def validate_environment():
    print("Validating VinaScreen environment...")
    print("=" * 50)
    checks = [
        check_vina_executable(),
        check_ligand_folder(),
        check_receptor_folder(),
        check_config_file(),
        check_output_folder()
    ]
    if all(checks):
        print("=" * 50)
        print("✓ Environment validation successful!")
        return True
    else:
        print("=" * 50)
        print("✗ Environment validation failed!")
        print("Please fix the errors above before proceeding.")
        return False

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)


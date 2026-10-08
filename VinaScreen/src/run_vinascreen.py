#!/usr/bin/env python3
"""
main.py - Main Controller Script for VinaScreen
Orchestrates config check, batch docking, progress tracking, and results output.
"""

import sys
import os
from config_utils import validate_environment
from docking import run_vina
from reporter import CsvProgressReporter

def get_ligand_files(folder="ligand"):
    return sorted([os.path.join(folder, f)
                   for f in os.listdir(folder) if f.endswith('.pdbqt')])

def get_receptor_files(folder="receptor"):
    return sorted([os.path.join(folder, f)
                   for f in os.listdir(folder) if f.endswith('.pdbqt')])

def main():
    print("="*60)
    print("VinaScreen High-Throughput Docking Pipeline (main.py)")
    print("="*60)

    # Step 1: Validate environment
    print("\nStep 1: Validating environment setup...")
    if not validate_environment():
        print("Environment validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Acquire ligand and receptor lists
    print("\nStep 2: Loading inputs...")
    ligand_files = get_ligand_files()
    receptor_files = get_receptor_files()
    if not ligand_files or not receptor_files:
        print("ERROR: Missing ligands or receptors for docking. Aborting.")
        sys.exit(1)

    print(f"  {len(ligand_files)} ligands loaded")
    print(f"  {len(receptor_files)} receptors loaded")

    # Step 3: Prepare reporter
    print("\nStep 3: Initializing CSV reporter and progress tracker...")
    total_jobs = len(ligand_files) * len(receptor_files)
    results_csv = "vinascreen_results.csv"
    reporter = CsvProgressReporter(csv_file=results_csv, total_jobs=total_jobs)
    print(f"  Output CSV: {results_csv}")
    print(f"  Total docking jobs: {total_jobs}")

    # Step 4: Docking loop
    print("\nStep 4: Launching docking runs...\n")
    count = 0
    for receptor_path in receptor_files:
        receptor_name = os.path.basename(receptor_path)
        for ligand_path in ligand_files:
            ligand_name = os.path.basename(ligand_path)
            # Run docking
            result = run_vina(receptor_path, ligand_path)
            success = result.get("success", False)
            affinity = result.get("affinity")
            rmsd_lb = result.get("rmsd_lb")
            output_file = result.get("output_file")
            error = result.get("error")
            # Update reporter for every job
            reporter.add_result(
                receptor=receptor_name,
                ligand=ligand_name,
                affinity=affinity,
                rmsd_lb=rmsd_lb,
                success=success,
                output_file=output_file,
                error_msg=error
            )
            count += 1

    print("\nStep 5: Summary")
    print("="*60)
    print(f"Docking complete. {count} jobs processed.")
    print(f"Results saved to: {results_csv}")
    print("="*60)

if __name__ == "__main__":
    main()


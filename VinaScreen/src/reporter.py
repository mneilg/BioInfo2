#!/usr/bin/env python3
"""
reporter.py – Combined CSV Result Reporter and Progress Tracker
"""

import csv
import os
from datetime import datetime

class CsvProgressReporter:
    def __init__(self, csv_file, total_jobs, log_file='errors.log'):
        self.csv_file = csv_file
        self.log_file = log_file
        self.total_jobs = total_jobs
        self.done = 0
        self.kept = 0
        self.skipped = 0
        self._write_header()
        open(self.log_file, 'w').close()

    def _write_header(self):
        write_header = not os.path.exists(self.csv_file)
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["receptor", "ligand", "affinity", "rmsd_lb", "status", "output_file", "timestamp"])

    def add_result(self, receptor, ligand, affinity, rmsd_lb, success, output_file=None, error_msg=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.done += 1
        
        if success:
            self.kept += 1
            status = "OK"
        else:
            self.skipped += 1
            status = "FAILED"
            # Explicitly alert the user if the new coordinate catch was triggered
            if "0,0,0" in str(error_msg) or "invalid" in str(error_msg).lower():
                print(f"\n[WARNING] {ligand} failed coordinate validation: {error_msg}")
            
            with open(self.log_file, "a") as logf:
                logf.write(f"[{now}] {receptor}+{ligand}: {error_msg}\n")
                
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                receptor, ligand, 
                affinity if affinity is not None else "",
                rmsd_lb if rmsd_lb is not None else "",
                status, output_file if output_file else "", now
            ])
        self.report_progress()

    def report_progress(self):
        print(f"Processed: {self.done}/{self.total_jobs} (kept: {self.kept}, skipped: {self.skipped})",
              end='\r' if self.done != self.total_jobs else '\n', flush=True)

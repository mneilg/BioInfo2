"""
Receptor PDBQT Batch Preparation Module (Meeko-based with Resilient Fallbacks)
Converts cleaned PDB structures into AutoDock PDBQT format programmatically without notebook shell magic.
Includes automated fallback handling for Meeko polymer padding errors (disulfide bridges, gaps, and altloc clashes).

Two-tier protonation model:
  Tier 1 - Global pH titration via PDB2PQR/PROPKA, applied to all standard titratable residues.
  Tier 2 - User-specified fixed protonation states for individual residues (e.g. catalytic /
           proton-relay residues assigned from crystallographic H-bond geometry), which override
           the Tier 1 titration for those residues only.
"""

import sys
import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import pandas as pd

# Configure module logger
logger = logging.getLogger("receptor_pdbqt")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Fixed (non-titratable) AMBER-family protonation tags recognized by PDB2PQR.
# Renaming a residue to one of these BEFORE running PDB2PQR/PROPKA locks its
# charge state so the automated titration at the global pH does not touch it.
# ---------------------------------------------------------------------------
ALLOWED_PROTONATION_TAGS = (
    "ASH",  # neutral Aspartate (protonated carboxylic acid)
    "GLH",  # neutral Glutamate (protonated carboxylic acid)
    "HID",  # neutral Histidine, proton on delta nitrogen
    "HIE",  # neutral Histidine, proton on epsilon nitrogen
    "HIP",  # doubly protonated (positive) Histidine
    "LYN",  # neutral Lysine (deprotonated amine)
    "CYM",  # deprotonated Cysteine (thiolate)
    "TYM",  # deprotonated Tyrosine (phenolate)
)

# Residues these tags map back to, used only for logging/validation messages.
_PARENT_RESIDUE = {
    "ASH": "ASP", "GLH": "GLU",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "LYN": "LYS", "CYM": "CYS", "TYM": "TYR",
}

RESIDUE_OVERRIDE_KEY = Tuple[str, int]  # (chain_id, res_seq_number)


def validate_residue_overrides(
    residue_overrides: Optional[Dict[RESIDUE_OVERRIDE_KEY, str]]
) -> Dict[RESIDUE_OVERRIDE_KEY, str]:
    """
    Validates a user-supplied {(chain, resnum): tag} override dict against
    ALLOWED_PROTONATION_TAGS. Raises ValueError on any unrecognized tag so
    a typo doesn't silently fall through to PDB2PQR and get ignored.
    """
    if not residue_overrides:
        return {}

    cleaned: Dict[RESIDUE_OVERRIDE_KEY, str] = {}
    for key, tag in residue_overrides.items():
        chain_id, res_seq = key
        tag_upper = str(tag).strip().upper()
        if tag_upper not in ALLOWED_PROTONATION_TAGS:
            raise ValueError(
                f"Invalid protonation tag '{tag}' for residue {chain_id}{res_seq}. "
                f"Allowed tags: {ALLOWED_PROTONATION_TAGS}"
            )
        cleaned[(str(chain_id).strip(), int(res_seq))] = tag_upper
        logger.info(
            f"Override: chain {chain_id}, residue {res_seq} "
            f"({_PARENT_RESIDUE[tag_upper]} -> {tag_upper}) fixed, exempt from pH titration."
        )
    return cleaned


def apply_residue_overrides(
    input_pdb: Union[str, Path],
    output_pdb: Union[str, Path],
    residue_overrides: Dict[RESIDUE_OVERRIDE_KEY, str],
) -> Dict[str, Any]:
    """
    Rewrites the residue name field on matching ATOM/HETATM lines so PDB2PQR
    treats these residues as fixed-state rather than titratable. Must be run
    BEFORE the PDB2PQR protonation step (Tier 2 applied ahead of Tier 1 titration).
    """
    in_path = Path(input_pdb)
    out_path = Path(output_pdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    applied_counts: Dict[str, int] = {}
    not_found = set(residue_overrides.keys())

    with open(in_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                chain_id = line[21].strip()
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    f_out.write(line)
                    continue

                key = (chain_id, res_seq)
                if key in residue_overrides:
                    new_resname = residue_overrides[key]
                    line = line[:17] + f"{new_resname:>3}" + line[20:]
                    applied_counts[key] = applied_counts.get(key, 0) + 1
                    not_found.discard(key)

            f_out.write(line)

    if not_found:
        logger.warning(
            f"{len(not_found)} residue override(s) not found in '{in_path.name}': {sorted(not_found)}"
        )

    return {
        "overrides_requested": len(residue_overrides),
        "overrides_applied": len(applied_counts),
        "overrides_missing": sorted(not_found),
    }


def protonate_receptor(
    input_pdb: Union[str, Path],
    output_pdb: Union[str, Path],
    global_ph: float = 7.4,
    ff: str = "AMBER",
    residue_overrides: Optional[Dict[RESIDUE_OVERRIDE_KEY, str]] = None,
) -> Dict[str, Any]:
    """
    Mandatory Tier 1 + Tier 2 protonation step, run ahead of Meeko conversion.

    Tier 1: pdb2pqr30 with PROPKA titration at the specified global pH, applied
            to all standard titratable residues (Asp/Glu/His/Lys/Cys/Tyr).
    Tier 2: any residues pre-renamed via apply_residue_overrides() are excluded
            from Tier 1 titration and keep their user-assigned fixed state.
    """
    in_path = Path(input_pdb).resolve()
    out_path = Path(output_pdb).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_info: Dict[str, Any] = {
        "filename": in_path.name,
        "global_ph": global_ph,
        "status": "Success",
        "output_file": str(out_path),
        "override_report": {},
        "error_message": "",
    }

    validated_overrides = validate_residue_overrides(residue_overrides)

    # Tier 2 first: lock fixed residues before PDB2PQR ever sees the file.
    if validated_overrides:
        overridden_pdb = out_path.parent / f"_overridden_{in_path.stem}.pdb"
        override_report = apply_residue_overrides(in_path, overridden_pdb, validated_overrides)
        result_info["override_report"] = override_report
        pdb2pqr_input = overridden_pdb
    else:
        pdb2pqr_input = in_path

    # Tier 1: global pH titration for everything not already fixed.
    pqr_temp = out_path.parent / f"_temp_{in_path.stem}.pqr"
    cmd = [
        "pdb2pqr30",
        f"--ff={ff}",
        f"--with-ph={global_ph}",
        "--titration-state-method=propka",
        "--pdb-output", str(out_path),
        str(pdb2pqr_input),
        str(pqr_temp),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        result_info["status"] = "Failed"
        err = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
        result_info["error_message"] = err.splitlines()[-1] if err else "pdb2pqr30 execution failed"
        logger.error(f"Protonation failed for '{in_path.name}': {result_info['error_message']}")
    else:
        logger.info(
            f"Protonated '{in_path.name}' at pH {global_ph} "
            f"({len(validated_overrides)} fixed residue(s)) -> '{out_path.name}'"
        )

    pqr_temp.unlink(missing_ok=True)
    if validated_overrides:
        Path(pdb2pqr_input).unlink(missing_ok=True)

    return result_info


def convert_single_receptor_to_pdbqt(
    pdb_path: Union[str, Path],
    output_pdbqt_path: Union[str, Path],
    has_cofactor: bool = False,
    allow_bad_res: bool = True,
    extra_args: Optional[List[str]] = None,
    apply_protonation: bool = True,
    global_ph: float = 7.4,
    residue_overrides: Optional[Dict[RESIDUE_OVERRIDE_KEY, str]] = None,
) -> Dict[str, Any]:
    """
    Executes the protonation + Meeko receptor preparation pipeline:
    - Tier 1/2 protonation via pdb2pqr30 (skipped only if apply_protonation=False)
    - Primary attempt: Standard Meeko CLI with --default_altloc A
    - Fallback 1: Retries with --delete_bad_res if polymer padding/crosslink fails
    - Fallback 2: Uses PDB2PQR pre-titration (if not already applied) to bypass polymer padding

    Returned dict is flat (no nested sub-dicts) so it can be dropped straight
    into a pandas DataFrame and indexed by column name in notebook display calls:
      filename, mode, status, output_file, error_message,
      global_ph, protonation_status,
      custom_protonation_used, residues_overridden, residues_missing_override
    """
    in_pdb = Path(pdb_path).resolve()
    out_pdbqt = Path(output_pdbqt_path).resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    out_stem = str(out_pdbqt.with_suffix(""))
    actual_output = out_pdbqt if out_pdbqt.suffix == ".pdbqt" else out_pdbqt.with_suffix(".pdbqt")

    custom_protonation_used = bool(residue_overrides)

    result_info: Dict[str, Any] = {
        "filename": in_pdb.name,
        "mode": "Protein+Cofactor" if has_cofactor else "Protein-Only",
        "status": "Success",
        "output_file": str(actual_output),
        "global_ph": global_ph if apply_protonation else None,
        "protonation_status": "Skipped",
        "custom_protonation_used": custom_protonation_used,
        "residues_overridden": 0,
        "residues_missing_override": [],
        "error_message": "",
    }

    conversion_input = in_pdb
    already_protonated = False

    # ---------------------------------------------------------------
    # Step 0: Mandatory (default-on) protonation step
    # ---------------------------------------------------------------
    if apply_protonation:
        protonated_pdb = out_pdbqt.parent / f"_protonated_{in_pdb.stem}.pdb"
        protonation_result = protonate_receptor(
            input_pdb=in_pdb,
            output_pdb=protonated_pdb,
            global_ph=global_ph,
            residue_overrides=residue_overrides,
        )
        result_info["protonation_status"] = protonation_result["status"]

        override_report = protonation_result.get("override_report", {})
        result_info["residues_overridden"] = override_report.get("overrides_applied", 0)
        result_info["residues_missing_override"] = override_report.get("overrides_missing", [])

        if protonation_result["status"] == "Success":
            conversion_input = protonated_pdb
            already_protonated = True
        else:
            result_info["error_message"] = protonation_result["error_message"]
            logger.warning(
                f"Proceeding to Meeko with un-protonated input for '{in_pdb.name}' "
                f"(protonation step failed: {protonation_result['error_message']})"
            )

    # Base flags
    base_flags = [
        "-i", str(conversion_input),
        "-o", out_stem,
        "-p",
        "--default_altloc", "A",  # Prevents atom duplication on altlocs
    ]
    if allow_bad_res:
        base_flags.append("-a")
    if extra_args:
        base_flags.extend(extra_args)

    # -------------------------------------------------------------
    # Attempt 1: Standard Meeko preparation
    # -------------------------------------------------------------
    cmd1 = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor"] + base_flags
    proc1 = subprocess.run(cmd1, capture_output=True, text=True, check=False)

    if proc1.returncode == 0 and actual_output.exists() and actual_output.stat().st_size > 0:
        logger.info(f"Converted '{in_pdb.name}' -> '{actual_output.name}'")
        if already_protonated:
            Path(conversion_input).unlink(missing_ok=True)
        return result_info

    # -------------------------------------------------------------
    # Attempt 2: Fallback with --delete_bad_res (handles abnormal gap/crosslink residues)
    # -------------------------------------------------------------
    logger.warning(f"Standard Meeko failed on '{in_pdb.name}'. Retrying with residue relaxation (--delete_bad_res)...")
    cmd2 = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor"] + base_flags + ["--delete_bad_res"]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)

    if proc2.returncode == 0 and actual_output.exists() and actual_output.stat().st_size > 0:
        logger.info(f"Converted '{in_pdb.name}' -> '{actual_output.name}' via fallback 1 (--delete_bad_res).")
        result_info["error_message"] = "Recovered via --delete_bad_res"
        if already_protonated:
            Path(conversion_input).unlink(missing_ok=True)
        return result_info

    # -------------------------------------------------------------
    # Attempt 3: Fallback via PDB2PQR (bypasses Meeko polymer padding entirely).
    # Skipped if Step 0 already produced a protonated structure at the target pH.
    # -------------------------------------------------------------
    if not already_protonated:
        pqr_temp = out_pdbqt.parent / f"_temp_{in_pdb.stem}.pqr"
        try:
            pqr_cmd = ["pdb2pqr30", "--ff=AMBER", "--keep-chain", str(in_pdb), str(pqr_temp)]
            pqr_proc = subprocess.run(pqr_cmd, capture_output=True, text=True, check=False)

            if pqr_proc.returncode == 0 and pqr_temp.exists():
                logger.info(f"Generated PQR for '{in_pdb.name}'. Converting via Meeko PQR reader...")
                cmd3 = [
                    sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
                    "--read_pqr", str(pqr_temp),
                    "--charge_model", "read",
                    "-o", out_stem,
                    "-p",
                    "-a",
                ]
                proc3 = subprocess.run(cmd3, capture_output=True, text=True, check=False)
                if proc3.returncode == 0 and actual_output.exists() and actual_output.stat().st_size > 0:
                    logger.info(f"Converted '{in_pdb.name}' -> '{actual_output.name}' via PDB2PQR fallback.")
                    result_info["error_message"] = "Recovered via PDB2PQR"
                    pqr_temp.unlink(missing_ok=True)
                    return result_info
        except Exception:
            pass
        finally:
            pqr_temp.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # If all attempts fail, capture the exact traceback
    # -------------------------------------------------------------
    if already_protonated:
        Path(conversion_input).unlink(missing_ok=True)

    result_info["status"] = "Failed"
    err = proc1.stderr.strip() or proc1.stdout.strip() or "Unknown error"
    result_info["error_message"] = err.splitlines()[-1] if err else "Execution failed"
    logger.error(f"Failed converting '{in_pdb.name}': {result_info['error_message']}")

    return result_info


def batch_convert_protein_only(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    audit_report_path: Optional[Union[str, Path]] = None,
    apply_protonation: bool = True,
    global_ph: float = 7.4,
    residue_overrides: Optional[Dict[RESIDUE_OVERRIDE_KEY, str]] = None,
) -> pd.DataFrame:
    """
    Batch conversion for standard protein receptors without cofactors.
    residue_overrides applies the SAME fixed protonation states to every
    structure in the batch -- appropriate for an aligned ensemble of one target.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(list(in_dir.glob("*.pdb")))
    logger.info(f"[Protein-Only Mode] Found {len(pdb_files)} structure(s) to convert.")

    records = []
    for pdb in pdb_files:
        base_name = pdb.stem.replace("clean_", "")
        out_file = out_dir / f"{base_name}.pdbqt"

        res = convert_single_receptor_to_pdbqt(
            pdb_path=pdb,
            output_pdbqt_path=out_file,
            has_cofactor=False,
            allow_bad_res=True,
            apply_protonation=apply_protonation,
            global_ph=global_ph,
            residue_overrides=residue_overrides,
        )
        records.append(res)

    df = pd.DataFrame(records)
    if audit_report_path:
        Path(audit_report_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(audit_report_path, index=False)
        logger.info(f"Report saved to '{audit_report_path}'.")

    return df


def batch_convert_with_cofactors(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    audit_report_path: Optional[Union[str, Path]] = None,
    custom_template_path: Optional[str] = None,
    apply_protonation: bool = True,
    global_ph: float = 7.4,
    residue_overrides: Optional[Dict[RESIDUE_OVERRIDE_KEY, str]] = None,
) -> pd.DataFrame:
    """
    Batch conversion for receptors with prosthetic groups/cofactors (Heme, Zinc, FAD, etc.).
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(list(in_dir.glob("*.pdb")))
    logger.info(f"[Cofactor Mode] Found {len(pdb_files)} structure(s) to convert.")

    extra_args = []
    if custom_template_path:
        extra_args.extend(["--add_templates", custom_template_path])

    records = []
    for pdb in pdb_files:
        base_name = pdb.stem.replace("clean_", "")
        out_file = out_dir / f"{base_name}_cofactor.pdbqt"

        res = convert_single_receptor_to_pdbqt(
            pdb_path=pdb,
            output_pdbqt_path=out_file,
            has_cofactor=True,
            allow_bad_res=True,
            extra_args=extra_args,
            apply_protonation=apply_protonation,
            global_ph=global_ph,
            residue_overrides=residue_overrides,
        )
        records.append(res)

    df = pd.DataFrame(records)
    if audit_report_path:
        Path(audit_report_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(audit_report_path, index=False)
        logger.info(f"Report saved to '{audit_report_path}'.")

    return df

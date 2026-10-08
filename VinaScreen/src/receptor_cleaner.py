"""
Receptor Cleaner Module (Alignment-Free)
Supports local PDB files as ensemble members.
Robust multi-chain support (e.g., ['A', 'B', 'C']).
Fixes side chains and missing heavy atoms while suppressing artificial linear terminal loops.
Does NOT perform any structural alignment/superposition — ensemble members retain their
original coordinate frame so pre-existing ensemble alignment (e.g., aligned on specific
chains/residues for docking) is preserved.
"""

import sys
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Set, Tuple
import pandas as pd
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Configure module logger
logger = logging.getLogger("receptor_cleaner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_chains(chains_input: Optional[Union[str, List[str], Set[str], Tuple[str, ...]]]) -> Optional[List[str]]:
    """
    Robustly parses chain specifications regardless of input formatting:
    'A, B, C' -> ['A', 'B', 'C']
    ['A, B, C'] -> ['A', 'B', 'C']
    ['A', 'B', 'C']-> ['A', 'B', 'C']
    'A B C' -> ['A', 'B', 'C']
    None -> None (retains all chains)
    """
    if chains_input is None:
        return None

    raw_items = []
    if isinstance(chains_input, str):
        raw_items = [chains_input]
    elif isinstance(chains_input, (list, tuple, set)):
        raw_items = list(chains_input)
    else:
        raw_items = [str(chains_input)]

    parsed = []
    for item in raw_items:
        if not item:
            continue
        item_str = str(item).strip()
        tokens = re.split(r"[,;\s/]+", item_str)
        for tok in tokens:
            cleaned_tok = tok.strip().upper()
            if cleaned_tok:
                parsed.append(cleaned_tok)

    deduped = list(dict.fromkeys(parsed))
    return deduped if deduped else None


def clean_pdb(
    input_source: Union[str, Path],
    output_path: Path,
    keep_chains: Optional[Union[str, List[str]]] = None,
    keep_water: bool = False,
    replace_nonstandard: bool = True,
    add_missing_atoms: bool = True,
    build_missing_loops: bool = False
) -> Dict[str, Any]:
    """
    Clean and repair a PDB structure using PDBFixer while tracking modifications.
    Writes the cleaned structure directly to output_path with its original coordinate
    frame untouched (no alignment/superposition is applied).
    """
    input_path = Path(input_source)
    target_chains = parse_chains(keep_chains)

    log_info = {
        "filename": input_path.name,
        "original_chains": [],
        "retained_chains": [],
        "removed_chains": [],
        "missing_residues_modeled": 0,
        "nonstandard_replaced": 0,
        "missing_heavy_atoms_added": 0,
        "status": "Success",
        "error_message": ""
    }

    try:
        fixer = PDBFixer(filename=str(input_path))
        all_chains = [c.id for c in fixer.topology.chains()]
        log_info["original_chains"] = all_chains

        # 1. Multi-chain filtering with safety fallback
        if target_chains:
            valid_keep = [c for c in target_chains if c in all_chains]
            if not valid_keep:
                logger.warning(
                    f"None of target chains {target_chains} found in '{input_path.name}' (present: {all_chains}). "
                    f"Retaining default chain '{all_chains[0]}' to prevent creating an empty structure."
                )
                valid_keep = [all_chains[0]]

            remove_indices = [
                i for i, c in enumerate(fixer.topology.chains())
                if c.id not in valid_keep
            ]
            removed_ids = [c.id for i, c in enumerate(fixer.topology.chains()) if i in remove_indices]
            if remove_indices:
                fixer.removeChains(chainIndices=remove_indices)
                log_info["removed_chains"] = removed_ids

            log_info["retained_chains"] = valid_keep
        else:
            log_info["retained_chains"] = all_chains

        # 2. Heterogen and water removal
        fixer.removeHeterogens(keepWater=keep_water)

        # 3. Non-standard residue conversion
        if replace_nonstandard:
            fixer.findNonstandardResidues()
            log_info["nonstandard_replaced"] = len(fixer.nonstandardResidues)
            fixer.replaceNonstandardResidues()

        # 4. Handle missing residues vs. missing atoms
        if build_missing_loops:
            fixer.findMissingResidues()
            missing_res_count = sum(len(res_list) for res_list in fixer.missingResidues.values())
            log_info["missing_residues_modeled"] = missing_res_count
        else:
            # Suppress artificial straight-line terminal loops
            fixer.findMissingResidues()
            fixer.missingResidues = {}
            log_info["missing_residues_modeled"] = 0

        # 5. Model missing heavy atoms on existing, resolved residues
        if add_missing_atoms:
            fixer.findMissingAtoms()
            missing_atoms_count = sum(len(atom_list) for atom_list in fixer.missingAtoms.values())
            missing_term_count = sum(len(atom_list) for atom_list in fixer.missingTerminals.values())
            log_info["missing_heavy_atoms_added"] = missing_atoms_count + missing_term_count
            fixer.addMissingAtoms()

        # 6. Save coordinates directly - no alignment/superposition applied
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f_out:
            PDBFile.writeFile(fixer.topology, fixer.positions, f_out, keepIds=True)

    except Exception as e:
        log_info["status"] = "Failed"
        log_info["error_message"] = str(e)
        logger.error(f"Error repairing '{input_path.name}': {e}")

    return log_info


def batch_clean_ensemble(
    input_dir: str,
    output_dir: str,
    keep_chains: Optional[Union[str, List[str]]] = None,
    keep_water: bool = False,
    replace_nonstandard: bool = True,
    add_missing_atoms: bool = True,
    build_missing_loops: bool = False,
    audit_report_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch clean and repair an ensemble of PDBs, supporting multi-chain complexes.
    No alignment/superposition is performed on any structure — each member's original
    coordinate frame is preserved so pre-existing ensemble alignment (e.g., aligned on
    specific chains/residues for docking) is left intact.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(list(in_path.glob("*.pdb")))
    if not pdb_files:
        logger.error(f"No PDB structures found in '{in_path.resolve()}'.")
        return pd.DataFrame()

    parsed_target_chains = parse_chains(keep_chains)
    chain_desc = f"chains {parsed_target_chains}" if parsed_target_chains else "all chains"
    logger.info(f"Discovered {len(pdb_files)} structure(s). Retaining: {chain_desc}.")

    audit_records = []

    for pdb_file in pdb_files:
        logger.info(f"Processing: '{pdb_file.name}'")
        final_file = out_path / f"clean_{pdb_file.name}"

        clean_log = clean_pdb(
            input_source=pdb_file,
            output_path=final_file,
            keep_chains=parsed_target_chains,
            keep_water=keep_water,
            replace_nonstandard=replace_nonstandard,
            add_missing_atoms=add_missing_atoms,
            build_missing_loops=build_missing_loops
        )

        audit_records.append(clean_log)

    df_audit = pd.DataFrame(audit_records)
    if audit_report_path:
        rep_path = Path(audit_report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        df_audit.to_csv(rep_path, index=False)
        logger.info(f"Audit log saved to '{rep_path.resolve()}'.")

    logger.info("=" * 45)
    logger.info("BATCH ENSEMBLE CLEANUP SUMMARY (no alignment performed)")
    logger.info("=" * 45)
    logger.info(f"Total structures processed : {len(df_audit)}")
    logger.info(f"Successfully repaired      : {(df_audit['status'] == 'Success').sum()}")
    logger.info(f"Cleaned PDB directory      : {out_path.resolve()}")
    logger.info("=" * 45)

    return df_audit

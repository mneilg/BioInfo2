"""
SDF Extractor Module
Extracts molecules from large SDF databases matching query IDs from delimited tabular files.
Supports tag-based extraction and sets the matched ID as the title line (_Name) for downstream pipelines.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union, List
import pandas as pd
from rdkit import Chem

# Configure module logger
logger = logging.getLogger("sdf_extractor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_query_ids(
    file_path: str,
    id_column: str,
    strip_whitespace: bool = True
) -> Set[str]:
    """
    Load unique query IDs from a CSV, TSV, or other delimited file.
    Automatically detects the delimiter.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Query file not found: {path.resolve()}")
        raise FileNotFoundError(f"Query file not found: {path}")

    # Sniff delimiter automatically
    df = pd.read_csv(path, sep=None, engine="python", dtype={id_column: str})
    
    if id_column not in df.columns:
        available = ", ".join([f"'{col}'" for col in df.columns])
        logger.error(f"Column '{id_column}' missing in '{path.name}'. Available: [{available}]")
        raise KeyError(f"Column '{id_column}' not found. Available columns: [{available}]")

    series = df[id_column].dropna().astype(str)
    if strip_whitespace:
        series = series.str.strip()

    query_ids = set(series)
    logger.info(f"Loaded {len(query_ids):,} unique query ID(s) from '{path.name}'.")
    return query_ids


def filter_sdf_by_id(
    sdf_input_path: str,
    sdf_output_path: str,
    query_ids: Set[str],
    sdf_id_tag: Optional[Union[str, List[str]]] = "HMDB_ID",
    set_name_to_matched_id: bool = True,
    missing_report_path: Optional[str] = None,
    stop_when_all_found: bool = False,
    progress_interval: int = 10000
) -> Dict[str, Any]:
    """
    Stream an SDF database, match entries against query IDs, assign the matched ID
    to the title line (_Name), and write matching molecules into a new SDF file.

    Parameters:
    - sdf_input_path: Path to the source SDF database.
    - sdf_output_path: Path where matching molecules will be saved.
    - query_ids: Set of string IDs to match against.
    - sdf_id_tag: Property tag name (e.g. 'HMDB_ID', 'DATABASE_ID') or list of fallback tags.
                  If set to None, falls back to the molecule title line (_Name).
    - set_name_to_matched_id: If True, overwrites mol.SetProp('_Name', mol_id) so the first line
                              of each entry in the output SDF becomes the matched ID.
    - missing_report_path: Optional CSV/TXT path to write unmatched query IDs.
    - stop_when_all_found: Terminate early if all query IDs have been matched.
    - progress_interval: Log progress every N molecules scanned.
    """
    in_path = Path(sdf_input_path)
    out_path = Path(sdf_output_path)

    if not in_path.exists():
        logger.error(f"SDF database not found: {in_path.resolve()}")
        raise FileNotFoundError(f"SDF file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize tag names into a list
    if isinstance(sdf_id_tag, str):
        target_tags = [sdf_id_tag]
    elif isinstance(sdf_id_tag, (list, tuple)):
        target_tags = list(sdf_id_tag)
    else:
        target_tags = []

    supplier = Chem.ForwardSDMolSupplier(str(in_path), sanitize=False, removeHs=False)
    writer = Chem.SDWriter(str(out_path))

    matched_ids: Set[str] = set()
    total_scanned = 0
    total_written = 0
    parse_errors = 0

    tag_desc = f"property tag(s) {target_tags}" if target_tags else "title line (_Name)"
    logger.info(f"Starting scan of '{in_path.name}' matching against {tag_desc}...")

    try:
        for mol in supplier:
            total_scanned += 1

            if total_scanned % progress_interval == 0:
                logger.info(
                    f"Progress: {total_scanned:,} scanned | "
                    f"{len(matched_ids):,} / {len(query_ids):,} IDs matched"
                )

            if mol is None:
                parse_errors += 1
                continue

            # Extract molecule identifier from specified SD property tags or title line
            mol_id = None
            if target_tags:
                for tag in target_tags:
                    if mol.HasProp(tag):
                        val = mol.GetProp(tag).strip()
                        if val:
                            mol_id = val
                            break
            else:
                if mol.HasProp("_Name"):
                    mol_id = mol.GetProp("_Name").strip()

            # Check for match against query set
            if mol_id and mol_id in query_ids:
                # Overwrite the title line (_Name) with the matched ID for downstream scripts
                if set_name_to_matched_id:
                    mol.SetProp("_Name", mol_id)

                writer.write(mol)
                matched_ids.add(mol_id)
                total_written += 1

                if stop_when_all_found and len(matched_ids) == len(query_ids):
                    logger.info(f"Early stop: all {len(query_ids):,} query IDs found at record {total_scanned:,}.")
                    break

    finally:
        writer.close()

    missing_ids = query_ids - matched_ids
    recovery_rate = (len(matched_ids) / len(query_ids) * 100.0) if query_ids else 0.0

    # Write missing IDs report if requested
    if missing_report_path and missing_ids:
        rep_path = Path(missing_report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write("unmatched_id\n")
            for m_id in sorted(missing_ids):
                f.write(f"{m_id}\n")
        logger.info(f"Exported {len(missing_ids):,} missing ID(s) to '{rep_path.resolve()}'.")

    # Output summary
    logger.info("=" * 45)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 45)
    logger.info(f"Total entries scanned     : {total_scanned:,}")
    logger.info(f"Corrupted entries skipped : {parse_errors:,}")
    logger.info(f"Molecules exported        : {total_written:,}")
    logger.info(f"Unique IDs matched        : {len(matched_ids):,} / {len(query_ids):,} ({recovery_rate:.2f}%)")
    logger.info(f"Unique IDs missing        : {len(missing_ids):,}")
    logger.info(f"Extracted SDF location    : {out_path.resolve()}")
    logger.info("=" * 45)

    return {
        "matched_ids": matched_ids,
        "missing_ids": missing_ids,
        "total_scanned": total_scanned,
        "total_written": total_written,
        "parse_errors": parse_errors,
        "recovery_rate": recovery_rate,
    }

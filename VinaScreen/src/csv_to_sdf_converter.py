"""
CSV (SMILES) -> SDF converter with stereoisomer enumeration. Backend module.

- Reads one or more CSV files containing a SMILES column and an ID column.
- Parses each SMILES directly (no re-protonation / pH adjustment), so whatever
  protonation state is encoded in the SMILES (explicit formal charges) is preserved as-is.
- Detects unspecified tetrahedral/double-bond stereocenters that are NOT encoded in the
  SMILES but are chemically possible, and enumerates the resulting 3D stereoisomers.
    - If a molecule has no stereo ambiguity (0 or 1 valid enumerated isomer), it is written
      once to the SDF using its original ID (no suffix).
    - If a molecule has N possible stereoisomers (N > 1), each is embedded in 3D and written
      to the SDF with IDs suffixed _1, _2, ... _N.
- Writes a single combined output SDF (3D, explicit hydrogens retained) and a CSV audit log
  reporting, per input molecule: how many stereoisomers were enumerated, the resulting output
  IDs, and pass/fail status.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

# ==========================================
# LOGGER
# ==========================================
logger = logging.getLogger("csv_to_sdf")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_csv_entries(
    input_source: str,
    smiles_col: str,
    id_col: str
) -> List[Dict[str, Any]]:
    """
    Loads SMILES/ID pairs from a single CSV file or every *.csv file in a directory.
    Returns a flat list of dicts: {source_file, id, smiles}.
    """
    src_path = Path(input_source)
    if src_path.is_dir():
        csv_files = sorted(src_path.glob("*.csv"))
    elif src_path.is_file():
        csv_files = [src_path]
    else:
        raise FileNotFoundError(f"Input path '{input_source}' does not exist.")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at '{input_source}'.")

    entries = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if smiles_col not in df.columns or id_col not in df.columns:
            logger.warning(
                f"Skipping '{csv_file.name}': missing required column(s) "
                f"'{smiles_col}' and/or '{id_col}' (found: {list(df.columns)})."
            )
            continue
        for _, row in df.iterrows():
            mol_id = str(row[id_col]).strip()
            smiles = str(row[smiles_col]).strip()
            if not mol_id or not smiles or smiles.lower() == "nan":
                continue
            entries.append({"source_file": csv_file.name, "id": mol_id, "smiles": smiles})

    logger.info(f"Loaded {len(entries)} SMILES entries from {len(csv_files)} CSV file(s).")
    return entries


def enumerate_stereoisomers(
    mol: Chem.Mol,
    max_isomers: int = 32
) -> List[Chem.Mol]:
    """
    Enumerates 3D-embeddable stereoisomers for stereocenters left unspecified in the
    input SMILES. Stereocenters that ARE already defined in the SMILES are left untouched
    (onlyUnassigned=True). tryEmbedding=True discards isomers that are not physically
    realizable, so the count reflects chemically valid stereoisomers only.
    """
    opts = StereoEnumerationOptions(
        onlyUnassigned=True,
        unique=True,
        tryEmbedding=True,
        maxIsomers=max_isomers
    )
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    return isomers if isomers else [mol]


def embed_3d(mol: Chem.Mol, optimize: bool = True) -> Optional[Chem.Mol]:
    """
    Adds explicit hydrogens, generates a 3D conformer (ETKDGv3), and optionally
    energy-minimizes with MMFF94 (falling back to UFF if MMFF parameters are unavailable).
    Returns None if embedding fails.
    """
    mol_h = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if embed_status != 0:
        # Retry once with random coordinates before giving up
        embed_status = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3(), useRandomCoords=True)
    if embed_status != 0:
        return None

    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h)
            else:
                AllChem.UFFOptimizeMolecule(mol_h)
        except Exception as e:
            logger.warning(f"Optimization failed, keeping embedded (unoptimized) geometry: {e}")

    return mol_h


def process_entry(
    entry: Dict[str, Any],
    max_isomers: int
) -> Tuple[List[Chem.Mol], Dict[str, Any]]:
    """
    Processes a single CSV row: parses SMILES, enumerates stereoisomers, embeds 3D
    coordinates for each. Returns (list_of_output_mols, log_record).
    """
    mol_id = entry["id"]
    smiles = entry["smiles"]
    source_file = entry["source_file"]

    log_record = {
        "source_file": source_file,
        "original_id": mol_id,
        "input_smiles": smiles,
        "num_stereoisomers_enumerated": 0,
        "stereo_variation_applied": "No",
        "output_ids": "",
        "status": "Success",
        "error_message": ""
    }

    output_mols = []

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise ValueError("RDKit could not parse SMILES (invalid structure).")

        # Preserve protonation state exactly as encoded in the SMILES: no pH adjustment,
        # no re-charging step is applied here.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

        isomers = enumerate_stereoisomers(mol, max_isomers=max_isomers)
        log_record["num_stereoisomers_enumerated"] = len(isomers)

        if len(isomers) <= 1:
            # No unresolved stereo ambiguity: write once under the original ID.
            base_mol = isomers[0] if isomers else mol
            embedded = embed_3d(base_mol)
            if embedded is None:
                raise ValueError("3D embedding failed for the unique structure.")
            embedded.SetProp("_Name", mol_id)
            embedded.SetProp("Original_ID", mol_id)
            embedded.SetProp("Source_File", source_file)
            embedded.SetProp("Input_SMILES", smiles)
            embedded.SetProp("Isomeric_SMILES", Chem.MolToSmiles(base_mol))
            output_mols.append(embedded)
            log_record["stereo_variation_applied"] = "No"
            log_record["output_ids"] = mol_id
        else:
            # Multiple possible stereoisomers: embed and write each with a _1, _2, ... suffix.
            generated_ids = []
            for idx, iso in enumerate(isomers, start=1):
                embedded = embed_3d(iso)
                if embedded is None:
                    logger.warning(f"Embedding failed for stereoisomer {idx} of '{mol_id}'; skipping this variant.")
                    continue
                new_id = f"{mol_id}_{idx}"
                embedded.SetProp("_Name", new_id)
                embedded.SetProp("Original_ID", mol_id)
                embedded.SetProp("Source_File", source_file)
                embedded.SetProp("Input_SMILES", smiles)
                embedded.SetProp("Isomeric_SMILES", Chem.MolToSmiles(iso))
                embedded.SetProp("Stereoisomer_Index", f"{idx} of {len(isomers)}")
                output_mols.append(embedded)
                generated_ids.append(new_id)

            if not generated_ids:
                raise ValueError("All enumerated stereoisomers failed 3D embedding.")

            log_record["stereo_variation_applied"] = "Yes"
            log_record["output_ids"] = ";".join(generated_ids)

    except Exception as e:
        log_record["status"] = "Failed"
        log_record["error_message"] = str(e)
        logger.error(f"Error processing '{mol_id}': {e}")

    return output_mols, log_record


def batch_convert_csv_to_sdf(
    input_source: str,
    output_sdf_path: str,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    max_isomers: int = 32,
    log_report_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main driver: loads all SMILES/ID pairs from input_source (a CSV file or a directory of
    CSV files), enumerates unresolved stereoisomers per molecule, embeds 3D coordinates,
    and writes everything to a single combined SDF file. Returns (and optionally saves)
    an audit log DataFrame.
    """
    entries = load_csv_entries(input_source, smiles_col=smiles_col, id_col=id_col)

    out_path = Path(output_sdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))

    log_records = []
    total_written = 0

    for entry in entries:
        output_mols, log_record = process_entry(entry, max_isomers=max_isomers)
        for m in output_mols:
            writer.write(m)
            total_written += 1
        log_records.append(log_record)

    writer.close()

    df_log = pd.DataFrame(log_records)

    if log_report_path:
        rep_path = Path(log_report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        df_log.to_csv(rep_path, index=False)
        logger.info(f"Audit log saved to '{rep_path.resolve()}'.")

    n_with_variants = (df_log["stereo_variation_applied"] == "Yes").sum()
    n_without_variants = (df_log["stereo_variation_applied"] == "No").sum()
    n_failed = (df_log["status"] == "Failed").sum()

    logger.info("=" * 50)
    logger.info("CSV -> SDF CONVERSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Input molecules processed        : {len(df_log)}")
    logger.info(f"Molecules with stereo variants    : {n_with_variants}")
    logger.info(f"Molecules with no stereo ambiguity: {n_without_variants}")
    logger.info(f"Failed molecules                  : {n_failed}")
    logger.info(f"Total 3D records written to SDF   : {total_written}")
    logger.info(f"Output SDF                        : {out_path.resolve()}")
    logger.info("=" * 50)

    return df_log

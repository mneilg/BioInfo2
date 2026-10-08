"""
SDF -> PDBQT ligand preparation module. Backend module.

- Reads an SDF ligand library WITHOUT silently dropping records that fail RDKit
  sanitization due to an over-valent heteroatom (e.g. a protonated secondary/tertiary
  amine encoded with the extra H but missing its '+' formal charge). Such records are
  repaired by assigning the correct formal charge, not discarded.
- Molecules flagged for preservation (by name, or auto-detected via a pre-existing
  non-zero formal charge) keep their original protonation state and 3D coordinates
  exactly as given in the SDF - no Dimorphite-DL re-protonation, no re-embedding.
- All other molecules are re-protonated with Dimorphite-DL at target_ph and re-embedded.
- Writes one PDBQT per molecule and a single CSV audit log covering parsing repairs,
  protonation handling, and PDBQT-write status for every input record.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from dimorphite_dl import protonate_smiles

# ==========================================
# LOGGER
# ==========================================
logger = logging.getLogger("ligand_preparator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

MAX_NEUTRAL_VALENCE = {"N": 3, "O": 2, "S": 2, "P": 3}


def load_and_repair_sdf(input_sdf: str) -> List[Tuple[Optional[Chem.Mol], int, str]]:
    """
    Reads an SDF unsanitized so no record is silently dropped, then sanitizes each
    molecule. If sanitization fails purely because a heteroatom's explicit valence is
    exactly one above its neutral maximum with formal charge 0 (the signature of a
    protonated amine missing its '+'), the formal charge is set to +1 and sanitization
    is retried. Returns a list of (mol, record_index, note); mol is None only if the
    structure is unfixable by this rule.
    """
    raw_supplier = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    results = []

    for idx, mol in enumerate(raw_supplier):
        if mol is None:
            results.append((None, idx, "Unparsable record (no valid mol block)."))
            continue

        try:
            Chem.SanitizeMol(mol)
            results.append((mol, idx, ""))
            continue
        except Chem.AtomValenceException:
            pass
        except Exception as e:
            results.append((None, idx, f"Sanitize failed (non-valence): {e}"))
            continue

        repaired = False
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in MAX_NEUTRAL_VALENCE:
                continue
            explicit_val = atom.GetExplicitValence()
            neutral_max = MAX_NEUTRAL_VALENCE[symbol]
            if atom.GetFormalCharge() == 0 and explicit_val == neutral_max + 1:
                atom.SetFormalCharge(1)
                repaired = True

        if repaired:
            try:
                Chem.SanitizeMol(mol)
                results.append((mol, idx, "Repaired over-valent heteroatom(s) by assigning +1 formal charge."))
                continue
            except Exception as e:
                results.append((None, idx, f"Charge repair attempted but sanitization still failed: {e}"))
                continue

        results.append((None, idx, "AtomValenceException, not resolvable by +1 charge repair."))

    return results


def get_mol_name(mol: Chem.Mol, idx: int) -> str:
    if mol.HasProp('_Name') and mol.GetProp('_Name').strip():
        return mol.GetProp('_Name').strip()
    return f"record_{idx}"


def has_preset_ionization(mol: Chem.Mol) -> bool:
    """Flags molecules whose input structure already encodes a non-neutral
    (deliberately protonated/deprotonated) state via explicit formal charges."""
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def reprotonate_at_ph(mol: Chem.Mol, target_ph: float) -> Optional[Chem.Mol]:
    """Re-protonates a molecule with Dimorphite-DL at target_ph +/- 0.5 and re-embeds
    a fresh 3D conformer. Returns None if protonation, parsing, or embedding fails."""
    original_smiles = Chem.MolToSmiles(mol)
    protonated_smiles_list = protonate_smiles(
        original_smiles,
        ph_min=target_ph - 0.5,
        ph_max=target_ph + 0.5
    )
    if not protonated_smiles_list:
        return None

    best_mol = Chem.MolFromSmiles(protonated_smiles_list[0])
    if best_mol is None:
        return None

    best_mol = Chem.AddHs(best_mol)
    embed_status = AllChem.EmbedMolecule(best_mol, AllChem.ETKDGv3())
    if embed_status != 0:
        AllChem.Compute2DCoords(best_mol)
    AllChem.MMFFOptimizeMolecule(best_mol)
    return best_mol


def preserve_original(mol: Chem.Mol) -> Chem.Mol:
    """Keeps the molecule exactly as provided: original protonation state, original
    explicit hydrogens, and original 3D coordinates. Adds hydrogens only if missing,
    preserving any existing conformer."""
    best_mol = Chem.Mol(mol)
    if best_mol.GetNumConformers() > 0:
        best_mol = Chem.AddHs(best_mol, addCoords=True)
    else:
        best_mol = Chem.AddHs(best_mol)
        embed_status = AllChem.EmbedMolecule(best_mol, AllChem.ETKDGv3())
        if embed_status != 0:
            AllChem.Compute2DCoords(best_mol)
        AllChem.MMFFOptimizeMolecule(best_mol)
    return best_mol


def batch_convert_sdf_to_pdbqt(
    input_sdf: str,
    output_dir: str,
    target_ph: float = 7.0,
    preserve_names: Optional[Set[str]] = None,
    auto_detect_charged_states: bool = True,
    audit_report_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Converts every molecule in input_sdf to an individual PDBQT file. Molecules in
    preserve_names, or auto-detected as already carrying a non-zero formal charge
    (if auto_detect_charged_states=True), keep their original protonation state.
    Everything else is re-protonated with Dimorphite-DL at target_ph. Records that
    fail sanitization due to a fixable over-valent heteroatom are repaired rather
    than silently dropped. Returns the audit DataFrame (and writes it to CSV if a
    path is given).
    """
    preserve_names = preserve_names or set()
    os.makedirs(output_dir, exist_ok=True)
    mk_prep = MoleculePreparation()

    repaired_records = load_and_repair_sdf(input_sdf)
    audit_records = []

    for mol, idx, repair_note in repaired_records:
        if mol is None:
            audit_records.append({
                "record_index": idx,
                "mol_name": f"record_{idx}",
                "protonation_preserved": "N/A",
                "repair_note": repair_note,
                "status": "Dropped",
                "error_message": repair_note
            })
            logger.error(f"Record #{idx} dropped: {repair_note}")
            continue

        mol_name = get_mol_name(mol, idx)
        record = {
            "record_index": idx,
            "mol_name": mol_name,
            "protonation_preserved": "No",
            "repair_note": repair_note,
            "status": "Success",
            "error_message": ""
        }

        try:
            preserve_this_mol = (
                mol_name in preserve_names
                or (auto_detect_charged_states and has_preset_ionization(mol))
            )

            if preserve_this_mol:
                logger.info(f"Preserving original protonation state for {mol_name}.")
                best_mol = preserve_original(mol)
                record["protonation_preserved"] = "Yes"
            else:
                best_mol = reprotonate_at_ph(mol, target_ph)
                if best_mol is None:
                    raise ValueError("Could not determine protonation state via Dimorphite-DL.")

            mk_prep.prepare(best_mol)
            pdbqt_string = mk_prep.write_pdbqt_string()

            output_filepath = os.path.join(output_dir, f"{mol_name}.pdbqt")
            with open(output_filepath, "w") as f:
                f.write(pdbqt_string)

            logger.info(f"Successfully created: {output_filepath}")

        except Exception as e:
            record["status"] = "Failed"
            record["error_message"] = str(e)
            logger.error(f"Error processing {mol_name}: {e}")

        audit_records.append(record)

    df_audit = pd.DataFrame(audit_records)

    if audit_report_path:
        rep_path = Path(audit_report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        df_audit.to_csv(rep_path, index=False)
        logger.info(f"Audit log saved to '{rep_path.resolve()}'.")

    n_dropped = (df_audit["status"] == "Dropped").sum()
    n_failed = (df_audit["status"] == "Failed").sum()
    n_preserved = (df_audit["protonation_preserved"] == "Yes").sum()

    logger.info("=" * 50)
    logger.info("SDF -> PDBQT CONVERSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total records in SDF          : {len(df_audit)}")
    logger.info(f"Successfully written           : {(df_audit['status'] == 'Success').sum()}")
    logger.info(f"Protonation preserved           : {n_preserved}")
    logger.info(f"Dropped (unfixable valence)     : {n_dropped}")
    logger.info(f"Failed (other errors)           : {n_failed}")
    logger.info(f"Output directory                : {Path(output_dir).resolve()}")
    logger.info("=" * 50)

    return df_audit

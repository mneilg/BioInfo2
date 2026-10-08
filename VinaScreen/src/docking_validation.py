"""
docking_validation.py
Reusable functions for validating docking protocols by RMSD comparison
of docked poses (PDBQT) against a co-crystal reference ligand (MOL2).

Strategy: RMSD only requires knowing which reference heavy atom
corresponds to which pose heavy atom — it does not require correct
bond orders/aromaticity/kekulization. So instead of forcing both mols
onto a shared bond-order template (fragile with messy MOL2 files), we
strip Hs from both, then find the atom correspondence via a topology-
only Maximum Common Substructure match (bond order ignored). RMSD is
then computed directly from that atom mapping with plain coordinate
math — no dependency on RDKit's internal RMSD substructure matcher.
"""

import subprocess
import math
from rdkit import Chem
from rdkit.Chem import rdFMCS


def convert_pdbqt_to_sdf(pdbqt_file, sdf_file):
    """Convert docked PDBQT poses to SDF using Meeko's mk_export.py."""
    subprocess.run(
        ["mk_export.py", pdbqt_file, "-s", sdf_file],
        check=True
    )
    return sdf_file


def _load_mol2_heavy(mol2_file):
    """
    Load a MOL2 reference as a heavy-atom-only mol, WITHOUT requiring
    full sanitization/kekulization. We only need element identity,
    connectivity, and 3D coordinates for MCS matching + RMSD — not a
    valid Kekule structure — so we skip the sanitize steps that choke
    on ambiguous MOL2 aromatic ("ar") bond typing.
    """
    mol = Chem.MolFromMol2File(
        mol2_file, removeHs=True, sanitize=False, cleanupSubstructures=False
    )
    if mol is None:
        raise ValueError(f"Could not parse MOL2 file: {mol2_file}")
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def load_raw_reference(crystal_reference_mol2):
    """Load the co-crystal ligand directly from a MOL2 file (heavy atoms only)."""
    return _load_mol2_heavy(crystal_reference_mol2)


def _prep_pose_heavy(pose_mol):
    """Strip Hs from a docked pose (loaded from SDF) for heavy-atom comparison."""
    heavy = Chem.RemoveHs(pose_mol, sanitize=False)
    heavy.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(heavy)
    return heavy


def get_heavy_atom_mapping(ref_mol, pose_mol, timeout=30):
    """
    Find the heavy-atom correspondence between ref_mol and pose_mol using
    a bond-order-agnostic Maximum Common Substructure match. Returns a
    list of (ref_atom_idx, pose_atom_idx) pairs.
    """
    mcs = rdFMCS.FindMCS(
        [ref_mol, pose_mol],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        matchValences=False,
        timeout=timeout,
    )
    if mcs.canceled or mcs.numAtoms == 0:
        raise ValueError(
            "No common heavy-atom substructure found between reference and pose. "
            "This means the two files likely represent different numbers/types "
            "of heavy atoms (e.g. a genuinely different ligand or truncated atoms), "
            "not just a naming/bond-order mismatch."
        )

    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(patt)
    pose_match = pose_mol.GetSubstructMatch(patt)

    if not ref_match or not pose_match:
        raise ValueError("MCS pattern failed to map back onto reference or pose atoms.")

    n_ref = ref_mol.GetNumAtoms()
    n_pose = pose_mol.GetNumAtoms()
    if len(ref_match) < n_ref or len(pose_match) < n_pose:
        print(
            f"  [warning] Partial atom match: {len(ref_match)}/{n_ref} reference "
            f"heavy atoms and {len(pose_match)}/{n_pose} pose heavy atoms matched. "
            "RMSD below is computed over the matched atoms only."
        )

    return list(zip(ref_match, pose_match)), len(ref_match), n_ref


def calc_heavy_atom_rmsd(pose_mol, ref_mol):
    """
    Strip Hs from both mols, map heavy atoms via topology-only MCS, then
    compute plain (no realignment) RMSD directly from the 3D coordinates.
    No realignment is used deliberately: docking poses and the crystal
    reference should already share the receptor's coordinate frame, so a
    best-fit superposition would mask real placement errors.
    """
    ref_heavy = ref_mol  # already heavy-only from load_raw_reference
    pose_heavy = _prep_pose_heavy(pose_mol)

    mapping, n_matched, n_expected = get_heavy_atom_mapping(ref_heavy, pose_heavy)

    ref_conf = ref_heavy.GetConformer()
    pose_conf = pose_heavy.GetConformer()

    sq_sum = 0.0
    for ref_idx, pose_idx in mapping:
        rp = ref_conf.GetAtomPosition(ref_idx)
        pp = pose_conf.GetAtomPosition(pose_idx)
        sq_sum += (rp.x - pp.x) ** 2 + (rp.y - pp.y) ** 2 + (rp.z - pp.z) ** 2

    rmsd = math.sqrt(sq_sum / n_matched)
    return rmsd, n_matched, n_expected


def validate_docking(
    docked_pdbqt_file,
    crystal_reference_mol2,
    temp_sdf_file,
    text_report_file,
    rmsd_cutoff=2.0,
):
    """
    End-to-end validation: convert poses, map heavy atoms between each
    pose and the MOL2 reference by topology only, score direct RMSD,
    write a report.
    """
    ref_raw = load_raw_reference(crystal_reference_mol2)

    convert_pdbqt_to_sdf(docked_pdbqt_file, temp_sdf_file)

    suppl = Chem.SDMolSupplier(temp_sdf_file, removeHs=False)

    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 55)
    report_lines.append("  DOCKING PROTOCOL VALIDATION REPORT")
    report_lines.append("=" * 55)
    report_lines.append(f"  {'Pose':<8} {'RMSD (A)':<14} {'Matched':<10} {'Result'}")
    report_lines.append("-" * 55)

    best_rmsd, best_pose, pass_count, n_total = None, None, 0, 0

    for i, pose in enumerate(suppl):
        n_total = i + 1
        if pose is None:
            report_lines.append(f"  Pose {i+1:<4} {'SKIPPED':<14} {'':<10} Could not parse pose")
            continue
        try:
            rmsd, n_matched, n_expected = calc_heavy_atom_rmsd(pose, ref_raw)
            status = "PASS  <==" if rmsd <= rmsd_cutoff else "FAIL"
            if rmsd <= rmsd_cutoff:
                pass_count += 1
            if best_rmsd is None or rmsd < best_rmsd:
                best_rmsd, best_pose = rmsd, i + 1
            matched_str = f"{n_matched}/{n_expected}"
            report_lines.append(f"  Pose {i+1:<4} {rmsd:<14.3f} {matched_str:<10} {status}")
        except Exception as e:
            report_lines.append(f"  Pose {i+1:<4} {'ERROR':<14} {'':<10} {str(e)}")

    report_lines.append("=" * 55)
    report_lines.append(
        f"  Best pose  : Pose {best_pose} ({best_rmsd:.3f} A)" if best_pose else "  Best pose  : None"
    )
    report_lines.append(f"  Poses pass : {pass_count} / {n_total}")
    report_lines.append(
        f"  Verdict    : {'PROTOCOL VALIDATED' if pass_count > 0 else 'PROTOCOL FAILED'}"
    )
    report_lines.append("=" * 55)

    full_report = "\n".join(report_lines)
    print(full_report)

    with open(text_report_file, "w") as f:
        f.write(full_report)

    return {
        "best_pose": best_pose,
        "best_rmsd": best_rmsd,
        "pass_count": pass_count,
        "n_total": n_total,
        "report": full_report,
    }

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import tempfile
from typing import Tuple, List

def find_most_similar_frag(
    mol_true_frag, mol_pred_frags
):
    """Find the most similar fragment to the true fragment among the predicted
    fragments.

    :param mol_true_frag: True fragment molecule.
    :param mol_pred_frags: List of predicted fragment molecules.
    :return: Tuple of the most similar fragment molecule, the Tanimoto
        similarity, and the RMSD.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate the fingerprint for the true fragment
    fp_true = mfpgen.GetFingerprint(mol_true_frag)

    max_similarity = -1
    max_similar_index = -1
    min_rmsd = float("inf")
    most_similar_frag = None

    for i, mol_pred_frag in enumerate(mol_pred_frags):
        # Skip fragments with different number of atoms
        if mol_pred_frag.GetNumAtoms() != mol_true_frag.GetNumAtoms():
            continue

        # Generate the fingerprint for the predicted fragment
        mol_pred_frag.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol_pred_frag)  # Perceive rings for fingerprinting
        fp_pred = mfpgen.GetFingerprint(mol_pred_frag)

        # Calculate the Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)

        # Calculate the RMSD
        rmsd = (
            AllChem.GetBestRMS(mol_true_frag, mol_pred_frag) if similarity > 0.5 else float("inf")
        )

        # Update the most similar fragment if the current one is more similar or has a lower RMSD
        if similarity > max_similarity or (similarity == max_similarity and rmsd < min_rmsd):
            max_similarity = similarity
            min_rmsd = rmsd
            most_similar_frag = mol_pred_frag
            max_similar_index = i
        # print(similarity, i)
    # print(most_similar_frag, max_similar_index)

    return most_similar_frag, max_similar_index, min_rmsd


def align_poses_to_reference_site(
    gt_protein_path: str,
    gt_ligand_path: str,
    pred_protein_path: str,
    pred_ligand_paths: List[str],
    cutoff: float = 10.0,
) -> Tuple[List[Chem.Mol], str]:
    """
    Align predicted protein to GT binding site and apply same transform to multiple ligands.

    Args:
        gt_protein_path: Ground-truth holo protein PDB.
        gt_ligand_path: Ground-truth ligand file (PDB or SDF).
        pred_protein_path: Predicted apo protein PDB.
        pred_ligand_paths: List of predicted ligand files.
        cutoff: Binding site cutoff distance in Å.

    Returns:
        aligned_ligand_mols: List of RDKit Mol of aligned predicted ligands.
        aligned_protein_tempfile_path: Temp file path of aligned predicted protein.
    """
    from pymol import cmd

    cmd.reinitialize()

    # Load GT structures
    cmd.load(gt_protein_path, "gt_protein")
    cmd.load(gt_ligand_path, "gt_ligand")

    # Load predicted protein
    cmd.load(pred_protein_path, "pred_protein")

    # Heavy atoms
    cmd.select("gt_protein_heavy", "gt_protein and not elem H")
    cmd.select("gt_ligand_heavy", "gt_ligand and not elem H")

    # Binding site
    cmd.select(
        "binding_site",
        f"gt_protein_heavy within {cutoff} of gt_ligand_heavy"
    )

    # Align predicted protein once
    cmd.align("pred_protein", "binding_site")

    aligned_ligand_mols = []

    for i, ligand_path in enumerate(pred_ligand_paths):
        ligand_obj = f"pred_ligand_{i}"

        # Load ligand
        cmd.load(ligand_path, ligand_obj)

        # Apply same transformation
        cmd.matrix_copy("pred_protein", ligand_obj)

        # Save aligned ligand
        tmp_ligand = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
        cmd.save(tmp_ligand.name, ligand_obj)

        # Load with RDKit
        mol = Chem.SDMolSupplier(tmp_ligand.name, removeHs=False)[0]
        aligned_ligand_mols.append(mol)

        cmd.delete(ligand_obj)

    # Save aligned protein
    tmp_protein = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    cmd.save(tmp_protein.name, "pred_protein")

    return aligned_ligand_mols, tmp_protein.name

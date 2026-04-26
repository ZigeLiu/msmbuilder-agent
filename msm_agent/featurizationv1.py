from pathlib import Path
import os
import glob
import time
import itertools
from functools import partial
import numpy as np
import mdtraj as md
import msmbuilder.cluster as cluster_module

NORM_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
TERM_AA =['ACE', 'NME']
MODIFIED_AA = ['MSE', 'PTR', 'SEP', 'TPO', 'HIE'] #### include or not
AA_NAMES = NORM_AA + TERM_AA
N_NAMES = ['DA', 'DC', 'DG', 'DT', 'U', 'A', 'C', 'G']
SOLV_NAMES = ['HOH', 'WAT', 'SOL']
LIGAND_NAMES = ['LIG']

FEATURE_SET = {
    "CA_distances": {
        "name": "CA_distances",
        "description": "Alpha carbon pairwise distances",
        "use_case": "Default for large systems, good balance between speed and accuracy",
        "parameters": {
            "atom_selection": ["name CA"],
            "type": "distance",
            "selection": "distances"
            }
    },
    "heavy_atom_distances": {
        "name": "heavy_atom_distances",
        "description": "Heavy atom pairwise distances",
        "use_case": "Good for small systems, can capture more detailed interactions but noiser",
        "parameters": {
            "atom_selection": ["not element H"], ################## have agent updated
            "type": "distance",
            "selection": "distances"
            }
    },
    "backbone_torsions": {
        "name": "backbone_torsions",
        "description": "Backbone torsion angles (phi, psi)",
        "use_case": "Good for capturing backbone transitions",
        "parameters": {
            "atom_selection": ["backbone"],
            "type": "angle",
            "selection": ["phi", "psi"]
        }
    },
    "side_chain_torsions": {
        "name": "side_chain_torsions",
        "description": "Side chain torsion angles (chi)",
        "use_case": "Good for capturing side chain rearrangements",
        "parameters": {
            "atom_selection": ["not element H"],
            "type": "angle",
            "selection": ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "omega"] 
                    }
    },
    "interface": {
        "name": "interface",
        "description": "Distances between specified interface residues",
        "use_case": "Good for studying interactions between specific regions (e.g. protein-protein interfaces)",
        "parameters": {
            "atom_selection": [["not element H"], ["not element H"]], ################ have agent update 
            "type": "distance",
            "selection": "distances" 
        },
    }
}

def _make_run_dir(cfg: dict) -> Path:
    base = Path(cfg["run"]["output_dir"])
    base.mkdir(parents=True, exist_ok=True)
    name = cfg["run"]["run_name"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def inspect_data(cfg: dict):
    try:
        top = md.load(cfg["data"]["topology"]).topology
    except Exception as e:
        raise ValueError(f"Error loading topology file: {e}")
    seq = {}
    for chain in top.chains:
        seq[chain.index] = []
        for res in chain.residues:
            seq[chain.index].append(res.name)
    entity = {}
    for chain in top.chains:
        c = seq[chain.index]
        if set(c).issubset(set(AA_NAMES)):
            entity[chain.index] = "protein"
        elif set(c).issubset(set(N_NAMES)):
            entity[chain.index] = "nucleic_acid"
        elif set(c).issubset(set(SOLV_NAMES)):
            entity[chain.index] = "solvent"
        elif set(c).issubset(set(LIGAND_NAMES)):
            entity[chain.index] = "ligand"
        else:
            entity[chain.index] = "other"
    return {
        "n_atoms": top.n_atoms,
        "n_residues": top.n_residues,
        "chain_lengths": [len(seq[chain.index]) for chain in top.chains],
        "entity": [entity[chain.index] for chain in top.chains]
    }

def decide_feature_selection(cfg: dict, user_request: str = "") -> dict:
    inspect = inspect_data(cfg)
    text = user_request.lower()
    decision = {
        "feature": None,
        "fallbacks": [],
        "reason": [],
        "warnings": [],
    }

    has_ligand = (inspect['entity'].count('ligand') > 0)
    has_nucleic_acid = (inspect['entity'].count('nucleic_acid') > 0)
    n_res = inspect.get("n_residues", 0)

    # User intent flags
    wants_angles = any(k in text for k in [
        "angle", "torsion", "dihedral", "rotamer", "sidechain"
    ])
    wants_binding = any(k in text for k in [
        "ligand", "binding", "unbinding", "pocket", "pose"
    ])

    if has_nucleic_acid:
        decision["feature"] = FEATURE_SET["heavy_atom_distances"]
        decision["feature"]["parameters"]["atom_selection"] = ["name N or name P"]
        decision["fallbacks"].append(FEATURE_SET["heavy_atom_distances"])
        decision["reason"].append(
            "Nucleic acid detected; try N P first, fall back to heavy atom."
        )
        return decision

    if has_ligand and wants_binding:
        ligands = [e == 'ligand' for e in inspect['entity']]
        ligand_id = np.where(ligands)[0]

        decision["feature"] = FEATURE_SET["interface"]
        decision["feature"]["parameters"]["atom_selection"][0].append("protein and name CA and not (resname ACE or resname NME)")
        decision['feature']['parameters']['atom_selection'][1].append("or ".join([f"chainid {id}" for id in ligand_id]))

        decision["fallbacks"].append(FEATURE_SET["heavy_atom_distances"])
        decision["reason"].append(
            "Ligand detected; try protein-ligand interface features, fall back to heavy atom."
        )
        return decision

    if wants_angles:
        decision["feature"] = FEATURE_SET["backbone_torsions"]
        decision["fallbacks"].append(FEATURE_SET["side_chain_torsions"])
        decision["reason"].append(
            "User requested angular features."
        )
        return decision

    if n_res <= 5:
        decision["feature"] = FEATURE_SET["backbone_torsions"]
        decision["fallbacks"].append(FEATURE_SET["heavy_atom_distances"])
        decision["reason"].append(
            "Very small peptide detected; torsions are usually more meaningful than CA distances."
        )
        return decision

    # 8. Large protein default
    decision["feature"] = FEATURE_SET["CA_distances"]
    decision["reason"].append("Default to CA distances.")

    return decision

def _find_featurizer(frame, feature_selection, atom_selection, dist_cutoff):
    atom_slice, atom_slice_1, atom_slice_2 = None, None, None
    try:
        if len(atom_selection) == 1: # single stom selection
            atom_slice = " and ".join(f"{x}" for x in atom_selection)
            atom_slice = frame.topology.select(atom_slice)
            pairs = list(itertools.combinations(atom_slice, 2))
        elif len(atom_selection) == 2: # two set of selected atoms
            atom_slice_1 = " and ".join(f"{x}" for x in atom_selection[0])
            atom_slice_2 = " and ".join(f"{x}" for x in atom_selection[1])
            atom_slice_1 = frame.topology.select(atom_slice_1)
            atom_slice_2 = frame.topology.select(atom_slice_2)
            pairs = list(itertools.product(atom_slice_1, atom_slice_2))
        else:
            raise ValueError(f"atom_selection must be either a single selection or a list of two selections. Got: {atom_selection}")
    except Exception:
        raise ValueError(f"Error parsing atom_selection: {atom_selection}. Must be one or two lists of strings compatible with mdtraj's topology.select syntax.")

    if feature_selection in ["distances", "displacements"]:
        return partial(getattr(md, f"compute_{feature_selection}"), atom_pairs=pairs), pairs
    elif feature_selection == "neighbors":
        assert atom_slice is not None, "Neighbors feature requires a single set of selected atoms."
        return partial(getattr(md, f"compute_{feature_selection}"), cutoff=dist_cutoff, query_indices=atom_slice), pairs
    else:
        fun_list = []
        for angle_feature in feature_selection:
            fun_list.append(partial(getattr(md, f"compute_{angle_feature}")))
        return fun_list if len(fun_list) >0 else None, None
    
def _transform_data(featurizer, traj):
    if isinstance(featurizer, list):
        features = [f(traj)[1] for f in featurizer]
        return np.concatenate(features, axis=1)
    else:
        return featurizer(traj)
    
def _load_feature(cfg: dict, run_dir: Path):
    kind = cfg["data"]["kind"]
    assert kind in ["xtc", "dcd", "trr"], f"Unsupported data.kind: {kind}. Supported: xtc, dcd, trr"

    data_dir = cfg["data"]["dir"]
    top = cfg["data"]["topology"]
    stride = int(cfg["data"].get("stride", 1))
    feature_type = cfg["features"]["type"]
    feature_selection = cfg["features"]["selection"] # list of angles or single distacne type
    dist_cutoff = float(cfg["features"].get("distance_cutoff", 0.8))
    atom_selection = cfg["features"].get("atom_selection", None)
    prepossed_dir = cfg["data"].get("load_preprocessed_dir", None)

    if not data_dir or not top:
        raise ValueError("Both data_dir and topology are required for kind=xtc,dcd,trr")

    files = list(glob.glob(os.path.join(data_dir, f"*.{kind}")))
    loaded_features = []
    if prepossed_dir is not None:
        print("Features already exist, loading from disk...")
        for file in files:
            feature_file = Path(prepossed_dir) / (Path(file).stem + ".npy")
            try:
                loaded_features.append(np.load(feature_file))
            except Exception as e:
                raise ValueError(f"Error loading feature file {feature_file}: {e}")
    else:
        frame = md.load(top)
        if feature_type == "angle":
            assert len(feature_selection) > 1, "Must specify at least two angle types"
            assert all(angle_feature in ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "omega"] \
                        for angle_feature in feature_selection), f"Unsupported angle type: {feature_selection}. Supported: phi, psi, chi1, chi2, chi3, chi4, omega"
        elif feature_type == "distance":
            assert feature_selection in ["distances", "displacements", "neighbors"], f"Unsupported distance type: {feature_selection}. Supported: distances, displacements, neighbors"
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}. Supported: angle, distance")
        featurizer, pairs = _find_featurizer(frame, feature_selection, atom_selection, dist_cutoff)
        assert featurizer is not None, f"Could not find featurizer for selection: {feature_selection}, {atom_selection}"

        (run_dir / "features").mkdir(exist_ok=True)
        contact = {}
        for file in files:
            traj = md.load(file, top=top, stride=stride)
            processed_feature = _transform_data(featurizer,traj)
            out_file = str(run_dir / "features" / (Path(file).stem + ".npy"))
            np.save(out_file, processed_feature)
            if feature_type == "distance":
                contact_freq = (processed_feature < dist_cutoff).mean(axis=0)
                contact[len(processed_feature)] = contact_freq # dict of traj length to contact frequency for each pair
            loaded_features.append(processed_feature)
        total_contact_freq = np.mean([int(key) * float(val) for key, val in contact.items()])
        np.savez(run_dir / f"contact_freq_{dist_cutoff}.npz", contact_freq=total_contact_freq, pairs=pairs)
        ################################## retrieve later for feature selection ######################################
    dt_ps = float(cfg["data"]["saving_interval"]) * stride
    return loaded_features, dt_ps

def _find_clusterer(random_state, cl_cfg):
    method = cl_cfg["method"]
    assert method in ["KCenters","KMeans","KMedoids","MiniBatchKMedoids","MiniBatchKMeans"], \
        f"Unsupported clustering method: {method}. Supported: KCenters, KMeans, KMedoids, MiniBatchKMedoids, MiniBatchKMeans"
    return getattr(cluster_module, method)(n_clusters=int(cl_cfg["n_clusters"]), random_state=random_state, \
                                            **{k: cl_cfg[k] for k in cl_cfg if k not in ["method", "n_clusters","tiny_threshold"]})

def _save_intermediate(data, out_path: Path):
    out_path.mkdir(exist_ok=True)
    feature_dir = out_path.parent / "features" # refer to feature file names for naming consistency
    file_name = glob.glob(str(feature_dir / "*.npy"))
    for i,file in enumerate(file_name):
        name = Path(file).stem
        np.save(out_path / f"{name}.npy", data[i])
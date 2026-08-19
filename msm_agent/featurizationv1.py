from pathlib import Path
import os, json, html
import glob
import itertools
from functools import partial
import numpy as np
import mdtraj as md
import msmbuilder.cluster as cluster_module
from msm_agent.parameters import Metric_param
from msm_agent.config import save_config

NORM_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
TERM_AA =['ACE', 'NME']
MODIFIED_AA = ['MSE', 'PTR', 'SEP', 'TPO', 'HIE', 'NLE'] #### include or not
AA_NAMES = NORM_AA + TERM_AA + MODIFIED_AA
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
            "pair_selection": None, #### loaded if exist #####
            "type": "distance",
            "selection": "distances" 
        },
    }
}

def inspect_data(cfg: dict):
    try:
        top = md.load(cfg["data"]["topology"]).topology
    except Exception as e:
        raise ValueError(f"Error loading topology file: {e}")
    seq = {}
    for chain in top.chains:
        seq[chain.index] = []
        for res in chain.residues:
            unique_atom = np.unique([atom.element.symbol for atom in res.atoms])
            if unique_atom.size == 1 and unique_atom[0] == 'H': # remove added H 
                #print(f"Skipping residue {res.name} in chain {chain.index} because it only contains hydrogen atoms.")
                continue
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
        #"n_residues": top.n_residues,
        "chain_lengths": [len(seq[chain.index]) for chain in top.chains],
        "entity": [entity[chain.index] for chain in top.chains]
    }

def decide_feature_selection(cfg: dict, request: str, run_dir: str | Path) -> dict:
    inspect = inspect_data(cfg)
    text = request.lower()
    decision = {
        "feature": None,
        "fallbacks": [],
        "reason": [],
        "warnings": [],
    }
    # contact test result exist, prioritized
    contact_test_path = run_dir / f"contact_freq_{Metric_param.distance_cutoff}.npz"
    if "contact frequency" in text and os.path.exists(contact_test_path):
        pairs = str(contact_test_path) # track file path 
        decision["feature"] = FEATURE_SET["interface"]
        decision["feature"]["parameters"]["pair_selection"] = pairs
        decision["reason"].append(
            "Loading atomic pairs in contact with high frequency from contact test."
        )
        return decision

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

    if "{" in text and "}" in text:
        sele = text.split("{")[1].split("}")[0].strip().split(",")
        chain = [s.split(":")[0].strip() for s in sele]
        seleid = [s.split(":")[1].strip() for s in sele]
        decision["feature"] = FEATURE_SET["interface"]
        decision["feature"]["parameters"]["atom_selection"][0].append(f"chainid {chain[0]} and name CA and ".join([f"resSeq {id}" for id in seleid[0]]))
        decision['feature']['parameters']['atom_selection'][1].append(f"chainid {chain[1]} and name CA and ".join([f"resSeq {id}" for id in seleid[1]]))
        decision["reason"].append(
            f"User provided residue selections: {sele}. Using these selections for interface feature."
        )
        return decision

    if has_nucleic_acid:
        decision["feature"] = FEATURE_SET["heavy_atom_distances"]
        decision["feature"]["parameters"]["atom_selection"] = ["name N or name P"]
        decision["fallbacks"].append(FEATURE_SET["heavy_atom_distances"])
        decision["reason"].append(
            "Nucleic acid detected; try N P first, fall back to heavy atom."
        )
        return decision

    if has_ligand and wants_binding: # ligand binding
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

def _find_featurizer(frame, feature_selection, atom_selection, dist_cutoff, pair_selection=None):
    atom_slice, atom_slice_1, atom_slice_2 = None, None, None

    # Angle features do not use atom/pair selections. Handle them before the
    # distance-selection code so a perfectly valid ``atom_selection: null``
    # does not reject torsion features.
    if isinstance(feature_selection, (list, tuple)):
        return [partial(getattr(md, f"compute_{angle}")) for angle in feature_selection], None

    if pair_selection is not None:
        try:
            if isinstance(pair_selection, Path) or isinstance(pair_selection, str):
                file = np.load(Path(pair_selection), allow_pickle=True)
                pairs = file['pairs'] #np.load(pair_selection)
            elif isinstance(pair_selection, (list, np.ndarray)):
                pairs = pair_selection
        except Exception as e:
            raise ValueError(f"Error loading pair selection file: {pair_selection}. Error: {e}")
    else:
        try:
            # Preserve the original featurizer's default: distance features
            # without an explicit atom selection use alpha carbons.
            if atom_selection is None:
                atom_selection = "name CA"
            if isinstance(atom_selection, str):
                atom_slice = frame.topology.select(atom_selection)
                pairs = list(itertools.combinations(atom_slice, 2))
            elif isinstance(atom_selection, list):
                if atom_selection and all(isinstance(sel, str) for sel in atom_selection):
                    # A list of strings is one selection made from multiple
                    # rules, e.g. ["protein", "name CA"].
                    atom_slice = " and ".join(atom_selection)
                    atom_slice = frame.topology.select(atom_slice)
                    pairs = list(itertools.combinations(atom_slice, 2))
                elif (len(atom_selection) == 2 and
                      all(isinstance(group, list) and
                          all(isinstance(rule, str) for rule in group)
                          for group in atom_selection)):
                    # Two lists describe two atom groups whose Cartesian
                    # product forms an interface pair selection.
                    atom_slice_1 = " and ".join(atom_selection[0])
                    atom_slice_2 = " and ".join(atom_selection[1])
                    atom_slice_1 = frame.topology.select(atom_slice_1)
                    atom_slice_2 = frame.topology.select(atom_slice_2)
                    pairs = list(itertools.product(atom_slice_1, atom_slice_2))
                else:
                    raise ValueError(f"Invalid atom selection structure: {atom_selection}")
            else:
                raise ValueError(f"atom_selection must be either a string or list of selections. Got: {atom_selection}")
        except Exception as e:
            raise ValueError(
                f"Error parsing atom_selection: {atom_selection}. Must be an mdtraj "
                "selection string, a list of selection rules, or two lists of rules. "
                f"Cause: {e}"
            ) from e

    if feature_selection in ["distances", "displacements"]:
        return partial(getattr(md, f"compute_{feature_selection}"), atom_pairs=pairs), pairs
    elif feature_selection == "neighbors":
        assert atom_slice is not None, "Neighbors feature requires a single set of selected atoms."
        return partial(getattr(md, f"compute_{feature_selection}"), cutoff=dist_cutoff, query_indices=atom_slice), pairs
    raise ValueError(f"Unsupported feature selection: {feature_selection}")
    
def _transform_data(featurizer, traj):
    if isinstance(featurizer, list):
        features = [f(traj)[1] for f in featurizer]
        return np.concatenate(features, axis=1)
    else:
        return featurizer(traj)
    
def load_feature(cfg: dict, run_dir: Path):
    kind = cfg["data"].get("kind", None)
    if kind is not None:
        assert kind in ["xtc", "dcd", "trr"], f"Unsupported data.kind: {kind}. Supported: xtc, dcd, trr"
    data_dir = cfg["data"].get("dir", None)
    top = cfg["data"].get("topology", None)
    stride = int(cfg["data"].get("stride", 1))
    preprocessed_dir = cfg["data"].get("load_preprocessed_dir", None)
    note = cfg["data"].get("note")

    loaded_features = []
    if preprocessed_dir is not None:
        cfg['features']['type'] = "preprocessed"
        cfg['features']['selection'] = "preprocessed"
        cfg['features']['atom_selection'] = "preprocessed"
        cfg['features']['pair_selection'] = "preprocessed"
        print("Features already exist, loading from disk...")
        files = list(glob.glob(os.path.join(preprocessed_dir, "*.npy")))
        for file in files:
            try:
                loaded_features.append(np.load(file))
            except Exception as e:
                raise ValueError(f"Error loading feature file {file}: {e}")
    else:
        if not data_dir or not top:
            raise ValueError("Both data_dir and topology are required for processing data of kind xtc, dcd, trr")
        # decide feature selection 
        if cfg["features"]["type"] is None: # empty feature, update
            decision = decide_feature_selection(cfg, note, run_dir)
            cfg['features'] = decision['feature']['parameters']  # update cfg with the selected feature parameters
        feature_type = cfg["features"]["type"]
        feature_selection = cfg["features"]["selection"] # list of angles or single distacne type
        pair_selection = cfg["features"].get("pair_selection", None)
        atom_selection = cfg["features"].get("atom_selection", None)

        files = sorted(glob.glob(os.path.join(data_dir, f"*.{kind}")))
        assert files, f"No {kind} files found in {data_dir}"
        frame = md.load(top)
        if feature_type == "angle":
            if isinstance(feature_selection, str):
                feature_selection = [feature_selection]
            if not isinstance(feature_selection, list) or not feature_selection:
                raise ValueError("Angle selection must be an angle name or a non-empty list of angle names")
            supported_angles = ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "omega"]
            unsupported_angles = [angle for angle in feature_selection if angle not in supported_angles]
            if unsupported_angles:
                raise ValueError(
                    f"Unsupported angle type(s): {unsupported_angles}. "
                    f"Supported: {', '.join(supported_angles)}"
                )
            # Save the normalized representation to the generated config.
            cfg["features"]["selection"] = feature_selection
        elif feature_type == "distance":
            assert feature_selection in ["distances", "displacements", "neighbors"], f"Unsupported distance type: {feature_selection}. Supported: distances, displacements, neighbors"
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}. Supported: angle, distance")
        featurizer, pairs = _find_featurizer(frame, feature_selection, atom_selection, Metric_param.distance_cutoff, pair_selection)
        assert featurizer is not None, f"Could not find featurizer for selection: {feature_selection}, {atom_selection}"

        #(run_dir / "features").mkdir(exist_ok=True)
        contact = {}
        for i, file in enumerate(files):
            traj = md.load(file, top=top, stride=stride)
            processed_feature = _transform_data(featurizer,traj)
            #out_file = str(run_dir / "features" / f"_{i+1}.npy")
            #np.save(out_file, processed_feature)
            if feature_type == "distance":
                contact_freq = (processed_feature < Metric_param.distance_cutoff).mean(axis=0) # [n_pairs]
                contact[len(processed_feature)] = contact_freq # dict of traj length to contact frequency for each pair
            loaded_features.append(processed_feature)
        if contact:
            total_contact_freq = np.mean([int(key) * np.array(val,dtype=float) for key, val in contact.items()]) if contact else None
            np.savez(run_dir / f"contact_freq_{Metric_param.distance_cutoff}.npz", contact_freq=total_contact_freq, pairs=pairs)
        ################################## retrieve later for feature selection ######################################
    dt_ps = float(cfg["data"]["saving_interval"]) * stride
    save_config(cfg, run_dir / "config.yaml")
    return loaded_features, dt_ps

def find_clusterer(cl_cfg):
    method = cl_cfg["method"]
    assert method in ["KCenters","KMeans","KMedoids","MiniBatchKMedoids","MiniBatchKMeans"], \
        f"Unsupported clustering method: {method}. Supported: KCenters, KMeans, KMedoids, MiniBatchKMedoids, MiniBatchKMeans"
    return getattr(cluster_module, method)(n_clusters=int(cl_cfg["n_clusters"]), random_state=cl_cfg.get("random_seed", None))

def _save_intermediate(data, out_path: Path):
    out_path.mkdir(exist_ok=True)
    #feature_dir = out_path.parent / "features" # refer to feature file names for naming consistency
    #file_name = glob.glob(str(feature_dir / "*.npy"))
    #for i, in enumerate(file_name):
    #    name = Path(file).stem
    #    np.save(out_path / f"{name}.npy", data[i])
    for i in range(len(data)):
        np.save(out_path / f"{i+1}.npy", data[i]) ## if using sorted the name should be consistent

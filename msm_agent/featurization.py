from pathlib import Path
import os
import glob
import time
import itertools
from functools import partial
import numpy as np
import mdtraj as md
import msmbuilder.cluster as cluster_module


def _make_run_dir(cfg: dict) -> Path:
    base = Path(cfg["run"]["output_dir"])
    base.mkdir(parents=True, exist_ok=True)
    name = cfg["run"]["run_name"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _find_featurizer(frame, feature_selection, atom_selection):
    if atom_selection is None or atom_selection.upper() == "CA": # default to CA
        atom_slice = frame.topology.select("name CA")
    elif atom_selection.upper() == "BACKBONE":
        atom_slice = frame.topology.select("backbone")
    elif atom_selection.upper() == "HEAVY":
        atom_slice = frame.topology.select("not element H")
    else:
        try:
            atom_slice = frame.topology.select(atom_selection)
        except Exception:
            raise ValueError(f"Accepted atom selections are : CA, BACKBONE, HEAVY, \
                    or a custom selection string compatible with mdtraj's topology.select syntax. Got: {atom_selection}")

    if feature_selection in ["distances", "displacements"]:
        pairs = list(itertools.combinations(atom_slice, 2))
        return partial(getattr(md, f"compute_{feature_selection}"), atom_pairs=pairs)
    elif feature_selection == "neighbors":
        return partial(getattr(md, f"compute_{feature_selection}"), cutoff=5, query_indices=atom_slice)
    else:
        fun_list = []
        for angle_feature in feature_selection:
            fun_list.append(partial(getattr(md, f"compute_{angle_feature}")))
        return fun_list if len(fun_list) >0 else None

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
        featurizer = _find_featurizer(frame, feature_selection, atom_selection)
        assert featurizer is not None, f"Could not find featurizer for selection: {feature_selection}, {atom_selection}"

        (run_dir / "features").mkdir(exist_ok=True)
        for file in files:
            traj = md.load(file, top=top, stride=stride)
            processed_feature = _transform_data(featurizer,traj)
            out_file = str(run_dir / "features" / (Path(file).stem + ".npy"))
            np.save(out_file, processed_feature)
            loaded_features.append(processed_feature)

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
    feature_dir = out_path.parent / "features"
    file_name = glob.glob(str(feature_dir / "*.npy"))
    for i,file in enumerate(file_name):
        name = Path(file).stem
        np.save(out_path / f"{name}.npy", data[i])
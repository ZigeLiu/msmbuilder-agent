#msm_agent/msm_agent/stage.py
from __future__ import annotations

import os
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional
import glob
import json
import time
import pickle

from dataclasses import asdict
import numpy as np
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel
import msmbuilder.lumping as lump_module
from msmbuilder import tpt

# ===== Adjust these imports to your real project layout =====
from msm_agent.featurizationv1 import (
    _save_intermediate,
    find_clusterer,
    load_feature,
)

from msm_agent.summary import (
    build_stage1_summary,
    build_stage2_summary,
    build_stage3_summary,
    build_stage4_summary,
    build_stage5_summary,
    build_stage6_summary,
    build_stage7_summary,
)
from msm_agent.metrics import (
    compute_msm_its, 
    compute_tica_its,
    in_contact_test, 
    its_plateau_check, 
    compute_occupancy_stats,
    compute_transition_sparsity,
    ck_test,
)
from msm_agent.plots import (
    plot_its_curve,
    plot_occupancy_hist,
    plot_projection,
    plot_free_energy,
)
from msm_agent.parameters import Metric_param
from msm_agent.config import AgentConfig, save_config

# ----------------------------
# JSON / IO helpers
# ----------------------------
def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def write_json(obj: Any, path: Path) -> None:
    if path.exists():
        existing_data = json.loads(path.read_text())
        obj = existing_data | obj
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"JSON file not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_processed_from_run_dir(run_dir: str | Path, data_type: str | None = None) -> List[np.ndarray]:
    run_dir = Path(run_dir)
    if data_type is None:
        load_dir = run_dir
    else:
        assert data_type in ["features", "tica_trajs", "clustered_trajs", "macro_trajs"], f"Unknown data_type: {data_type}"
        load_dir = run_dir / data_type
    if not load_dir.exists():
        #raise ValueError(f"Directory not found: {load_dir}")
        print(f"Directory not found: {load_dir}")
        return None

    files = sorted(glob.glob(str(load_dir / "*.npy")))
    if not files:
        raise ValueError(f"No files found in: {load_dir}")

    return [np.load(f) for f in files]


# ----------------------------
# Stage 1
# ----------------------------
def run_stage1_featurization(cfg: AgentConfig, run_dir: Path = None) -> Dict[str, Any]:
    cfg = asdict(cfg)  
    try:
        # priority: processed, pair selection, atom selection
        features, dt_ps_effective = load_feature(cfg, run_dir)
        traj_lens = [len(x) for x in features]
        feature_dims = [
            int(x.shape[1]) if getattr(x, "ndim", None) == 2 else None
            for x in features
        ]
        _save_intermediate(features, run_dir / "features")
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage1_featurization",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage1_featurization",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
    dist_cutoff = Metric_param.distance_cutoff 
    freq_cutoff = Metric_param.conact_freq_threshold 
    contact_test = in_contact_test(run_dir, dist_cutoff, freq_cutoff) # return dict or empty

    manifest = {
        "stage1": {
            "dt_ns_effective": float(dt_ps_effective / 1000.0),
            "n_trajs": len(features),
            "traj_lens": np.unique(traj_lens).tolist(),
            "feature_dims": np.unique(feature_dims).tolist(),
            "feature_dir": str(run_dir / "features"),
            "in_contact_fraction": contact_test.get("in_contact_fraction", "Do not apply"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")

    summary = build_stage1_summary(
        cfg=cfg,
        run_dir=run_dir,
        n_trajs=len(features),
        traj_lens=traj_lens,
        feature_dims=feature_dims,
        contact_test=contact_test,
        dt_ps_effective=float(dt_ps_effective),
    )

    return {
        "success": True,
        "stage": "stage1_featurization",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": None,
    }


# ----------------------------
# Stage 2
# ----------------------------
def run_stage2_tica_scan(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    cfg = asdict(cfg)
    
    try:
        manifest = read_json(run_dir / "manifest.json")
        features = load_processed_from_run_dir(run_dir, "features")

        dt_ns = float(manifest["stage1"]["dt_ns_effective"])
        assert len(manifest["stage1"]["feature_dims"]) == 1, "All features must have the same dimension for tICA."
        feat_dim = int(manifest["stage1"]["feature_dims"][0])

        tica_cfg = cfg["tica"]
        lag_min = int(tica_cfg["lag_time_frames_range"][0])
        lag_max = int(tica_cfg["lag_time_frames_range"][1])
        grid_size = int(tica_cfg["lag_time_frames_grid_size"])
        n_components = int(tica_cfg["n_components"])
        n_components = min(feat_dim, n_components) 
        # update cfg and save
        cfg["tica"]["n_components"] = n_components
        save_config(cfg, run_dir / "config.yaml")

        lag_list = np.linspace(lag_min, lag_max, num=grid_size, dtype=int)
        lag_list = np.unique(lag_list)

        tica_its = compute_tica_its(
            features=features,
            lag_list=lag_list,
            n_components=n_components,
            dt_ns=dt_ns,
        )

        plot_path = run_dir / "figs" / "tica_its_curve.png"
        plot_its_curve(
            tica_its,
            outpath=plot_path,
        )
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage2_tica_param_scan",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage2_tica_param_scan",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
    plateau_check = its_plateau_check(tica_its, lag_list.tolist())
    manifest = {
        "stage2": {
            "plot_path": str(plot_path),
            "tica_its": tica_its,
            "plateau_check": plateau_check,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")
    
    summary = build_stage2_summary(
        cfg=cfg,
        run_dir=run_dir,
        lag_list=lag_list.tolist(),
        dt_ns=dt_ns,
        plot_path=plot_path,
        plateau_check=plateau_check,
    )

    return {
        "success": True,
        "stage": "stage2_tica_param_scan",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": [str(plot_path)],
    }


# ----------------------------
# Stage 3
# ----------------------------
def run_stage3_tica_fit(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    cfg = asdict(cfg)
    
    try:
        features = load_processed_from_run_dir(run_dir, "features")

        tica_cfg = cfg["tica"]
        if tica_cfg["selected_lag_time"] is None:
            raise ValueError(
                "cfg['tica']['selected_lag_time'] in frames is required for Stage 3. "
                "Please set it first, for example after reviewing Stage 2."
            )

        selected_lag_time = int(tica_cfg["selected_lag_time"])
        selected_n_components = int(tica_cfg.get("selected_n_components", tica_cfg["n_components"]))

        tica_model = tICA(
            lag_time=selected_lag_time,
            n_components=selected_n_components,
        )
        tics = tica_model.fit_transform(features)
        tica_shapes = [list(x.shape) for x in tics]
        _save_intermediate(tics, run_dir / "tica_trajs")

        txx = np.concatenate(tics, axis=0)
        density_plot_path: Optional[Path] = None
        if txx.ndim == 2 and txx.shape[1] >= 2:
            density_plot_path = run_dir / "figs" / "tica_density_hexbin.png"
            plot_projection(
                txx[:, 0],
                txx[:, 1],
                outpath=density_plot_path,
                labels=["tIC 1", "tIC 2"],
            )
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage3_tica_fit",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage3_tica_fit",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
   
    manifest = {
        "stage3": {
            "tica_lag_time": selected_lag_time,
            "tica_n_components": selected_n_components,
            "tica_shapes": tica_shapes,
            "tica_traj_dir": str(run_dir / "tica_trajs"),
            "plot_path": str(density_plot_path) if density_plot_path else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")

    summary = build_stage3_summary(
        run_dir=run_dir,
        selected_lag_time=selected_lag_time,
        selected_n_components=selected_n_components,
        tica_shapes=tica_shapes,
        density_plot_path=density_plot_path,
    )

    return {
        "success": True,
        "stage": "stage3_tica_fit",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": [str(density_plot_path)] if density_plot_path else None,
    }


# ----------------------------
# Stage 4
# ----------------------------
def run_stage4_cluster(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    cfg = asdict(cfg)

    try:
        tics = load_processed_from_run_dir(run_dir, "tica_trajs")
        if tics is not None: # full pipeline
            dt_ns = read_json(run_dir / "manifest.json")["stage1"]["dt_ns_effective"]
        else: # tica do not exist, start from cv
            tics = load_processed_from_run_dir(cfg["clustering"]["cv_path"])
            assert tics is not None, "Starting from user provided CV for clustering, but provided cv path is invalid."
            dt_ns = cfg["clustering"]["dt_ns"]
            os.makedirs(run_dir / "figs", exist_ok=True)
        cl_cfg = cfg["clustering"]
        clusterer = find_clusterer(cl_cfg=cl_cfg)
        clustered_trajs = clusterer.fit_transform(tics)
        _save_intermediate(clustered_trajs, run_dir / "clustered_trajs")
        np.savetxt(run_dir / "clustered_trajs" / "cluster_centers.txt", clusterer.cluster_centers_)

        micro_assign = np.concatenate([np.asarray(t).reshape(-1) for t in clustered_trajs])
        occ_stats = compute_occupancy_stats(micro_assign, n_clusters=int(cl_cfg["n_clusters"]))
        plot_occupancy_hist(
            occ_stats["occupancies"],
            outpath=run_dir / "figs" / "occupancy_hist.png",
        )
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage4_cluster",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage4_cluster",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
    manifest = {
        "stage4": {
            "dt_ns": float(dt_ns),
            "cluster_traj_dir": str(run_dir / "cluster_trajs"),
            "cluster_centers_path": str(run_dir / "clustered_trajs" / "cluster_centers.txt"),
            "occupancy": {key: occ_stats[key] for key in occ_stats if key != "occupancies"},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")

    summary = build_stage4_summary(
        run_dir=run_dir,
        occupancy=occ_stats,
        cl_cfg=cl_cfg,
    )

    return {
        "success": True,
        "stage": "stage4_cluster",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": [str(run_dir / "figs" / "occupancy_hist.png")],
    }


# ----------------------------
# Stage 5
# ----------------------------
def run_stage5_msm_scan(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    cfg = asdict(cfg)

    try:
        if cfg["microMSM"]["micro_assign_path"] is not None:
            clustered_trajs = load_processed_from_run_dir(cfg["microMSM"]["micro_assign_path"])
            dt_ns = cfg["microMSM"]["dt_ns"]
            os.makedirs(run_dir / "figs", exist_ok=True)
        else:
            clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")
            dt_ns = read_json(run_dir / "manifest.json")["stage4"]["dt_ns"]
        all_state = np.unique(np.concatenate(clustered_trajs).reshape(-1))
        
        msm_cfg = cfg["microMSM"]
        lag_list = np.linspace(int(msm_cfg["lag_time_frames_range"][0]),int(msm_cfg["lag_time_frames_range"][1]),\
                               num=int(msm_cfg["lag_time_frames_grid_size"]), dtype=int)
        if msm_cfg["n_timescales"] is not None:
            n_timescales = int(msm_cfg["n_timescales"])
        else:
            n_timescales = int(cfg["clustering"]["n_clusters"])-1
            cfg["microMSM"]["n_timescales"] = n_timescales
            save_config(cfg, run_dir / "config.yaml")
        its = compute_msm_its(
            clustered_trajs=clustered_trajs,
            lag_list=lag_list,
            n_timescales=n_timescales,
            dt_ns=dt_ns,
            reversible_type=msm_cfg["reversible_type"],
            ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]),
        )
        plot_its_curve(
            its,
            outpath=run_dir / "figs" / "microstateMSM_its_curve.png",
        )

        # check msm quality
        sparsity = compute_transition_sparsity(clustered_trajs, n_states = len(all_state), lagtimes=lag_list.tolist()) 
        plateau_check = its_plateau_check(its, 
                                    lag_list=lag_list.tolist(),
                                    #top_k=cfg["evaluation"]["plateau_k"], 
                                    )
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage5_msm_scan",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage5_msm_scan",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }

    manifest = {
        "stage5": {
            "dt_ns": float(dt_ns),
            "sparsity": sparsity,
            "plateau_check": plateau_check,
            "plot_path": str(run_dir / "figs" / "microstateMSM_its_curve.png"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")

    summary = build_stage5_summary(
        run_dir=run_dir,
        sparsity=sparsity,
        plateau_check=plateau_check,
    )
    return {
        "success": True,
        "stage": "stage5_msm_scan",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": [str(run_dir / "figs" / "microstateMSM_its_curve.png")]
    }
    
def run_stage6_msm_fit(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = asdict(cfg)
    try:
        if cfg["microMSM"]["micro_assign_path"] is not None:
            clustered_trajs = load_processed_from_run_dir(cfg["microMSM"]["micro_assign_path"])
            dt_ns = cfg["microMSM"]["dt_ns"]
            os.makedirs(run_dir / "figs", exist_ok=True)
        else:
            clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")
            dt_ns = read_json(run_dir / "manifest.json")["stage5"]["dt_ns"]

        msm_cfg = cfg["microMSM"]
        selected_lag_time = msm_cfg["selected_lag_time"]
        if msm_cfg["selected_n_timescales"] is not None:
            selected_n_timescales = int(msm_cfg["selected_n_timescales"])
        else:
            selected_n_timescales = int(cfg["clustering"]["n_clusters"])-1
            cfg["microMSM"]["selected_n_timescales"] = selected_n_timescales
            save_config(cfg, run_dir / "config.yaml")
        
        if selected_lag_time is None:
            raise ValueError(
                "cfg.microMSM.selected_lag_time in frames is required for fitting a MSM. "
                "Please set it first, for example after reviewing Stage 5."
            )

        msm = MarkovStateModel(lag_time=int(selected_lag_time), n_timescales=int(selected_n_timescales), 
                               reversible_type=msm_cfg["reversible_type"], ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]))
        msm.fit(clustered_trajs)
        ck_results = ck_test(msm, clustered_trajs, num_states=Metric_param.ck_test_states, plot_dir=run_dir / "figs")
        all_plot_path = [str(run_dir / "figs" / "CK_test.png")]
        if cfg["data"]["physical_coord"] is not None:
            tics = load_processed_from_run_dir(cfg["data"]["physical_coord"])
            proj_label = ["Physical Coord 1", "Physical Coord 2"]
            cluster_centers = None
        else:
            tics = load_processed_from_run_dir(run_dir, "tica_trajs")
            proj_label = ["tIC 1", "tIC 2"]
            cluster_centers = np.loadtxt(read_json(run_dir / "manifest.json")["stage4"]["cluster_centers_path"])
        if tics is not None:
            txx = np.concatenate(tics, axis=0)
            weights = msm.populations_[np.concatenate(clustered_trajs)]
            plot_free_energy(txx[:,0], txx[:,1], weights, msm, run_dir / "figs" / "weighted_freeenergy.png", proj_label, centers=cluster_centers)
            all_plot_path.append(str(run_dir / "figs" / "weighted_freeenergy.png"))
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage6_msm_fit",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage6_msm_fit",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
    
    with open(run_dir / "microstateMSM_model.pkl", 'wb') as f:
        pickle.dump(msm, f)

    ts = np.asarray(dt_ns*msm.timescales_, dtype=float)
    manifest = {
        "stage6": {
            "dt_ns": float(dt_ns),
            "timescales_ns": ts.tolist(),
            "ck_test_results": ck_results,
            "microMSM_dir": str(run_dir / "microstateMSM_model.pkl"),
            "plot_path": all_plot_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    write_json(manifest, run_dir / "manifest.json")
    summary = build_stage6_summary(
        run_dir=run_dir,
        ck_test_results=ck_results,
        ts=ts,
    )
    return {
        "success": True,
        "stage": "stage6_msm_fit",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": all_plot_path,
    }

    
def run_stage7_lumpeval(cfg: AgentConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = asdict(cfg)
    try:
        msm_cfg = cfg["macroMSM"]
        if msm_cfg["macro_assign_path"] is not None:
            macro_trajs = load_processed_from_run_dir(msm_cfg["macro_assign_path"])
            dt_ns = msm_cfg["dt_ns"]
            lagtime = int(msm_cfg["lag_time"])
            msm_cfg["n_macrostates"] = len(np.unique(np.concatenate(macro_trajs, axis=0))) 
            cfg["macroMSM"]["n_macrostates"] = msm_cfg["n_macrostates"]
            save_config(cfg, run_dir / "config.yaml")
            os.makedirs(run_dir / "figs", exist_ok=True)
        else:
            stage6_manifest = read_json(run_dir / "manifest.json")["stage6"]
            with open(stage6_manifest["microMSM_dir"], 'rb') as f:
                microMSM = pickle.load(f)
            if cfg["microMSM"]["micro_assign_path"] is not None:
                clustered_trajs = load_processed_from_run_dir(cfg["microMSM"]["micro_assign_path"])
            else:
                clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")
            dt_ns = read_json(run_dir / "manifest.json")["stage6"]["dt_ns"]
            lagtime = int(cfg["microMSM"]["selected_lag_time"])
            msm_cfg["n_macrostates"] = int(msm_cfg["n_macrostates"]) if msm_cfg["n_macrostates"] else cfg["microMSM"]["selected_n_timescales"]+1
            cfg["macroMSM"]["n_macrostates"] = msm_cfg["n_macrostates"]
            save_config(cfg, run_dir / "config.yaml")
            lumper = getattr(lump_module, msm_cfg["lump_method"]).from_msm(microMSM, n_macrostates=msm_cfg["n_macrostates"])
            macro_trajs = lumper.transform(clustered_trajs)
            _save_intermediate(macro_trajs, run_dir / "macro_trajs")

        msm = MarkovStateModel(lag_time=lagtime,\
                                    reversible_type=msm_cfg["reversible_type"], ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]))
        msm.fit(macro_trajs)
        modes = msm.eigtransform(macro_trajs)
        modes = np.concatenate(modes, axis=0)
        occ_stat = compute_occupancy_stats(np.concatenate(macro_trajs).reshape(-1), n_clusters=int(msm_cfg["n_macrostates"]))
        lag_list = np.linspace(int(cfg["microMSM"]["lag_time_frames_range"][0]),int(cfg["microMSM"]["lag_time_frames_range"][1]),\
                                       num=int(cfg["microMSM"]["lag_time_frames_grid_size"]), dtype=int)
        its = compute_msm_its(
            clustered_trajs=macro_trajs,
            lag_list=lag_list,
            n_timescales=msm_cfg["n_macrostates"]-1,
            dt_ns=dt_ns,
            reversible_type=msm_cfg["reversible_type"],
            ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]),
            )
        plot_its_curve(
            its,
            outpath=run_dir / "figs" / "macrostateMSM_its_curve.png",
        )
        all_plot_path = [str(run_dir / "figs" / "macrostateMSM_its_curve.png")]
        if cfg["data"]["physical_coord"] is not None:
            tics = load_processed_from_run_dir(cfg["data"]["physical_coord"])
            labelxy = ["Physical Coord 1", "Physical Coord 2"]
        else:
            tics = load_processed_from_run_dir(run_dir, "tica_trajs")
            labelxy = ["tIC 1", "tIC 2"]
        if tics is not None:
            txx = np.concatenate(tics, axis=0)
            plot_projection(
                txx[:, 0],
                txx[:, 1],
                outpath=run_dir / "figs" / "macrostate_assignment.png",
                labels=labelxy + ["Macrostate Assignment"],
                z=np.concatenate(macro_trajs),
            )
            all_plot_path.append(str(run_dir / "figs" / "macrostate_assignment.png"))
            for i in range(modes.shape[1]): 
                plot_projection(
                    txx[:, 0],
                    txx[:, 1],
                    outpath=run_dir / "figs" / f"macrostate_kinetic_mode{i+1}.png",
                    labels=labelxy + [f"Kinetic mode {i+1}"],
                    cmap="coolwarm",
                    z=modes[:,i],
                )
                all_plot_path.append(str(run_dir / "figs" / f"macrostate_kinetic_mode{i+1}.png"))
        mfpt = tpt.mfpts(msm,lag_time=lagtime*dt_ns)
    except (ValueError, AssertionError) as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage7_lumpeval",
            "run_dir": str(run_dir),
            "errors": [{
                "type": "InputError",
                "message": str(e),
                "hint": "Invalid input arguments. The agent should modify parameters."
            }],
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "stage": "stage7_lumpeval",
            "run_dir": str(run_dir),
            "error": [{
                "type": "Exception",
                "message": str(e),
                "hint": "Error using tool. The agent should modify tool choice."
            }],
        }
    with open(run_dir / "macrostateMSM_model.pkl", 'wb') as f:
        pickle.dump(msm, f)

    ts = np.asarray(msm.timescales_, dtype=float)*dt_ns
    manifest = {
        "stage7": {
            "timescales_ns": ts.tolist(),
            "cooccupancy": {key: occ_stat[key] for key in occ_stat if key != "occupancies"},
            "macroMSM_dir": str(run_dir / "macrostateMSM_model.pkl"),
            "macro_traj_dir": str(run_dir / "macro_trajs"),
            "MFPT": mfpt,
            "plot_path": all_plot_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    write_json(manifest, run_dir / "manifest.json")
    summary = build_stage7_summary(
        run_dir=run_dir,
        macro_occupancy=occ_stat,
        mfpt=mfpt,
        ts=ts,
    )
    return {
        "success": True,
        "stage": "stage7_lumpeval",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": all_plot_path,
    }

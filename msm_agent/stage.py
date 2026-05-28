#msm_agent/msm_agent/stage.py
from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional
import glob
import json
import time
import pickle

import numpy as np
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel
import msmbuilder.lumping as lump_module

# ===== Adjust these imports to your real project layout =====
from msm_agent.featurizationv1 import (
    _make_run_dir,
    _load_feature,
    _save_intermediate,
    _find_clusterer,
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
    plot_tica_density_hexbin,
    plot_free_energy,
)


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
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"JSON file not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_processed_from_run_dir(run_dir: str | Path, data_type: str) -> List[np.ndarray]:
    assert data_type in ["features", "tica_trajs", "clustered_trajs"], f"Unknown data_type: {data_type}"
    run_dir = Path(run_dir)
    load_dir = run_dir / data_type
    if not load_dir.exists():
        raise ValueError(f"Directory not found: {load_dir}")

    files = sorted(glob.glob(str(load_dir / "*.npy")))
    if not files:
        raise ValueError(f"No files found in: {load_dir}")

    return [np.load(f) for f in files]


# ----------------------------
# Stage 1
# ----------------------------
def run_stage1_featurization(cfg: Dict[str, Any], run_dir: Path = None) -> Dict[str, Any]:
    if run_dir is None or not run_dir.exists():
        run_dir = _make_run_dir(cfg)

    try:
        # priority: processed, pair selection, atom selection
        features, dt_ps_effective = _load_feature(cfg, run_dir)
        traj_lens = [len(x) for x in features]
        feature_dims = [
            int(x.shape[1]) if getattr(x, "ndim", None) == 2 else None
            for x in features
        ]
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
    dist_cutoff = float(cfg["features"].get("distance_cutoff", 0.8))
    freq_cutoff = float(cfg["evaluation"].get("contact_freq_cutoff", 0.1))
    contact_test = in_contact_test(run_dir, dist_cutoff, freq_cutoff) # return dict or empty

    manifest = {
        "stage": "stage1_featurization",
        "run_dir": str(run_dir),
        "dt_ps_effective": float(dt_ps_effective),
        "dt_ns_effective": float(dt_ps_effective / 1000.0),
        "n_trajs": len(features),
        "traj_lens": traj_lens,
        "feature_dims": np.unique(feature_dims).tolist(),
        "feature_dir": str(run_dir / "features"),
        "in_contact_fraction": contact_test.get("in_contact_fraction", "Do not apply"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest, run_dir / "stage1_manifest.json")

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
        "manifest_path": str(run_dir / "stage1_manifest.json"),
    }


# ----------------------------
# Stage 2
# ----------------------------
def run_stage2_tica_scan(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    
    try:
        stage1_manifest = read_json(run_dir / "stage1_manifest.json")
        features = load_processed_from_run_dir(run_dir, "features")

        dt_ps_effective = float(stage1_manifest["dt_ps_effective"])
        dt_ns = dt_ps_effective / 1000.0

        tica_cfg = cfg["tica"]
        lag_min = int(tica_cfg["lag_time_frames_range"][0])
        lag_max = int(tica_cfg["lag_time_frames_range"][1])
        grid_size = int(tica_cfg.get("lag_time_frames_grid_size", 20))
        n_components = int(tica_cfg["n_components"])

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
    tica_param = {k: tica_cfg[k] for k in tica_cfg}
    manifest = {
        "stage": "stage2_tica_param_scan",
        "run_dir": str(run_dir),
        "dt_ns": float(dt_ns),
        "plot_path": str(plot_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **tica_param
    }
    write_json(manifest, run_dir / "stage2_manifest.json")
    
    plateau_check = its_plateau_check(tica_its, lag_list.tolist())

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
        "manifest_path": str(run_dir / "stage2_manifest.json"),
    }


# ----------------------------
# Stage 3
# ----------------------------
def run_stage3_tica_fit(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")
    
    try:
        features = load_processed_from_run_dir(run_dir, "features")

        tica_cfg = cfg["tica"]
        selected_lag_time = tica_cfg.get("selected_lag_time", None)
        selected_n_components = tica_cfg.get("selected_n_components", tica_cfg["n_components"])

        if selected_lag_time is None:
            raise ValueError(
                "cfg['tica']['selected_lag_time'] in frames is required for Stage 3. "
                "Please set it first, for example after reviewing Stage 2."
            )

        selected_lag_time = int(selected_lag_time)
        selected_n_components = int(selected_n_components)

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
            plot_tica_density_hexbin(
                txx[:, 0],
                txx[:, 1],
                outpath=density_plot_path,
                gridsize=int(cfg["plots"]["gridsize"]),
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
    tica_param = {k: tica_cfg[k] for k in tica_cfg}
    manifest = {
        "stage": "stage3_tica_fit",
        "run_dir": str(run_dir),
        "tica_shapes": tica_shapes,
        "tica_traj_dir": str(run_dir / "tica_trajs"),
        "plot_path": str(density_plot_path) if density_plot_path else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **tica_param
    }
    write_json(manifest, run_dir / "stage3_manifest.json")

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
        "manifest_path": str(run_dir / "stage3_manifest.json"),
    }


# ----------------------------
# Stage 4
# ----------------------------
def run_stage4_cluster(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")

    try:
        tics = load_processed_from_run_dir(run_dir, "tica_trajs")
        cl_cfg = cfg["clustering"]
        clusterer = _find_clusterer(random_state=int(cfg["run"]["seed"]), cl_cfg=cl_cfg)
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
        "stage": "stage4_cluster",
        "run_dir": str(run_dir),
        "cluster_traj_dir": str(run_dir / "cluster_trajs"),
        "cluster_centers_path": str(run_dir / "clustered_trajs" / "cluster_centers.txt"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": cl_cfg["method"],
        **{key: occ_stats[key] for key in occ_stats if key != "occupancies"},
    }
    write_json(manifest, run_dir / "stage4_manifest.json")

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
        "manifest_path": str(run_dir / "stage4_manifest.json"),
    }


# ----------------------------
# Stage 5
# ----------------------------
def run_stage5_msm_scan(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")

    try:
        clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")
        dt_ns = read_json(run_dir / "stage1_manifest.json")["dt_ns_effective"]
        all_state = np.unique(np.concatenate(clustered_trajs).reshape(-1))
        
        msm_cfg = cfg["microMSM"]
        lag_list = np.linspace(int(msm_cfg["lag_time_frames_range"][0]),int(msm_cfg["lag_time_frames_range"][1]),\
                               num=int(msm_cfg["lag_time_frames_grid_size"]), dtype=int)
        its = compute_msm_its(
            clustered_trajs=clustered_trajs,
            lag_list=lag_list,
            n_timescales=int(msm_cfg["n_timescales"]),
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

    msm_param = {k: msm_cfg[k] for k in msm_cfg}
    manifest = {
        "stage": "stage5_msm_scan",
        "run_dir": str(run_dir),
        "plot_path": str(run_dir / "figs" / "microstateMSM_its_curve.png"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **msm_param,
    }
    write_json(manifest, run_dir / "stage5_manifest.json")


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
        "plot_path": [str(run_dir / "figs" / "microstateMSM_its_curve.png")],
        "manifest_path": str(run_dir / "stage5_manifest.json"),
    }
    
def run_stage6_msm_fit(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    try:
        clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")
        cluster_centers = np.loadtxt(read_json(run_dir / "stage4_manifest.json")["cluster_centers_path"])
        dt_ns = read_json(run_dir / "stage1_manifest.json")["dt_ns_effective"]
        
        msm_cfg = cfg["microMSM"]
        selected_lag_time = msm_cfg.get("selected_lag_time", None)
        selected_n_timescales = msm_cfg.get("selected_n_timescales", msm_cfg["n_timescales"])

        if selected_lag_time is None:
            raise ValueError(
                "cfg['microMSM']['selected_lag_time'] in frames is required for fitting a MSM. "
                "Please set it first, for example after reviewing Stage 5."
            )

        msm = MarkovStateModel(lag_time=int(selected_lag_time), n_timescales=int(selected_n_timescales), 
                               reversible_type=msm_cfg.get("reversible_type",'transpose'), ergodic_cutoff=float(msm_cfg.get("ergodic_cutoff", 0.0)))
        msm.fit(clustered_trajs)
        ck_results = ck_test(msm, clustered_trajs, num_states=int(cfg['evaluation']['ck_test_states']), plot_dir=run_dir / "figs")
        tics = load_processed_from_run_dir(run_dir, "tica_trajs")
        txx = np.concatenate(tics, axis=0)
        weights = msm.populations_[np.concatenate(clustered_trajs)]
        plot_free_energy(txx[:,0], txx[:,1], weights, msm, run_dir / "figs" / "weighted_freeenergy.png", centers=cluster_centers)
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
    msm_param = {k: msm_cfg[k] for k in msm_cfg}
    manifest = {
        "stage": "stage6_msm_fit",
        "run_dir": str(run_dir),
        "microMSM_dir": str(run_dir / "microstateMSM_model.pkl"),
        "ck_plot_path": str(run_dir / "figs" / "CK_test.png"),
        "free_energy_plot_path": str(run_dir / "figs" / "weighted_freeenergy.png"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timescales_ns": ts.tolist(),
        "ck_test_results": ck_results,
        "dt_ns_effective": dt_ns,
        **msm_param,
    }
    write_json(manifest, run_dir / "stage6_manifest.json")
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
        "plot_path": [str(run_dir / "figs" / "CK_test.png"), str(run_dir / "figs" / "weighted_freeenergy.png")],
        "manifest_path": str(run_dir / "stage6_manifest.json"),
    }

    
def run_stage7_lumpeval(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    try:
        stage6_manifest = read_json(run_dir / "stage6_manifest.json")
        with open(stage6_manifest["microMSM_dir"], 'rb') as f:
            microMSM = pickle.load(f)
        clustered_trajs = load_processed_from_run_dir(run_dir, "clustered_trajs")

        msm_cfg = cfg["macroMSM"]
        lumper = getattr(lump_module, msm_cfg["lump_method"]).from_msm(microMSM, n_macrostates=msm_cfg["n_macrostates"])
        macro_trajs = lumper.transform(clustered_trajs)
        _save_intermediate(macro_trajs, run_dir / "macro_trajs")

        msm = MarkovStateModel(lag_time=int(cfg["microMSM"]["selected_lag_time"]),\
                                    reversible_type=msm_cfg.get("reversible_type",'mle'), ergodic_cutoff=float(msm_cfg.get("ergodic_cutoff", 0.0)))
        msm.fit(macro_trajs)
        occ_stat = compute_occupancy_stats(np.concatenate(macro_trajs).reshape(-1), n_clusters=int(msm_cfg["n_macrostates"]))
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
    
    msm_param = {k: msm_cfg[k] for k in msm_cfg}
    ts = np.asarray(msm.timescales_, dtype=float)*stage6_manifest["dt_ns_effective"]
    manifest = {
        "stage": "stage7_lumpeval",
        "run_dir": str(run_dir),
        "macroMSM_dir": str(run_dir / "macrostateMSM_model.pkl"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timescales_ns": ts.tolist(),
        **msm_param,
        **occ_stat,
    }
    write_json(manifest, run_dir / "stage7_manifest.json")
   
    return {
        "success": True,
        "stage": "stage7_lumpeval",
        "run_dir": str(run_dir),
        "summary": build_stage7_summary(run_dir, occ_stat, ts),
        "plot_path": None,
        "manifest_path": None,
    }

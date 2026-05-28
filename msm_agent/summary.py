from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

def build_stage1_summary(
    cfg: Dict[str, Any],
    run_dir: Path,
    n_trajs: int,
    traj_lens: List[int],
    feature_dims: List[Optional[int]],
    contact_test:Dict[str, Any],
    dt_ps_effective: float,
) -> str:
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})

    lines = [
        "Stage 1 completed: data loading and featurization finished.",
        f"Run dir: {run_dir}",
        f"Data kind: {data_cfg.get('kind', 'NA')}",
        f"Number of trajectories: {n_trajs}",
        f"Effective timestep: {dt_ps_effective:.4f} ps",
        f"Feature type: {feat_cfg.get('type', 'NA')}",
    ]
    if feat_cfg.get('selection'):
        lines.append(f"Feature selection: {feat_cfg.get('selection')}")
        lines.append(f"Atom selection: {feat_cfg.get('atom_selection')}")
    if feat_cfg.get('pair_selection'):
        lines.append(f"Pair selection: {feat_cfg.get('pair_selection')}")
    if traj_lens:
        lines.append(
            f"Trajectory length range (frames): min={min(traj_lens)}, max={max(traj_lens)}"
        )
    uniq_dims = np.unique(feature_dims)
    lines.append(f"Feature dimension(s): {uniq_dims}")
    if len(uniq_dims) > 1:
        lines.append(f"Warning: Feature dimensions are not consistent, have {uniq_dims}")
    elif uniq_dims[0] < 5:
        lines.append("Warning: Feature dimension is already low, skip stage 2 or use a different set of features.")
    if contact_test:
        lines.append(f"Fraction of pairs in contact: {contact_test.get('in_contact_fraction', 'NA')}")
        if contact_test.get("in_contact_fraction", 0) < 0.1:
            lines.append("Warning: Low fraction of frames with contacts, may be due to a small distance cutoff or a large set of distances.")
            lines.append("Warning: First consider increasing the distance cutoff")
            lines.append(f"Warning: Or set features.pair_selection to '{run_dir} / in_contact_pairs_{contact_test['dist_cutoff']}_{contact_test['contact_freq_cutoff']}.npy'.")
    
    lines += [
        f"Saved features: {run_dir / 'features'}",
        "",
        "Please review featurization results. Address all warning messages before proceeding.",
    ]
    return "\n".join(lines)


def build_stage2_summary(
    cfg: Dict[str, Any],
    run_dir: Path,
    lag_list: List[int],
    dt_ns: float,
    plot_path: Path,
    plateau_check: Dict[str, Any],
) -> str:
    tica_cfg = cfg.get("tica", {})
    lines = [
        "Stage 2 completed: tICA parameter scan finished.",
        f"Run dir: {run_dir}",
        f"Scan lag range (in frames): {lag_list[0]} to {lag_list[-1]}",
        f"Scan n_components: {tica_cfg.get('n_components', 'NA')}",
        f"Effective timestep: {dt_ns:.6f} ns",
        f"ITS plot: {plot_path}",
        f"Auto plateau check: for top {plateau_check['top_k']} components, the last {plateau_check['last_step']} steps are {plateau_check['plateaued']}",
    ]
    if not all(plateau_check["plateaued"]):
        lines += [
            "Warning: ITS not plateaued for some components. This indicates that the lag times in the scan maybe too short to be markovian.",
            "Warning: Try increasing the lag time range and rerun stage 2"
        ]
    else: 
        lines += [
            f"Recommendation: ITS plateaued for selected components, recommended selected_lag_time for proceed: {plateau_check['min_lag']}.",
            f"Recommendation: recommended selected_n_component for proceed: {plateau_check['separated_component']}"
        ]
    lines += [
        "",
        "Please review the ITS curve for tICA parameter scan. Address all warning messages before proceeding",
    ]
    return "\n".join(lines)


def build_stage3_summary(
    run_dir: Path,
    selected_lag_time: int,
    selected_n_components: int,
    tica_shapes: List[List[int]],
    density_plot_path: Optional[Path],
) -> str:
    lines = [
        "Stage 3 completed: final tICA fit finished.",
        f"Run dir: {run_dir}",
        f"Selected lag_time (frames): {selected_lag_time}",
        f"Selected n_components: {selected_n_components}",
        f"Saved tICA trajectories: {run_dir / 'tica_trajs'}",
        f"tICA trajectory shapes: {tica_shapes}",
    ]
    if density_plot_path is not None:
        lines.append(f"Density plot: {density_plot_path}")

    lines += [
        "",
        "Please review the final tICA embeddings. Address all warning messages before proceeding",
    ]
    return "\n".join(lines)


def build_stage4_summary(
    run_dir: Path,
    occupancy: Dict[str, Any],
    cl_cfg: Dict[str, Any],
) -> str:
    lines = [
        "Stage 4 completed: clustering finished.",
        f"Run dir: {run_dir}",
        f"Cluster type: {cl_cfg['method']}",
        f"Cluster n_clusters: {cl_cfg['n_clusters']}",
        f"Saved clustered trajectories: {run_dir / 'cluster_trajs'}",
        f"Occupied clusters: {occupancy['n_used']} out of {occupancy['n_clusters']} total clusters",
        f"Tiny clusters (occupancy < {cl_cfg.get('tiny_threshold', 10)}): {occupancy['tiny_frac']:.4f} fraction",
    ]
    if occupancy['tiny_flag']:
        lines += [
            "Warning: A large fraction of clusters are tiny, which may indicate that the clustering is too fine-grained.",
            "Warning: Consider reducing n_clusters or adjusting clustering parameters.",
        ]
    lines += [
        "",
        "Please review the clustering results. Address all warning messages before proceeding",
    ]
    return "\n".join(lines)

def build_stage5_summary(
    run_dir: Path,
    sparsity: List[Dict[str, Any]],
    plateau_check: Dict[str, Any],
) -> str:
    lines = [
        "Stage 5 completed: MSM scanning finished.",
        f"Run dir: {run_dir}",
    ]

    for s in sparsity:
        if s["disconnected"] > 0:
            lines.append(f"Warning: {s['disconnected']} disconnected states found at lag {s['lagtime']} frames. \
                         Try decreasing the MSM lag time or reducing the number of clusters")
   
    if not all(plateau_check["plateaued"]):
        lines += [
            "Warning: ITS not plateaued for some components. This indicates that the lag times in the scan maybe too short to be markovian.",
            "Warning: Try increasing the lag time range and rerun stage 5"
        ]
    else: 
        lines += [
            f"Recommendation: ITS plateaued for selected components, recommended selected_lag_time for proceed: {plateau_check['min_lag']}.",
            f"Recommendation: recommended selected_n_component for proceed: {plateau_check['separated_component']}"
        ]
    lines += [
        "",
        "Please review the MSM quality metrics. Address all warning messages before proceeding.",
    ]
    return "\n".join(lines)

def build_stage6_summary(
    run_dir: Path,
    ck_test_results: Dict[str, Any],
    ts: List[float],
) -> str:
    lines = [
        "Stage 6 completed: microstateMSM fit and quality test finished.",
        f"Run dir: {run_dir}",
        f"Captured timescales (ns): {ts}",
        f"CK test pass: {ck_test_results['pass']} with note {ck_test_results['note']}",
    ]
    if not ck_test_results["pass"]:
        lines += [
            "Warning: CK test failed, which may indicate that the MSM does not capture the kinetics well.",
            "Warning: Consider adjusting MSM parameters (e.g. lag time, n_timescales) and rerunning Stage 5 and Stage 6.",
        ]
    lines += [
        "",
        "Please review the MSM test results. Address all warning messages before proceeding.",
        "Note that the estimated timescales will be slower after lumping.",
    ]
    return "\n".join(lines)

def build_stage7_summary(
    run_dir: Path,
    macro_occupancy: Dict[str, Any],
    ts: List[float],
) -> str:
    lines = [
        "Stage 7 completed: macrostate analysis finished.",
        f"Run dir: {run_dir}",
        f"Number of macrostates: {macro_occupancy['n_clusters']}",
        f"Macrostate populations: {macro_occupancy['occupancies']}",
        f"Captured timescales (ns): {ts}",
    ]
    if macro_occupancy['tiny_frac'] > 0.2:
        lines += [
            "Warning: A large fraction of macrostates are tiny, which may indicate that the lumping is too fine-grained.",
            "Warning: Consider reducing n_macrostates or adjusting lumping parameters.",
        ]
    lines += [
            "",
            "Please review the macrostate analysis results. Address all warning messages before proceeding.",
    ]
    return "\n".join(lines)
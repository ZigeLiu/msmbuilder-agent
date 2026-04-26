from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

def build_stage1_summary(
    cfg: Dict[str, Any],
    run_dir: Path,
    n_trajs: int,
    traj_lens: List[int],
    feature_dims: List[Optional[int]],
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
        f"Feature selection: {feat_cfg.get('selection', 'NA')}",
        f"Atom selection: {feat_cfg.get('atom_selection', 'NA')}",
    ]
    if traj_lens:
        lines.append(
            f"Trajectory length range (frames): min={min(traj_lens)}, max={max(traj_lens)}"
        )
    uniq_dims = sorted({d for d in feature_dims if d is not None})
    if uniq_dims < 5:
        lines.append(f"Feature dimension(s): {uniq_dims}")
        lines.append("Warning: Feature dimension is already low, skip stage 2 or use a different set of features.")

    lines += [
        f"Saved features: {run_dir / 'features'}",
        "",
        "Please review whether this featurization is acceptable. You can keep it or modify feature-related config and rerun Stage 1.",
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
        "Stage 2 completed: tICA lag scan finished.",
        f"Run dir: {run_dir}",
        f"Scan lag range (in frames): {lag_list[0]} to {lag_list[-1]}",
        f"Scan n_components: {tica_cfg.get('n_components', 'NA')}",
        f"Effective timestep: {dt_ns:.6f} ns",
        f"ITS plot: {plot_path}",
        f"Auto plateau check: for top {plateau_check['top_k']} components, the last {plateau_check['last_step']} steps are {plateau_check['plateaued']}",
    ]
    if not all(plateau_check["plateaued"]):
        lines += [
            "Warning: ITS not plateaued for some components. This may indicate that the lag times in the scan are too short to capture the slow dynamics.",
        ]
    lines += [
        "",
        "Please review the ITS curve. ",
        "Set selected_lag_time based on the minimal lag time that produces a plateau.",
        "Set selected_n_components based on the largest timescale separation."
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
        "Please review the final tICA embedding.",
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
    if occupancy['tiny_frac'] > 0.2:
        lines += [
            "Warning: A large fraction of clusters are tiny, which may indicate that the clustering is too fine-grained.",
            "Consider reducing n_clusters or adjusting clustering parameters.",
        ]
    lines += [
        "",
        "Please review the clustering results. You can keep the current clustering settings or modify clustering-related config and rerun Stage 4.",
    ]
    return "\n".join(lines)

def build_stage5_summary(
    run_dir: Path,
    sparsity: List[Dict[str, Any]],
    its_plateau: Dict[str, Any],
) -> str:
    lines = [
        "Stage 5 completed: MSM scanning finished.",
        f"Run dir: {run_dir}",
    ]

    for s in sparsity:
        if s["disconnected"] > 0:
            lines.append(f"Sparsity warning: {s['disconnected']} disconnected states found at lag {s['lagtime']} frames. \
                         Try decreasing the MSM lag time or reducing the number of clusters")
    for i, p in enumerate(its_plateau["plateaued"]):
        if not p:
            lines.append(f"ITS warning: ITS not plateaued for {i+1} timescales, consider increasing the MSM lag time.")

    lines += [
        "",
        "Please review the MSM quality metrics. Please adjust parameters as needed and rerun Stage 5.",
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
            "Consider adjusting MSM parameters (e.g. lag time, n_timescales) and rerunning Stage 5 and Stage 6.",
        ]
    lines += [
        "",
        "Please review the MSM test results. Note that the estimated timescales will be slower after lumping.",
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
            "Consider reducing n_macrostates or adjusting lumping parameters.",
        ]
    lines += [
            "",
            "Please review the macrostate analysis results. You can adjust lumping parameters and rerun Stage 7 if needed.",
    ]
    return "\n".join(lines)
from __future__ import annotations
import numpy as np
from pathlib import Path
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel
from msm_agent.ck_test import remaining_probability_from_model, remaining_probability_from_data, \
    get_data_standard_error, get_model_standard_error, evaluate_ck_pass, plot_ck_test
from parameters import Metric_param

def compute_occupancy_stats(assign_1d: np.ndarray, n_clusters: int) -> dict:
    assign_1d = np.asarray(assign_1d).reshape(-1)
    occ = np.bincount(assign_1d, minlength=n_clusters)
    tiny_threshold = Metric_param.occupancy_tiny_threshold
    tiny_frac_threshold = Metric_param.occupancy_tiny_frac_threshold
    tiny_frac = float(np.mean(occ < tiny_threshold))
    return {
        "n_clusters": int(n_clusters),
        "n_used": int(np.sum(occ > 0)),
        "tiny_frac": float(tiny_frac),
        "tiny_flag": bool(tiny_frac > tiny_frac_threshold),
        "occupancies": occ.tolist(),
    }

def compute_transition_sparsity(clustered_trajs, n_states: int, lagtimes: list) -> list[dict]:
    sparsity = []
    for lagtime in lagtimes:
        out_count = {}
        for traj in clustered_trajs:
            a = np.asarray(traj).reshape(-1)
            for i in range(len(a) - lagtime):
                s = int(a[i])
                t = int(a[i+lagtime])
                if s != t:
                    if s not in out_count.keys():
                        out_count[s] = 1
                    else:
                        out_count[s] += 1
        out_count_list = list(out_count.values())
        sparsity.append({
            "lagtime": int(lagtime),
            "disconnected": n_states - int(len(out_count_list)),
            "avg_out_degree": float(np.mean(out_count_list)) if len(out_count_list) > 0 else 0.0,
        })
    return sparsity

def compute_tica_its(features, lag_list, n_components: int, dt_ns: float) -> dict:
    table = {}
    for lag in lag_list:
        tica = tICA(lag_time=lag, n_components=int(n_components))
        tica.fit(features)
        ts = np.asarray(tica.timescales_, dtype=float) # [1, n_components]
        table[str(lag*dt_ns)] = (ts * dt_ns).tolist()
    return table 

def compute_msm_its(clustered_trajs, lag_list, n_timescales: int, dt_ns: float, reversible_type=Metric_param.msm_reversible_type, ergodic_cutoff=Metric_param.msm_ergodic_cutoff) -> dict:
    table = {}
    for lag in lag_list:
        msm = MarkovStateModel(lag_time=int(lag), n_timescales=int(n_timescales),\
                                    reversible_type=reversible_type, ergodic_cutoff=ergodic_cutoff)
        msm.fit(clustered_trajs)
        ts = np.asarray(msm.timescales_, dtype=float)
        assert len(ts) >= n_timescales, f"MSM at lag {lag} has only {len(ts)} timescales, less than n_timescales={n_timescales}, try decreasing n_timescales or lagtime."
        table[str(lag*dt_ns)] = (ts * dt_ns).tolist()
    return table

def its_plateau_check(its: dict, lag_list: list, top_k=Metric_param.plateau_top_k, \
                      threshold=Metric_param.plateau_threshold, last_step=Metric_param.plateau_last_steps, separate_cutoff=Metric_param.plateau_separate_cutoff) -> dict:
    top_k = -1 if top_k is None else top_k
    timescales = []
    for key, val in its.items():
        if len(val) < top_k:
            raise ValueError(f"ITS for lag {key} has only {len(val)} timescales, less than top_k={top_k}")
        timescales.append(np.asarray(val[:top_k], dtype=float)) # [num of lag, top k]
    lagstep = np.array(lag_list, dtype=int) # [num of lag]
    timescales = np.array(timescales)
    # plateau check
    assert len(lagstep) > last_step, f"Need at least {last_step+1} lag times for plateau check, but got {len(lagstep)}"
    d_lag = np.diff(lagstep) 
    d_ts = np.diff(timescales, axis=0) # [num of lag - 1, top k]
    rel_d_ts = np.abs(d_ts / (d_lag.reshape(-1, 1) + 1e-12)) # [num of lag - 1, top k]
    all_plateaued = rel_d_ts < threshold
    rev_all_plateaued = ~all_plateaued
    min_lag = np.max(rev_all_plateaued.sum(1))+1
    plateaued = all_plateaued[-last_step:, :].all(axis=0) # make sure the time range is long for plateaue

    # TODO: find largest timescale separation
    timescale_separation = timescales[:,1:] / (timescales[:,:-1] + 1e-12) # [num of lag, top k - 1]
    timescale_separated_inv = timescale_separation > separate_cutoff # bool of [num of lag, top k -1]
    separated_component = np.where(timescale_separated_inv.sum(0) > 0)[0][0] # the first non zero in [top k - 1] 

    return {
        "top_k": int(top_k),    
        "min_lag": int(lag_list[min_lag]),
        "separated_component": int(separated_component),
        "last_step": int(last_step),
        "plateaued": plateaued,
    }

def ck_test(mdl, clustered_trajs, num_states: int, plot_dir: Path, \
            plot_only=Metric_param.ck_plot_only, n_steps=Metric_param.ck_n_steps, \
            block_percentage=Metric_param.ck_block_percentage, n_samples=Metric_param.ck_n_bootstrap, threshold=Metric_param.ck_pass_threshold) -> dict:              
    pop_sort = sorted(range(len(mdl.populations_)), key=lambda k: mdl.populations_[k])
    prob_model = remaining_probability_from_model(num_states, n_steps, len(mdl.state_labels_), mdl.transmat_, pop_sort)
    state_flag, prob_data = remaining_probability_from_data(num_states, n_steps, mdl.lag_time, pop_sort, clustered_trajs)
    data_se = get_data_standard_error(num_states, n_steps, state_flag, block_percentage=block_percentage)
    plot_ck_test(num_states, n_steps, prob_data, data_se, prob_model , plot_dir / "CK_test.png")
    if not plot_only:
        model_se = get_model_standard_error(num_states, n_steps, clustered_trajs, mdl, n_samples=n_samples)
        eval_results = evaluate_ck_pass(prob_model, prob_data, model_se, data_se, threshold=threshold)

    return eval_results if not plot_only else {"plot_only": True}

def in_contact_test(run_dir: Path, dist_cutoff: float, freq_cutoff: float) -> dict:
    contact_path = run_dir / f"contact_freq_{dist_cutoff}.npz"
    if not contact_path.exists():
        return {}
    else:
        contact = np.load(contact_path, allow_pickle=True)
        in_contact = contact['contact_freq'] > freq_cutoff
        contact_pairs = contact['pairs'][in_contact]
        np.save(run_dir / f"in_contact_pairs_{dist_cutoff}_{freq_cutoff}.npy", contact_pairs)
        return {
            "dist_cutoff": float(dist_cutoff),
            "contact_freq_cutoff": float(freq_cutoff),
            "in_contact_fraction": float(np.mean(in_contact)),
        }
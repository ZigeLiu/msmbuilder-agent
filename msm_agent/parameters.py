from dataclasses import dataclass

@dataclass(frozen=False)
class Metric_param:
    occupancy_tiny_threshold: int = 10
    occupancy_tiny_frac_threshold: float = 0.2
    msm_reversible_type = 'mle'
    msm_ergodic_cutoff = False
    plateau_threshold: float = 0.1
    plateau_last_steps: int = 4
    plateau_separate_cutoff: float = 0.9
    plateau_top_k: int = 4
    ck_plot_only: bool = False
    ck_n_steps: int = 4
    ck_block_percentage: float = 0.1
    ck_n_bootstrap: int = 20
    ck_pass_threshold: float = 1.96
    conact_freq_threshold: float = 0.1

@dataclass(frozen=False)
class Auto_set:
    max_try: int = 100
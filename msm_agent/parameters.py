from dataclasses import dataclass
from matplotlib import rcParams

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
    ck_test_states: int = 6
    ck_block_percentage: float = 0.1
    ck_n_bootstrap: int = 20
    ck_pass_threshold: float = 1.96
    conact_freq_threshold: float = 0.1
    distance_cutoff: float = 0.8

@dataclass(frozen=False)
class Auto_set:
    max_try: int = 100

@dataclass
class Plot_param:
    # Figure
    figsize: tuple[float, float] = (4.5, 3.5)
    dpi: int = 120
    save_dpi: int = 300

    # Fonts
    font_family: str = "sans-serif"
    font_size: int = 11
    title_size: int = 13
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10

    # Axes
    axes_linewidth: float = 1.2
    show_top_spine: bool = False
    show_right_spine: bool = False
    grid: bool = False

    # Ticks
    tick_direction: str = "in"
    major_tick_size: float = 5
    minor_tick_size: float = 3
    major_tick_width: float = 1.0
    minor_tick_width: float = 0.8
    show_minor_ticks: bool = True

    # Lines
    linewidth: float = 2.0
    markersize: float = 6

    # Legend
    legend_frame: bool = False

    # Error bars
    capsize: float = 3

    def apply(self):
        rcParams.update({
            # Figure
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.save_dpi,
            "savefig.bbox": "tight",

            # Fonts
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,

            # Axes
            "axes.linewidth": self.axes_linewidth,
            "axes.spines.top": self.show_top_spine,
            "axes.spines.right": self.show_right_spine,
            "axes.grid": self.grid,

            # Ticks
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "xtick.major.size": self.major_tick_size,
            "ytick.major.size": self.major_tick_size,
            "xtick.minor.size": self.minor_tick_size,
            "ytick.minor.size": self.minor_tick_size,
            "xtick.major.width": self.major_tick_width,
            "ytick.major.width": self.major_tick_width,
            "xtick.minor.width": self.minor_tick_width,
            "ytick.minor.width": self.minor_tick_width,
            "xtick.minor.visible": self.show_minor_ticks,
            "ytick.minor.visible": self.show_minor_ticks,

            # Lines
            "lines.linewidth": self.linewidth,
            "lines.markersize": self.markersize,

            # Legend
            "legend.frameon": self.legend_frame,

            # Error bars
            "errorbar.capsize": self.capsize,

            # PDF
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
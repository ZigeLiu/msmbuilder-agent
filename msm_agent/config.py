from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from datetime import datetime
import yaml

SECTION_NAMES = ("data", "features", "tica", "clustering", "microMSM", "macroMSM")


@dataclass
class DataConfig:
    kind: str = "xtc"
    dir: str = "path/to/your/xtc/files"
    topology: str = "path/to/your/topology/file.pdb"
    saving_interval: int = 1
    stride: int = 1
    load_preprocessed_dir: str = None


@dataclass
class FeaturesConfig:
    type: str = "distance"
    selection: str = "distances"
    atom_selection: str = "HEAVY"
    pair_selection: Path | None = None


@dataclass
class TICAConfig:
    lag_time_frames_range: list = field(default_factory=lambda: [1, 50])
    n_components: int = 4
    lag_time_frames_grid_size: int = 20
    selected_lag_time: int = None
    selected_n_components: int = 4

@dataclass
class ClusterConfig:
    method: str = "KMeans"
    n_clusters: int = 200
    random_seed: int = 42


@dataclass
class microMSMConfig:
    lag_time_frames_range: list = field(default_factory=lambda: [1, 50])
    lag_time_frames_grid_size: int = 20
    n_timescales: int = 5
    reversible_type: str = "transpose"
    ergodic_cutoff: bool = False
    selected_lag_time: int = None
    selected_n_timescales: int = None


@dataclass
class macroMSMConfig:
    n_macrostates: int = 5
    lump_method: str = "PCCAPlus"
    reversible_type: str = "mle"
    ergodic_cutoff: bool = False


@dataclass
class AgentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    tica: TICAConfig = field(default_factory=TICAConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    microMSM: microMSMConfig = field(default_factory=microMSMConfig)
    macroMSM: macroMSMConfig = field(default_factory=macroMSMConfig)
    run_dir:  Path = field(
    default_factory=lambda: Path(f"results/{datetime.now():%Y%m%d_%H%M%S}")
)


@dataclass
class ConfigState:
    config: AgentConfig
    touched_sections: set[str] = field(default_factory=set)


def from_dict(cls, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        value = data.get(f.name, None)
        if is_dataclass(f.type):
            kwargs[f.name] = from_dict(f.type, value or {})
        elif value is not None:
            kwargs[f.name] = value
    return cls(**kwargs)

def field_names(obj, prefix=""):
    names = []
    for f in fields(obj):
        value = getattr(obj, f.name)
        name = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(value):
            names.extend(field_names(value, name))
        else:
            names.append(name)
    return names

def dump_config_yaml(state: ConfigState) -> str:
    data = serialize_config_subset(state)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

def load_yaml_config_state(text: str, touched_sections: set[str] | None = None) -> ConfigState:
    if isinstance(text, Path):
        text = text.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return ConfigState(config=from_dict(AgentConfig, data), touched_sections=touched_sections)

def serialize_config_subset(state: ConfigState) -> dict[str, Any]:
    full_data = asdict(state.config)
    data = {"run_dir": str(full_data["run_dir"])}
    for section_name in SECTION_NAMES:
        if section_name in state.touched_sections:
            data[section_name] = full_data[section_name]
    return data

def save_config(state, path):
    if isinstance(state, ConfigState):
        #data = serialize_config_subset(state)
        data = asdict(state.config)
        data["run_dir"] = str(data["run_dir"])
    elif isinstance(state, AgentConfig):
        data = asdict(state)
        data["run_dir"] = str(data["run_dir"])
    elif isinstance(state, dict):
        data = state
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    else:
        path.write_text(json.dumps(data, indent=4))

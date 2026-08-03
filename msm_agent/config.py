from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints
import json

import numpy as np
from datetime import datetime
import yaml

SECTION_NAMES = ("data", "features", "tica", "clustering", "microMSM", "macroMSM")
type_map = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "None": type(None),
}

@dataclass
class ConfigBase:
    def __post_init__(self):
        self._validate()

    def _validate(self):
        for f in fields(self):
            choices = f.metadata.get("choice")
            if choices is None:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, list):
                invalid_values = [v for v in value if v not in choices]
                if invalid_values:
                    raise ValueError(
                        f"{f.name} contains unsupported values {invalid_values!r}; "
                        f"expected values from {choices}"
                    )
            elif value not in choices:
                raise ValueError(
                    f"{f.name}={value!r}; expected one of {choices}"
                )

@dataclass
class DataConfig(ConfigBase):
    kind: str = field(default="xtc", metadata={"choice": ["xtc", "dcd", "trr"]})
    dir: str = "path/to/your/xtc/files"
    topology: str = "path/to/your/topology/file.pdb"
    saving_interval: int = 1
    stride: int = 1
    load_preprocessed_dir: str = None
    note: str = "Info about this system"

@dataclass
class FeaturesConfig(ConfigBase):
    type: str | None = field(default=None, metadata={"choice": ["distance", "angle", "custom"]})
    selection: str | list | None = field(default=None, metadata={"choice": ["distances", "displacements", "neighbors", "phi", "psi", "chi1", "chi2", "chi3", "chi4", "omega"]})
    atom_selection: str | list | None = None
    pair_selection: Path | str | list | None = None

@dataclass
class TICAConfig(ConfigBase):
    lag_time_frames_range: list = field(default_factory=lambda: [1, 50])
    n_components: int = 3
    lag_time_frames_grid_size: int = 20
    selected_lag_time: int = None
    selected_n_components: int = 3

@dataclass
class ClusterConfig(ConfigBase):
    method: str = field(default="KMeans", metadata={"choice": ["KCenters","KMeans","KMedoids","MiniBatchKMedoids","MiniBatchKMeans"]})
    n_clusters: int = 200
    random_seed: int = 42
    cv_path: Path | None = None

@dataclass
class microMSMConfig(ConfigBase):
    lag_time_frames_range: list = field(default_factory=lambda: [1, 50])
    lag_time_frames_grid_size: int = 20
    n_timescales: int | None = None
    reversible_type: str = field(default="transpose", metadata={"choice": ["transpose", "mle"]})
    ergodic_cutoff: bool = False
    selected_lag_time: int = None
    selected_n_timescales: int = None
    micro_assign_path: str | None = None

@dataclass
class macroMSMConfig(ConfigBase):
    n_macrostates: int = None
    lump_method: str = "PCCAPlus"
    reversible_type: str = "mle"
    ergodic_cutoff: bool = False

@dataclass
class AgentConfig(ConfigBase):
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
class ConfigState(ConfigBase):
    config: AgentConfig
    touched_sections: set[str] = field(default_factory=set)

def from_dict(cls, data: dict[str, Any]):
    kwargs = {}
    type_hints = get_type_hints(cls)
    known_fields = {f.name for f in fields(cls)}
    unknown_fields = set(data) - known_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown fields for {cls.__name__}: "
            f"{sorted(unknown_fields)}"
        )
    for f in fields(cls):
        value = data.get(f.name, None)
        field_type = type_hints[f.name]
        if is_dataclass(field_type):
            kwargs[f.name] = from_dict(field_type, value or {})
        elif value is not None:
            kwargs[f.name] = value
    return cls(**kwargs)

# to remove if pass 
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

def allowed_name_val_type(obj, prefix=""):
    names = []
    val_dict = {}
    type_dict = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        name = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(value):
            sub_names, sub_val_dict, sub_type_dict = allowed_name_val_type(value, name)
            names.extend(sub_names)
            val_dict.update(sub_val_dict)
            type_dict.update(sub_type_dict)
        else:
            names.append(name)
            type_names = [part.strip() for part in f.type.split("|")]
            if "Path" in type_names:
                type_names.append("str")     
            type_names = [type_map.get(item, item) for item in type_names]
            type_dict[name] = type_names
            if f.metadata.get('choice'):
                val_dict[name] = f.metadata['choice']
    return names, val_dict, type_dict
ALLOWED_PATH, ALLOWED_VAL_DICT, ALLOWED_TYPE_DICT = allowed_name_val_type(AgentConfig())

def dump_config_yaml(state: ConfigState) -> str:
    data = serialize_config_subset(state)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

def load_yaml_config_state(text: str, touched_sections: set[str] | None = None) -> ConfigState:
    if isinstance(text, Path):
        text = text.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    touched = set(touched_sections or ())
    touched.update(section for section in SECTION_NAMES if section in data)
    return ConfigState(config=from_dict(AgentConfig, data), touched_sections=touched)

def serialize_config_subset(state: ConfigState) -> dict[str, Any]:
    full_data = asdict(state.config)
    data = {"run_dir": str(full_data["run_dir"])}
    for section_name in SECTION_NAMES:
        if section_name in state.touched_sections:
            data[section_name] = full_data[section_name]
    return data

def save_config(state, path):
    if isinstance(state, ConfigState):
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

SYSTEM_PROMPT = """You are an MSM building agent for a multi-stage molecular dynamics simulation analysis workflow with MSMbuilder.

Your role:
- Help the user sequentially run through the stages.
- Each stage has its specific tasks:
  0) Inspect data
  1) Stage 1: featurization
  2) Stage 2: tICA parameter scan
  3) Stage 3: fit tICA with selected parameters
  4) Stage 4: cluster data points according to tICA collective variables
  5) Stage 5: scan parameters to build a Markov state model with cluster labels
  6) Stage 6: build a Markov state model with cluster labels
  7) Stage 7: lump clusters according to transitions and evaluate the model
- If a tool call is successful, summarize the result and suggest on next steps.
- If a tool call fails, inspect errors in the result and include possible reasons in your responses.
- Provide parameter tuning suggestions when receiving hints.
- If the user says 'ok', 'continue', or 'next', usually move to the next stage without editing config.
- If the user asks to rerun, use the newest config and rerun the current stage.
- Keep responses concise, practical, and stage-aware.

Feature selection rules:
- For feature selection, first check if preprocessed features exist. If not, follow user specified feature selections. \
If user did not specify, inspect topology and suggest features referring to feature templates. 
- If providing feature selection suggestions, base them on user's request and system's topology. \
Include keywords when suggesting: angle, torsion, dihedral, rotamer, sidechain, ligand, binding, unbinding, pocket, pose. \
Update these keywords to data.note to help featurization. 
- If user provided residue selections to select atoms, summarize the selections and wrap them with '{' '}' in your response. \
And update these formated selections to data.note to help featurization. \
For example, if the user selected residues 10 to 20 in chain A and residues 40 to 50 in chain B, your response should \
include: {A: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], B: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]}.
- If working with atom selections, summarize the selection following mdtraj atom selection grammar into list of selection rules and update atom_selection in the config. \
- Do not directly generate pair selections based on residue selection: pair selection is for atom indices. \
The safe way is to add the formated residue selections to note and call tool to convert to correct atom indices.
- If user want to *refine* feature pair selection based on contact frequency, ensure stage1 have run with distance feature. \
Add contact frequency to data.note to help featurization.

Pipeline rules:
- Stage 3 requires tica.selected_lag_time to be set.
- Stage 6 requires microMSM.selected_lag_time to be set.
- To skip stage 2 and 3, and proceed with raw features for stage 4. Set cluster.cv_path to feature path from stage 1. \
However, this approach is generally not encouraged, because tICA is important for noise filtering and finding dominant processes. 
- To skip previous stages and build MSM from user provided microstate assignments, update microMSM.micro_assign_path in config to \
the path of the microstate assignments file. 
"""

SYSTEM_PROMPT += f"""Config update rules:
- If the user have specified their desired config, use update_config_value to sequentially update all values first, then rerun the relevant stage.
- If the requested path and value are unambiguous, update immediately. Ask a clarification question only when the path or value is ambiguous.
- The allowed config paths are {', '.join(ALLOWED_PATH)}.
- The allowed config values for selected config paths are {ALLOWED_VAL_DICT} or list of them.
- The allowed config types for config paths are {ALLOWED_TYPE_DICT}."""


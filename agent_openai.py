from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml
from openai import OpenAI

# ===== Adjust imports to your real project layout =====
from msm_agent.stage import (
    run_stage1_featurization,
    run_stage2_tica_scan,
    run_stage3_tica_fit,
    run_stage4_cluster,
    run_stage5_msm_scan,
    run_stage6_msm_fit,
    run_stage7_lumpeval 
)


# ----------------------------
# YAML / config helpers
# ----------------------------
def safe_yaml_load(s: str) -> Dict[str, Any]:
    obj = yaml.safe_load(s)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return obj


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def set_nested_key(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_nested_key(cfg: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    cur = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(path)
        cur = cur[p]
    return cur


def init_default_config() -> str:
    example_path = Path("examples/interactive_pipeline.yaml")
    if example_path.exists():
        return example_path.read_text()

    fallback = {
        "run": {
            "output_dir": "runs",
            "run_name": "ala2_test",
            "seed": 42,
            "run_dir": None,  
        },
        "data": {
            "kind": "xtc",
            "dir": "/path/to/xtc_dir",
            "topology": "/path/to/topology.pdb",
            "stride": 1,
            "saving_interval": 1.0,
            "load_preprocessed_dir": None,
        },
        "features": {
            "type": "angle",
            "selection": ["phi", "psi"],
            "atom_selection": "BACKBONE",
        },
        "tica": {
            "lag_time_frames_range": [1, 10],
            "lag_time_frames_grid_size": 20,
            "n_components": 2,
            "selected_lag_time": None,
            "selected_n_components": None,
        },
        "microMSM": {
            "lag_time_frames_range": [1, 10],
            "lag_time_frames_grid_size": 20,
            "n_timescales": 5,
            "reversible_type": "transpose",
            "ergodic_cutoff": 1.0,
        },
        "macroMSM": {
            "n_macrostates": 4,
            "lump_method": "PCCAPlus"
        },
        "clustering": {
            "method": "KMeans",
            "n_clusters": 50,
        },
        "plots": {
            "gridsize": 40,
            "bins": 50,
        },
        "evaluation": {
            "plateau_k": None,
            "min_occupancy": 10,
            "ck_plot_only": False,
            "ck_test_steps": 4,
            "ck_test_states": 6,
        },
    }
    return yaml_dump(fallback)


# ----------------------------
# Session state
# ----------------------------
@dataclass
class SessionState:
    current_cfg_yaml: str = ""
    current_cfg_obj: Optional[Dict[str, Any]] = None

    current_run_dir: Optional[str] = None
    current_stage: str = "init"

    latest_summary: str = ""
    latest_plot_path: Optional[str] = None
    error_msg: Optional[Dict[str, Any]] = None

    # optional: keep tool events for debugging
    tool_log: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------
# LLM agent
# ----------------------------
CLIENT = OpenAI()
MODEL = "gpt-5.2"

SYSTEM_PROMPT = """You are an MSM building agent for a multi-stage molecular dynamics simulation analysis workflow with MSMbuilder.

Your role:
- Help the user sequentially run thorugh Stages.
- Each stage have their specific tasks:
  1) Stage 1: featurization
  2) Stage 2: tICA parameter scan
  3) Stage 3: fit tICA with selected parameters
  4) Stage 4: cluster data points according to tica collevtive variables
  5) Stage 5: scan parameters to build markov state model with cluter labels
  6) Stage 6: build markov state model with cluster labels
  7) Stage 7: lump clusters acording to transitions and evaluate model
- If the user asks to modify config, use update_config_value first, then rerun the relevant stage.
- Do not rewrite the whole YAML unless necessary. Prefer update_config_value.
- After each tool result, summarize clearly and ask the user what they want to do next.
- If tool call results is not success, inspect errors in result and include possible reasons in your responses.
- Provide parameter tuning suggestions when receiving hints. 

Important rules:
- Each stage need the output from previous stage to run.
- Stage 3 requires tica.selected_lag_time set.
- Stage 6 requires microMSM.selected_lag_time set.
- If the user says 'ok', 'continue', or 'next', usually move to the next stage without editing config.
- If the user asks to rerun, use the newest config and rerun current stage.
- If the user asks to change parameters, confirm the user with parameters to change, update config and rerun current stage.
- Keep responses concise, practical, and stage-aware.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_current_status",
        "description": "Get current workflow status, current stage, current run_dir, latest summary, and latest plot path.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_current_config",
        "description": "Get the current config object.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_config_value",
        "description": "Update one config field by dotted path. value_yaml can be a scalar, list, dict, string, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value_yaml": {"type": "string"}, ############################ check #################################
            },
            "required": ["path", "value_yaml"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage1_featurization",
        "description": "Run Stage 1: load data and featurize.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage2_tica_scan",
        "description": "Run Stage 2: tICA lag scan using the latest Stage 1 result in current_run_dir.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage3_tica_fit",
        "description": "Run Stage 3: final tICA fit using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage4_cluster",
        "description": "Run Stage 4: clustering using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage5_msm_scan",
        "description": "Run Stage 5: MSM parameter scan using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage6_msm_fit",
        "description": "Run Stage 6: MSM fit using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_stage7_lumpeval",
        "description": "Run Stage 7: lump and evaluate model using current_run_dir and current config.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


# ----------------------------
# Tool executor
# ----------------------------
def tool_get_current_status(st: SessionState) -> Dict[str, Any]:
    return {
        "current_stage": st.current_stage,
        "current_run_dir": st.current_run_dir,
        "latest_summary": st.latest_summary,
        "latest_plot_path": st.latest_plot_path,
    }


def tool_get_current_config(st: SessionState) -> Dict[str, Any]:
    return copy.deepcopy(st.current_cfg_obj or {})


def tool_update_config_value(st: SessionState, path: str, value_yaml: str) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")
    value = yaml.safe_load(value_yaml)
    set_nested_key(st.current_cfg_obj, path, value)
    st.current_cfg_yaml = yaml_dump(st.current_cfg_obj)
    return {
        "success": True,
        "updated_path": path,
        "new_value": value,
    }

def tool_run_stage(st: SessionState, stage: int) -> Dict[str, Any]:
    if st.current_cfg_obj is None:
        raise ValueError("No current config loaded.")

    stage_map = {
        1: (run_stage1_featurization, "stage1_done"),
        2: (run_stage2_tica_scan, "stage2_done"),
        3: (run_stage3_tica_fit, "stage3_done"),
        4: (run_stage4_cluster, "stage4_done"),
        5: (run_stage5_msm_scan, "stage5_done"),
        6: (run_stage6_msm_fit, "stage6_done"),
        7: (run_stage7_lumpeval, "stage7_done"),
    }
    if stage not in stage_map:
        raise ValueError(f"Unsupported stage: {stage}")

    fn, stage_done_flag = stage_map[stage]

    if stage == 1:
        result = fn(st.current_cfg_obj, st.current_run_dir)
        if st.current_run_dir is None:
            st.current_run_dir = result.get("run_dir")
            set_nested_key(st.current_cfg_obj, "run.run_dir", st.current_run_dir)
            st.current_cfg_yaml = yaml_dump(st.current_cfg_obj)
    elif not st.current_run_dir:
        raise ValueError("No current_run_dir found. Please run Stage 1 first.")
    else:
        result = fn(st.current_cfg_obj, st.current_run_dir)

    st.current_stage = stage_done_flag
    st.latest_summary = result.get("summary", "")
    st.latest_plot_path = result.get("plot_path")
    st.error_msg = result.get("errors","")
    return result


def execute_tool(st: SessionState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_current_status":
        result = tool_get_current_status(st)
    elif name == "get_current_config":
        result = tool_get_current_config(st)
    elif name == "update_config_value":
        result = tool_update_config_value(
            st,
            path=args["path"],
            value_yaml=args["value_yaml"],
        )
    elif name in {"run_stage1", "run_stage1_featurization"}:
        result = tool_run_stage(st, 1)
    elif name in {"run_stage2", "run_stage2_tica_scan"}:
        result = tool_run_stage(st, 2)
    elif name in {"run_stage3", "run_stage3_tica_fit"}:
        result = tool_run_stage(st, 3)
    elif name in {"run_stage4", "run_stage4_cluster"}:
        result = tool_run_stage(st, 4)
    elif name in {"run_stage5", "run_stage5_msm_scan"}:
        result = tool_run_stage(st, 5)
    elif name in {"run_stage6", "run_stage6_msm_fit"}:
        result = tool_run_stage(st, 6)
    elif name in {"run_stage7", "run_stage7_lumpeval"}:
        result = tool_run_stage(st, 7)
    else:
        raise ValueError(f"Unknown tool: {name}")

    st.tool_log.append(
        {
            "tool": name,
            "args": args,
            "result": result,
        }
    )
    return result


# ----------------------------
# Agent loop
# ----------------------------
def normalize_chat_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "value" in content and isinstance(content["value"], str):
            return content["value"]
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, list):
        parts = [normalize_chat_content(x) for x in content]
        return "\n".join([p for p in parts if p])

    return str(content)

def to_llm_messages(chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    msgs = []
    for m in chat_history or []:
        role = str(m.get("role", "user"))
        if role not in {"system", "user", "assistant", "developer"}:
            continue

        text = normalize_chat_content(m.get("content", ""))

        msgs.append(
            {
                "role": role,
                "content": text,
            }
        )
    return msgs


def run_agent_once(
    user_message: str,
    chat_history: List[Dict[str, str]],
    yaml_text: str,
    st: SessionState,
) -> Tuple[List[Dict[str, str]], SessionState, str, str, str, Optional[str]]:
    # 1) Sync YAML editor -> session config
    try:
        cfg_obj = safe_yaml_load(yaml_text)
    except Exception as e:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": f"Config YAML parse error: {e}"})
        return (
            chat_history,
            st,
            yaml_text,
            st.latest_summary,
            st.current_run_dir or "",
            st.latest_plot_path,
        )

    st.current_cfg_obj = cfg_obj
    st.current_cfg_yaml = yaml_text
    st.current_run_dir = cfg_obj.get("run", {}).get("run_dir", st.current_run_dir) # read from config or none

    # 2) Append user message to UI chat history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})

    # 3) Build LLM inputs
    context_text = (
    f"Current stage: {st.current_stage}\n"
    f"Current run_dir: {st.current_run_dir}\n"
    f"Latest summary:\n{st.latest_summary or 'None'}\n"
    f"Latest plot path: {st.latest_plot_path or 'None'}\n"
)

    input_msgs = [
    {
        "role": "developer",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "system",
        "content": context_text,
    },
    ] + to_llm_messages(chat_history)


    # 4) Initial model call
    response = CLIENT.responses.create(
        model=MODEL,
        input=input_msgs,
        tools=TOOLS,
    )

    # 5) Tool-calling loop
    max_loops = 8
    loops = 0
    while loops < max_loops:
        loops += 1
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            break

        tool_outputs = []
        for fc in function_calls:
            try:
                args = json.loads(fc.arguments or "{}")
            except Exception:
                args = {}

            try:
                result = execute_tool(st, fc.name, args)
                output = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:
                output = json.dumps(
                    {"success": False, "error": str(e)},
                    ensure_ascii=False,
                )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": output,
                }
            )

        response = CLIENT.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
        )

    assistant_text = response.output_text or "Done."

    chat_history.append({"role": "assistant", "content": assistant_text})

    # 6) Sync config back to YAML editor in case tool updated it
    yaml_text = st.current_cfg_yaml or yaml_text

    return (
        chat_history,
        st,
        yaml_text,
        st.latest_summary,
        [(p,p) for p in st.latest_plot_path] if st.latest_plot_path else None,
    )


# ----------------------------
# UI
# ----------------------------
def build_app():
    with gr.Blocks(title="MSM Agent MVP") as demo:
        st = gr.State(SessionState())

        gr.Markdown("## MSM building agent \nLLM-empowered molecular dynamics simulation analysis tool")

        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Chat", height=560)
                user_in = gr.Textbox(
                    label="Message",
                    placeholder='Examples: "run featurization", "rerun with current config", "ok continue", "set selected tica lagtime to 3 and run tica scan"',
                )
                btn_send = gr.Button("Send")
                #latest_image = gr.Image(label="Output figure", type="filepath", height=420) 
                latest_image = gr.Gallery(label="Output figure", columns=1, height="500", object_fit="contain")

            with gr.Column(scale=1):
                cfg_editor = gr.Code(
                    label="Current config (YAML)",
                    language="yaml",
                    value=init_default_config(),
                )
                latest_summary = gr.Textbox(label="Latest summary", lines=12)
                #current_run_dir = gr.Textbox(label="Current run dir")
                #latest_image = gr.Image(label="Output figure", type="filepath", height=320) ########## what happen if two out #######################
                #out_image = gr.Gallery(label="Visualized results").style(grid=[1], height="auto")

        btn_send.click(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, latest_summary, latest_image],
            api_visibility="private",
        )

        user_in.submit(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, latest_summary, latest_image],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch()
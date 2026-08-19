from __future__ import annotations

import copy
import json, html
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import gradio as gr
import yaml
from openai import OpenAI

from msm_agent.stage import (
    run_stage1_featurization,
    run_stage2_tica_scan,
    run_stage3_tica_fit,
    run_stage4_cluster,
    run_stage5_msm_scan,
    run_stage6_msm_fit,
    run_stage7_lumpeval 
)
from msm_agent.featurizationv1 import inspect_data
from msm_agent.config import (
    AgentConfig,
    ConfigState,
    dump_config_yaml,
    load_yaml_config_state,
    save_config,
    ALLOWED_VAL_DICT,
    ALLOWED_TYPE_DICT,
    SYSTEM_PROMPT,
)
from msm_agent.parameters import Plot_param


# ----------------------------
# Config helpers
# ----------------------------
def init_config_state() -> ConfigState:
    init_config = AgentConfig()
    run_dir = init_config.run_dir
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
    return ConfigState(config=init_config, touched_sections=set(["data"]))

# ----------------------------
# Session state
# ----------------------------
@dataclass
class SessionState:
    current_cfg_yaml: str = "" # for editor display
    current_cfg_state: Optional[ConfigState] = None # config with flag

    current_run_dir: Optional[Path] = None
    current_stage: str = "init"

    latest_summary: str = ""
    plot_path: Optional[List[str]] = None
    error_msg: Optional[Dict[str, Any]] = None

    # optional: keep tool events for debugging
    tool_log: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------
# LLM agent
# ----------------------------
CLIENT = OpenAI()
MODEL = "gpt-5.2"

TOOLS = [
    {
        "type": "function",
        "name": "get_current_status",
        "description": "Get current workflow status, current stage, current run_dir, latest summary, and latest plot path.",
        "strict": True,
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
        "strict": True,
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
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value_yaml": {"type": "string"}
            },
            "required": ["path", "value_yaml"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_inspect_topology",
        "description": "Inspect the topology of the system. Pass user message for later use.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "user_message": {"type": "string"},
            },
            "required": ["user_message"],
            "additionalProperties": False,
        },  
    },
    {
        "type": "function",
        "name": "run_stage1_featurization",
        "description": "Run Stage 1: load data and featurize based on message.",
        "strict": True,
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
        "strict": True,
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
        "strict": True,
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
        "strict": True,
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
        "strict": True,
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
        "strict": True,
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
        "strict": True,
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
        "plot_path": st.plot_path,
    }


def tool_get_current_config(st: SessionState) -> Dict[str, Any]:
    if st.current_cfg_state is None:
        return {}
    return copy.deepcopy(st.current_cfg_state.config)

def set_nested_key(cfg: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if is_dataclass(cur):
            cur = getattr(cur, p)
        else:
            cur = cur[p]
    if isinstance(cur, dict):
        cur[parts[-1]] = value
    else:
        setattr(cur, parts[-1], value)

def tool_update_config_value(st: SessionState, path: str, value_yaml: str) -> Dict[str, Any]:
    value = yaml.safe_load(value_yaml)
    st.current_stage = "config_update"
    if st.current_cfg_state is None:
        return {
        "success": False,
        "updated_path": path,
        "new_value": value,
        "error": "No current config loaded. Please load a config before updating values.",
    }
    if path not in ALLOWED_TYPE_DICT:
        return {
        "success": False,
        "updated_path": path,
        "new_value": value,
        "error": f"Unsupported config path. Allowed paths: \
        {', '.join(ALLOWED_TYPE_DICT.keys())}",
    }
    if path in ALLOWED_TYPE_DICT and not type(value) in ALLOWED_TYPE_DICT[path]:
        return {
        "success": False,
        "updated_path": path,
        "new_value": value,
        "error": f"Unsupported config type. Allowed type for {path}: \
        {', '.join(ALLOWED_TYPE_DICT[path])}",
    }
    if path in ALLOWED_VAL_DICT:
        if isinstance(value, list):
            if not all(v in ALLOWED_VAL_DICT[path] for v in value):
                return {
                "success": False,
                "updated_path": path,
                "new_value": value,
                "error": f"Unsupported config value. Allowed values for {path}: \
                {', '.join(ALLOWED_VAL_DICT[path])}",
            }
        elif value not in ALLOWED_VAL_DICT[path]:
            return {
                "success": False,
                "updated_path": path,
                "new_value": value,
                "error": f"Unsupported config value. Allowed values for {path}: \
                {', '.join(ALLOWED_VAL_DICT[path])}",
            }
    set_nested_key(st.current_cfg_state.config, path, value)
    st.current_cfg_yaml = dump_config_yaml(st.current_cfg_state)
    save_config(st.current_cfg_state, st.current_run_dir / "config.yaml")
    return {
        "success": True,
        "updated_path": path,
        "new_value": value,
    }

def tool_inspect_topology(st: SessionState) -> Dict[str, Any]:
    cfg_state = st.current_cfg_state
    st.current_stage = "inspect_topology"
    output = inspect_data(asdict(cfg_state.config))
    return {
        "success": True,
        "inspection": output,
    }


def tool_run_stage(st: SessionState, stage: int, args: Dict[str, Any]) -> Dict[str, Any]:
    if st.current_cfg_state is None:
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
    result = fn(st.current_cfg_state.config, st.current_run_dir)

    stage_sections = {
        1: ["features"],
        2: ["tica"],
        3: ["tica"],
        4: ["clustering"],
        5: ["microMSM"],
        6: ["microMSM"],
        7: ["macroMSM"],
    }
    if result.get("success"):
        for section_name in stage_sections.get(stage, []):
            st.current_cfg_state.touched_sections.add(section_name)

    config_path = st.current_run_dir / "config.yaml"
    if config_path.exists():
        touched_sections = set(st.current_cfg_state.touched_sections)
        st.current_cfg_state = load_yaml_config_state(
            config_path,
            touched_sections=touched_sections,
        )
        st.current_cfg_yaml = dump_config_yaml(st.current_cfg_state)

    st.current_stage = stage_done_flag
    st.latest_summary = result.get("summary", "")
    plot_path = result.get("plot_path")
    if isinstance(plot_path, str):
        st.plot_path.append(plot_path)
    elif isinstance(plot_path, list):
        st.plot_path.extend(plot_path)
    st.plot_path = list(set(st.plot_path))  # remove duplicates
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
    elif name == "run_inspect_topology":
        result = tool_inspect_topology(st)
    elif name == "run_stage1_featurization":
        result = tool_run_stage(st, 1, args)
    elif name == "run_stage2_tica_scan":
        result = tool_run_stage(st, 2, args)
    elif name == "run_stage3_tica_fit":
        result = tool_run_stage(st, 3, args)
    elif name == "run_stage4_cluster":
        result = tool_run_stage(st, 4, args)
    elif name == "run_stage5_msm_scan":
        result = tool_run_stage(st, 5, args)
    elif name == "run_stage6_msm_fit":
        result = tool_run_stage(st, 6, args)
    elif name == "run_stage7_lumpeval":
        result = tool_run_stage(st, 7, args)
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

def _extract_passed_stages(tool_log: List[Dict[str, Any]]) -> str: #add note from LLM
    seen: List[str] = []
    for item in tool_log:
        if item.get("tool").split("_")[0] == "run":
            stage = item.get("tool") #.split("_")[1:]
            seen.append(stage)
    if not seen:
        return "No stage completed yet."
    return "\n".join(f"🟢 {stage}" for stage in seen)

TEMPLATE = Path("./msm_agent/vis_box.html").read_text()
def build_html(file: str, st: SessionState):
    if file is None:
        return "<h3>No structure loaded</h3>"
    pdb_text = Path(file).read_text()
    viewer_html = TEMPLATE.replace(
        "__PDB_TEXT__",
        json.dumps(pdb_text),
    )
    output_path = Path(st.current_run_dir / "viewer_content.html")
    output_path.write_text(viewer_html, encoding="utf-8")
    print(f"Wrote viewer HTML to {output_path.resolve()}")
    st.current_cfg_state.config.data.topology = str(file)
    save_config(st.current_cfg_state, st.current_run_dir / "config.yaml")
    st.current_cfg_yaml = dump_config_yaml(st.current_cfg_state)
    return (
        '<iframe style="width:100%;height:80vh;border:0" '
        f'srcdoc="{html.escape(viewer_html, quote=True)}"></iframe>', st, st.current_cfg_yaml
    )

def set_data_dir(directory: str, st: SessionState):
    if not directory:
        return st, st.current_cfg_yaml
    st.current_cfg_state.config.data.dir = str(directory)
    st.current_cfg_state.touched_sections.add("data")
    st.current_cfg_yaml = dump_config_yaml(st.current_cfg_state)
    save_config(st.current_cfg_state, st.current_run_dir / "config.yaml")
    return st, st.current_cfg_yaml

def run_agent_once(
    user_message: str,
    chat_history: List[Dict[str, str]],
    yaml_text: str,
    st: SessionState,
) -> Tuple[List[Dict[str, str]], SessionState, str, str, Optional[List[Tuple[str, str]]]]:
    # 1) Sync YAML editor -> session config
    try:
        touched_sections = set(st.current_cfg_state.touched_sections) if st.current_cfg_state is not None else None
        cfg_state = load_yaml_config_state(yaml_text, touched_sections=touched_sections) # read from yaml 
    except Exception as e:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": f"Config YAML parse error: {e}"})
        return (
            chat_history,
            st,
            st.current_cfg_yaml,
            st.latest_summary,
            st.plot_path,
        )
    st.current_cfg_state = cfg_state # sync editor to st
    
    # 2) Append user message to UI chat history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})

    # 3) Build LLM inputs
    context_text = (
    f"Current stage: {st.current_stage}\n"
    f"Current run_dir: {st.current_run_dir}\n"
    f"Latest summary:\n{st.latest_summary or 'None'}\n"
    f"Latest plot path: {st.plot_path or 'None'}\n"
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


    # 4) Initial model call, tool routing or final response
    response = CLIENT.responses.create(
        model=MODEL,
        input=input_msgs,
        tools=TOOLS,
    )

    # 5) Tool-calling loop TODO: wrap into a loop and retry 1 time when not success 
    max_loops = 8 # max call 8 tools in a single pass
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

    # read from disk for newest config after tool call, safer than passing cfg back
    st.current_cfg_state = load_yaml_config_state(st.current_run_dir / "config.yaml", touched_sections=set(st.current_cfg_state.touched_sections))
    st.current_cfg_yaml = dump_config_yaml(st.current_cfg_state) # updated cfg state

    tool_log = _extract_passed_stages(st.tool_log)

    return (
        chat_history,
        st,
        st.current_cfg_yaml,
        tool_log,
        #st.tool_log,
        [(p,p) for p in st.plot_path] if st.plot_path else None,
    )


# ----------------------------
# UI
# ----------------------------
def build_app():
    initial_cfg_state = init_config_state()
    initial_session = SessionState(
        current_cfg_state=initial_cfg_state,
        current_cfg_yaml=dump_config_yaml(initial_cfg_state),
        current_run_dir=Path(initial_cfg_state.config.run_dir),
        plot_path=[],
    )
   
    plotstyle = Plot_param()
    plotstyle.apply()

    if initial_session.current_run_dir is not None:
        save_config(initial_cfg_state, initial_session.current_run_dir / "config.yaml")

    with gr.Blocks(title="MSMbuilder Agent") as demo:
        st = gr.State(initial_session)

        gr.Markdown("## MSM building agent \nLLM-empowered molecular dynamics simulation analysis tool")
        gr.Markdown("Start by loading your PDB, adding exact path to your local data folder in config editor, " \
                    "and a brief introduction of what you are interested in about your system." \
                    "If you have your curated features, add the path to load preprocessed dir")

        with gr.Row():
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Chat", height=560)
                user_in = gr.Textbox(
                    label="Message",
                    placeholder='Examples: "run featurization", "set selected tica lagtime to 3 and run tica scan"',
                )
                btn_send = gr.Button("Send")
                tool_log = gr.Textbox(label="Tool Usage", lines=12)
                latest_image = gr.Gallery(label="Output figure", columns=1, height="500", object_fit="contain")

            with gr.Column(scale=1):
                pdb_file = gr.File(
                    label="PDB structure",
                    file_types=[".pdb", ".ent"],
                    type="filepath",
                )
                load_button = gr.Button(
                    "Load structure",
                    variant="primary",
                )
                data_dir = gr.Dropdown(
                    label="Trajectory data directory",
                    choices=[str(path.resolve()) for path in Path("data").iterdir() if path.is_dir()],
                )
                data_dir_button = gr.Button("Use data directory")
                viewer_html = gr.HTML(label="Topology viewer", height=400)
                cfg_editor = gr.Code(
                    label="Current config (YAML)",
                    language="yaml",
                    value=initial_session.current_cfg_yaml,
                )

        load_button.click(
            fn=build_html,
            inputs=[pdb_file, st],
            outputs=[viewer_html, st, cfg_editor],
        )

        data_dir_button.click(
            fn=set_data_dir,
            inputs=[data_dir, st],
            outputs=[st, cfg_editor],
        )

        btn_send.click(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, tool_log, latest_image],
            api_visibility="private",
        )

        user_in.submit(
            fn=run_agent_once,
            inputs=[user_in, chat, cfg_editor, st],
            outputs=[chat, st, cfg_editor, tool_log, latest_image],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch()

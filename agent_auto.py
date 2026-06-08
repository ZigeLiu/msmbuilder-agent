import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import yaml
from openai import OpenAI

from agent_openai import init_default_config
from controller_reviewer import PipelineAgent, ReviewerAgent


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
STAGE_ORDER = [
    "decide_feature",
    "config_update",
    "stage1_done",
    "stage2_done",
    "stage3_done",
    "stage4_done",
    "stage5_done",
    "stage6_done",
    "stage7_done",
]
CACHE_DIR_MARKERS = {"gradio", ".gradio", "tmp"}


def _build_allowed_paths() -> List[str]:
    paths = {
        str(Path.cwd().resolve()),
        str(Path(tempfile.gettempdir()).resolve()),
        "/private/tmp",
        "/private/var/folders",
    }
    return sorted(paths)


def _safe_stage_label(stage: str) -> str:
    if not stage:
        return "unknown"
    return stage.replace("_", " ")


def _extract_passed_stages(history: List[Dict[str, Any]]) -> str:
    seen: List[str] = []
    for item in history:
        if item.get("role") == "pipelineagent" and item.get("stage"):
            stage = item.get("stage")
            seen.append(stage)
    if not seen:
        return "No stages completed yet."
    return "\n".join(f"[x] {_safe_stage_label(stage)}" for stage in seen)


def _coerce_path_list(value: Any) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, Iterable):
        paths: List[Path] = []
        for item in value:
            if item:
                paths.append(Path(item))
        return paths
    return []


def _is_cache_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return any(marker in parts for marker in CACHE_DIR_MARKERS)


def _is_hidden_image(path: Path) -> bool:
    return path.name.startswith(".")


def _collect_gallery_images(run_dir: Optional[Path], latest_plot_path: Any) -> List[Tuple[str, str]]:
    seen: set[str] = set()
    gallery: List[Tuple[str, str]] = []
    resolved_run_dir = run_dir.resolve() if run_dir and run_dir.exists() else None

    for path in _coerce_path_list(latest_plot_path):
        resolved = path.expanduser()
        if not resolved.exists() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved_path = resolved.resolve()
        if _is_cache_path(resolved_path) or _is_hidden_image(resolved_path):
            continue
        if resolved_run_dir and resolved_run_dir not in resolved_path.parents:
            continue
        key = str(resolved_path)
        if key not in seen:
            seen.add(key)
            gallery.append((str(resolved_path), resolved_path.name))

    if resolved_run_dir:
        figs_dir = resolved_run_dir / "figs"
        if figs_dir.exists():
            for path in sorted(figs_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                resolved_path = path.resolve()
                if _is_cache_path(resolved_path) or _is_hidden_image(resolved_path):
                    continue
                key = str(resolved_path)
                if key in seen:
                    continue
                seen.add(key)
                gallery.append((str(resolved_path), resolved_path.name))

    return gallery


def _build_status_text(
    stage: str,
    run_dir: Optional[Path],
    latest_summary: str,
    step_index: int,
    max_try: int,
    latest_decision: Optional[str] = None,
) -> str:
    lines = [
        f"Stage: {_safe_stage_label(stage)}",
        f"Iteration: {step_index}/{max_try}",
        f"Run dir: {run_dir or 'not created yet'}",
    ]
    if latest_decision:
        lines.append(f"Reviewer decision: {latest_decision}")
    if latest_summary:
        lines.append("")
        lines.append("Latest summary:")
        lines.append(latest_summary)
    return "\n".join(lines)


def _snapshot(
    pipeline_agent: PipelineAgent,
    history: List[Dict[str, Any]],
    step_index: int,
    max_try: int,
    latest_review: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str, str, List[Tuple[str, str]]]:
    run_dir = pipeline_agent.st.current_run_dir
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    pipeline_status = _build_status_text(
        stage=pipeline_agent.st.current_stage,
        run_dir=run_dir,
        latest_summary=pipeline_agent.st.latest_summary,
        step_index=step_index,
        max_try=max_try,
    )
    reviewer_status = (
        f"Latest decision: {latest_review.get('decision') or 'pending'}\n"
        f"Iteration: {step_index}/{max_try}\n"
        f"Reasoning: {latest_review.get('message') or 'pending'}\n"
    )
    passed_stages = _extract_passed_stages(history)
    config_text = pipeline_agent.st.current_cfg_yaml or ""
    gallery = _collect_gallery_images(run_dir, pipeline_agent.st.latest_plot_path)
    return passed_stages, pipeline_status, reviewer_status, config_text, gallery


def run_auto_agent(yaml_cfg: str):
    client = OpenAI()
    pipeline_agent = PipelineAgent(client)
    reviewer_agent = ReviewerAgent(client)
    history: List[Dict[str, Any]] = []
    current_yaml = yaml_cfg
    max_try = 100

    initial_pipeline_status = "Stage: initializing\nIteration: 0/100\nRun dir: not created yet"
    initial_reviewer_status = "Latest decision: pending\nIteration: 0/100"
    yield "No stages completed yet.", initial_pipeline_status, initial_reviewer_status, current_yaml, []

    for step_index in range(1, max_try + 1):
        result = pipeline_agent.run_stage(history, current_yaml)
        history.append(
            {
                "role": "pipelineagent",
                "stage": result.get("current_stage"),
                "content": result.get("assistant_text", ""),
                "summary": result.get("latest_summary", ""),
            }
        )
        current_yaml = result.get("current_cfg_yaml", current_yaml)

        result = reviewer_agent.review(
            stage=result.get("current_stage", ""),
            history=history,
            summary=result.get("latest_summary", ""),
            params=result.get("current_cfg_yaml", ""),
        )
        history.append(
            {
                "role": "reviewagent",
                "stage": result.get("current_stage"),
                "decision": result.get("decision", ""),
                "content": result.get("message", ""),
            }
        )

        yield _snapshot(
            pipeline_agent=pipeline_agent,
            history=history,
            step_index=step_index,
            max_try=max_try,
            latest_review=result,
        )

        if result.get("decision") == "Stop":
            return

    history.append(
        {
            "role": "system",
            "content": (
                f"Exceeded maximum iterations ({max_try}). "
                "Try adjusting model quality criteria or stage parameters."
            ),
        }
    )
    yield _snapshot(
        pipeline_agent=pipeline_agent,
        history=history,
        step_index=max_try,
        max_try=max_try,
        latest_review={"decision": "Max iterations reached", "message": "Exceeded maximum iterations."},
    )


def refresh_gallery(yaml_cfg: str) -> List[Tuple[str, str]]:
    run_dir: Optional[Path] = None

    try:
        cfg = yaml.safe_load(yaml_cfg) or {}
        run_dir_value = cfg.get("run", {}).get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
    except Exception:
        run_dir = None

    return _collect_gallery_images(run_dir, latest_plot_path=None)


def build_app():
    with gr.Blocks(title="Automatic MSMbuilder Agent") as demo:
        gr.Markdown(
            "## Automatic MSMbuilder Agent\n"
            "Monitor the automatic pipeline agent, reviewer decisions, live config, and generated figures."
        )

        with gr.Row():
            btn_run = gr.Button("Run Automatic Agent", variant="primary")
            btn_refresh = gr.Button("Refresh Gallery")

        with gr.Row():
            passed_stages_box = gr.Textbox(
                label="Passed Stages",
                lines=14,
                interactive=False,
            )
            pipeline_status_box = gr.Textbox(
                label="PipelineAgent Status",
                lines=14,
                interactive=False,
            )
            reviewer_status_box = gr.Textbox(
                label="ReviewAgent Status",
                lines=14,
                interactive=False,
            )

        with gr.Row():
            cfg_editor = gr.Code(
                label="Live Config Editor",
                language="yaml",
                value=init_default_config(),
                lines=30,
            )
            gallery = gr.Gallery(
                label="Existing Pictures",
                columns=2,
                height=640,
                object_fit="contain",
            )

        btn_run.click(
            fn=run_auto_agent,
            inputs=[cfg_editor],
            outputs=[passed_stages_box, pipeline_status_box, reviewer_status_box, cfg_editor, gallery],
            api_visibility="private",
        )
        btn_refresh.click(
            fn=refresh_gallery,
            inputs=[cfg_editor],
            outputs=[gallery],
            api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch(allowed_paths=_build_allowed_paths())

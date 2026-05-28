from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from agent_openai import MODEL, TOOLS, SessionState, ALLOWED_CONFIG_UPDATE_PATHS
from agent_openai import execute_tool, to_llm_messages, safe_yaml_load

SYSTEM_PROMPT = "Run MSM construction with the following configuration." \
                "Start from decide feature selection then move from stage 1 to 7." 

class SearchController:
    '''
    Controller that manages the pipeline and reviewer agents for MSM construction.
    Pipeline agent will only proceed to the next stage if the reviewer agent approves 
    the current stage's results. If the reviewer agent suggests a rerun, the pipeline 
    agent will adjust parameters and rerun the stage. If the reviewer agent suggests 
    stopping, the controller will halt the process and return the history of stages 
    and decisions.
    '''
    def __init__(self, pipeline_agent, reviewer_agent, max_try=100):
        self.pipeline_agent = pipeline_agent
        self.reviewer_agent = reviewer_agent
        self.max_try = max_try
        self.history = []

    def run_pipeline(self, yaml_cfg):
        for i in range(self.max_try):
            result = self.pipeline_agent.run_stage(self.history, yaml_cfg)
            self.history.append(
                {"role": "assistant", 
                 "content": result.get("assistant_text")}
            )
            yaml_cfg = result.get("current_cfg_yaml", yaml_cfg) 

            decision = self.reviewer_agent.review(
                    stage=result.get("current_stage"," "),
                    history=self.history,
                    summary=result.get("latest_summary", " "),
                    params=result.get("current_cfg_yaml", " "),
                )
          
            self.history.append({
                "role": "reviewer", 
                "content": decision.get("message", "Finished")
            })

            if decision["decision"] == "Stop":
                break
        else:
            self.history.append({
                "role": "system", 
                "content": f"Exceeding maximum try {self.max_try}. Try changing metric parameters to lower reequired model quality"
            })

        return self.history
    
class PipelineAgent:
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools
        self.st = SessionState()

    def run_stage(
        self,
        history: List[Dict[str, str]],
        yaml_cfg: str,
    ) -> Tuple[List[Dict[str, str]], SessionState, str, str, str, Optional[str]]:
        # 1) Sync YAML editor -> session config
        try:
            cfg_obj = safe_yaml_load(yaml_cfg) # yaml to dict obj
        except Exception as e:
            assistant_text = f"Config YAML parse error: {e}"
            return (
                assistant_text,
                self.st,
                yaml_cfg,
            )

        self.st.current_cfg_obj = cfg_obj
        self.st.current_cfg_yaml = yaml_cfg
        self.st.current_run_dir = cfg_obj.get("run", {}).get("run_dir", self.st.current_run_dir) # read from config or none
        self.st.current_run_dir = Path(self.st.current_run_dir) if self.st.current_run_dir else None 

        context_text = (
        f"Current stage: {self.st.current_stage}\n"
        f"Current run_dir: {self.st.current_run_dir}\n"
        f"Latest summary:\n{self.st.latest_summary or 'None'}\n"
        f"Latest plot path: {self.st.latest_plot_path or 'None'}\n"
    )

        input_msgs = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": context_text,
        },
        ] + to_llm_messages(history)


        # 4) Initial model call, tool routing or final response
        response = self.client.responses.create(
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
                    result = execute_tool(self.st, fc.name, args)
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

            response = self.client.responses.create(
                model=MODEL,
                previous_response_id=response.id,
                input=tool_outputs,
            tools=TOOLS,
            )

        assistant_text = response.output_text or "Done."

        return {
            "assistant_text": assistant_text,
            "current_stage": self.st.current_stage,
            "current_cfg_yaml": self.st.current_cfg_yaml,
            "latest_summary": self.st.latest_summary,
        }
    
class ReviewerAgent:
    def __init__(self, client):
        self.client = client

    def review(self, stage, history, summary, params):
        message = """
            You are an expert reviewer for MSM construction. Inspect curent review and history results 
            to approve or decline the current stage. Start response with 'Approve |', 'Decline |' or 'Stop |'.
            """
        if stage in ["decide_feature", "update_config"]:
            param_check = []
            for para in params:
                if para not in ALLOWED_CONFIG_UPDATE_PATHS:
                    param_check.append(para) 
            message += f"""
                Check updated config to make sure they are in the correct format.
                There are paths that are not allowed: {param_check}.
                If no path that is not allowed, approve current step. 
                Otherwise decline current step, correct the format or path name and rerun.
            """
        else:
            review = [f"Checking {stage} results."]
            warnings = [line for line in summary.splitlines() if line.startswith("Warning")]
            if warnings:
                review += "Warnings detected:\n"
                for w in warnings:
                    review += f"- {w}\n"
                review += "Please address all the warnings before proceeding to the next stage.\n"
            else:
                review += "No warnings detected. You can proceed to the next stage.\n"
            recommendations = [line for line in summary.splitlines() if line.startswith("Recommendation")]
            if recommendations:
                review += recommendations
            message += f"""
                If declining, provide reasons and next step on the decision. 
                Next step move can be rerun current stage, go back to a certain stage or skip certain stage.
                If not sure, prefer decline with reason and rerun current stage with slightly tuned parameters. 
                If approving and already at stage 7, stop the building procedure. 
                Current stage: {stage}
                Current parameters: {params}
                Current review: {review}
                History: {history}
            """

        response = self.client.responses.create(
            model="gpt-5.3",
            input=message,
        )
        # TODO: customized output format {'message':,'decision':summarize the message}
        decision = response.output_text.split(' |')[0]

        return {
            "decision": decision,
            "message": response.output_text,
        }

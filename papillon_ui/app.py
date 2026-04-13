from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys

import dspy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
TEMPLATE_DIR = APP_DIR / "templates"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from papillon.dspy_compat import build_openai_lm
from papillon.pipeline_factory import build_pipeline
from papillon.prompt_paths import parse_model_prompt


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


class Query(BaseModel):
    query: str


class FinalInput(BaseModel):
    original_query: str
    original_prompt: str = ""
    edited_prompt: str = ""
    route: Optional[str] = None


class PipelineRuntime:
    def __init__(self):
        self.pipeline = None
        self.edit_history = []

    def configure(self, pipeline):
        self.pipeline = pipeline

    @staticmethod
    def _build_prompt_payload(route: str, cloud_prompt: str, editable: Optional[bool] = None) -> dict:
        is_direct = route == "direct"
        prompt_editable = (not is_direct) if editable is None else editable
        prompt_title = "Direct Cloud Query" if is_direct else "Protected Cloud Prompt"
        prompt_hint = (
            "Direct route: no prompt editing is needed. The original query will be sent directly to the cloud model."
            if is_direct
            else "Protected route: this is the exact text sent to the cloud model. Review and edit the structured Task / Context / Style prompt before continuing."
        )
        prompt_explanation = (
            "This is the exact text sent to the cloud model. On the direct route it matches the original user query."
            if is_direct
            else "This is the exact text sent to the cloud model. On the protected route it is rewritten into Task, Context, and Style sections to reduce raw PII exposure."
        )
        return {
            "prompt": cloud_prompt,
            "cloud_prompt": cloud_prompt,
            "prompt_title": prompt_title,
            "prompt_hint": prompt_hint,
            "prompt_explanation": prompt_explanation,
            "cloud_prompt_format": "direct_original_query" if is_direct else "structured_task_context_style",
            "prompt_editable": prompt_editable,
        }

    @classmethod
    def _preview_error_payload(cls, error: Exception) -> dict:
        error_message = str(error) or error.__class__.__name__
        payload = cls._build_prompt_payload("protected", "", editable=False)
        payload.update(
            {
                "route": "protected",
                "route_reason": "preview_error",
                "editable": False,
                "structured_fields": {},
                "detected_pii": [],
                "redacted_query": "",
                "placeholder_map": {},
                "detector_available": False,
                "detector_uncertain": True,
                "detector_error": error_message,
                "preview_error": error_message,
                "prompt_hint": "Protected route: the cloud prompt could not be generated. Check the local model or API connection, then retry.",
                "prompt_explanation": "Prompt generation failed before PAPILLON could produce the exact cloud prompt. The error details are returned so the UI can surface a clearer failure state.",
            }
        )
        return payload

    def preview_query(self, user_query: str) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been configured.")

        if hasattr(self.pipeline, "preview"):
            try:
                preview = self.pipeline.preview(user_query)
            except Exception as exc:
                return self._preview_error_payload(exc)

            payload = self._build_prompt_payload(
                preview["route"],
                preview["cloud_prompt"],
                editable=preview["route"] == "protected",
            )
            payload.update(
                {
                    "route": preview["route"],
                    "route_reason": preview["route_reason"],
                    "editable": preview["route"] == "protected",
                    "structured_fields": preview["structured_fields"],
                    "detected_pii": preview["detected_pii"],
                    "redacted_query": preview["redacted_query"],
                    "placeholder_map": preview["placeholder_map"],
                    "detector_available": preview["detector_available"],
                    "detector_uncertain": preview["detector_uncertain"],
                    "detector_error": preview["detector_error"],
                    "preview_error": None,
                }
            )
            return payload

        prompt = self.pipeline.prompt_creater(userQuery=user_query).createdPrompt
        payload = self._build_prompt_payload("protected", prompt, editable=True)
        payload.update(
            {
                "route": "protected",
                "route_reason": "legacy_pipeline",
                "editable": True,
                "structured_fields": {},
                "detected_pii": [],
                "redacted_query": "",
                "placeholder_map": {},
                "detector_available": True,
                "detector_uncertain": False,
                "detector_error": None,
                "preview_error": None,
            }
        )
        return payload

    def process_query(self, final_input: FinalInput) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been configured.")

        route = final_input.route or "protected"
        cloud_prompt = final_input.edited_prompt or final_input.original_prompt

        if hasattr(self.pipeline, "run_with_prompt"):
            prediction = self.pipeline.run_with_prompt(
                final_input.original_query,
                None if route == "direct" else cloud_prompt,
            )
        else:
            llm_response = self.pipeline.untrusted_model(cloud_prompt)[0]
            final_output = self.pipeline.info_aggregator(
                userQuery=final_input.original_query,
                modelExampleResponses=llm_response,
            ).finalOutput
            prediction = SimpleNamespace(
                output=final_output,
                route="protected",
                cloud_prompt=cloud_prompt,
                structured_fields={},
                detected_pii=[],
            )

        edit_record = None
        if getattr(prediction, "route", route) == "protected":
            edit_record = self.record_edit(
                final_input.original_prompt,
                final_input.edited_prompt or final_input.original_prompt,
                datetime.now().isoformat(),
            )

        resolved_route = getattr(prediction, "route", route)
        resolved_cloud_prompt = getattr(prediction, "cloud_prompt", cloud_prompt)
        payload = self._build_prompt_payload(resolved_route, resolved_cloud_prompt)
        payload.update(
            {
                "output": prediction.output,
                "route": resolved_route,
                "structured_fields": getattr(prediction, "structured_fields", {}),
                "detected_pii": getattr(prediction, "detected_pii", []),
                "edit_record": edit_record,
            }
        )
        return payload

    def record_edit(self, original_prompt: str, edited_prompt: str, timestamp: str) -> dict:
        edit = {
            "timestamp": timestamp,
            "original": original_prompt,
            "edited": edited_prompt,
            "diff_length": len(edited_prompt) - len(original_prompt),
        }
        self.edit_history.append(edit)
        return edit


runtime = PipelineRuntime()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/generate_prompt")
async def generate_prompt(query: Query):
    try:
        preview = runtime.preview_query(query.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=preview)


@app.post("/process_prompt")
async def process_prompt(final_input: FinalInput):
    try:
        result = runtime.process_query(final_input)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model", default=3012)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--prompt_file", type=str, default="ORIGINAL", help="The DSPy-optimized prompt, stored as a json file")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--server_port", type=int, help="Where you are hosting your SERVER, not models", default=8012)
    parser.add_argument("--pipeline", type=str, choices=["legacy", "structured_v1"], default="structured_v1")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", type=str, default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    resolved_prompt_file = args.prompt_file
    if resolved_prompt_file == "ORIGINAL":
        resolved_prompt_file = parse_model_prompt(args.model_name) if args.pipeline == "legacy" else None

    local_lm = dspy.LM(
        f"openai/{args.model_name}",
        api_base=f"http://0.0.0.0:{args.port}/v1",
        api_key="",
        max_tokens=4000,
    )
    dspy.configure(lm=local_lm)

    openai_lm = build_openai_lm(args.openai_model, max_tokens=4000)
    pipeline = build_pipeline(
        pipeline_name=args.pipeline,
        untrusted_model=openai_lm,
        allow_direct_bypass=args.allow_direct_bypass,
        privacy_filter_name=args.privacy_filter,
        pii_score_threshold=args.pii_score_threshold,
    )

    if resolved_prompt_file:
        prompt_path = Path(resolved_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = PROJECT_ROOT / "papillon" / prompt_path
        pipeline.load(str(prompt_path), use_legacy_loading=True)

    runtime.configure(pipeline)

    print("Starting FastAPI server...")
    print(f"You can access it at: http://127.0.0.1:{args.server_port}")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.server_port)

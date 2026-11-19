from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agan.hardware.detect import ProfileSpec
from agan.schemas import Answer


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "agan"


def create_app(agan: "AGAN") -> FastAPI:
    app = FastAPI(title="AGAN API", version="0.0.1")

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [ModelInfo(id=agan.model).model_dump()]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "No messages")
        last = req.messages[-1]["content"]
        from agan.schemas import Answer
        ans: Answer = await agan.answer(last)
        return {
            "id": "chatcmpl-agan",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ans.text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.get("/doctor")
    async def doctor():
        from agan.hardware.detect import get_spec
        prof: ProfileSpec = get_spec(agan.profile)
        return {
            "profile": prof.name,
            "ram_gb": round(prof.ram_gb_min + (prof.ram_gb_max - prof.ram_gb_min) / 2, 1),
            "gpu": prof.gpu_name if hasattr(prof, 'gpu_name') else None,
            "model": agan.model,
            "time_budget": prof.time_budget,
            "best_of_n": prof.best_of_n,
        }

    return app
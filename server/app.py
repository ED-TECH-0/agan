"""
FastAPI server — DESKHUB connector
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from core.pipeline import Pipeline, PipelineConfig
from core.governor import ProfileName

app = FastAPI(title="agan API", version="0.1.0")

pipeline = Pipeline(PipelineConfig(profile=ProfileName.STANDARD))

class QuestionRequest(BaseModel):
    question: str
    profile: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    profile: str

@app.get("/health")
async def health():
    return {"status": "ok", "service": "agan"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    try:
        profile = ProfileName(request.profile) if request.profile else ProfileName.STANDARD
        # Update pipeline config if profile changed
        pipeline.config.profile = profile
        answer = await pipeline.answer(request.question)
        return AnswerResponse(answer=answer, profile=profile.value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/doctor")
async def doctor(profile: str = "standard", report: bool = False):
    from core.doctor import BrainDoctor
    doc = BrainDoctor()
    report_obj = doc.run(profile, redact=not report)
    return {"report": doc.format_report(report_obj)}

@app.get("/profiles")
async def list_profiles():
    return {"profiles": [p.value for p in ProfileName]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

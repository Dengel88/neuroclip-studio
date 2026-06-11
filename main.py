import os
import logging
import secrets
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from dotenv import load_dotenv

from google import genai
from google.genai import types

from models import (
    ScriptwriterInput, ScriptwriterOutput, 
    StoryboarderInput, StoryboarderOutput,
    PromptEngineerInput, PromptEngineerOutput,
    VideoEditInput, VideoEditOutput
)

# Configure logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeuroclipStudio")

load_dotenv()
app = FastAPI(title="Neuroclip Studio - AI Video Production Platform")

# ==========================================
# 🔒 SECURITY: HTTP BASIC AUTHENTICATION
# ==========================================
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Validates username and password against hardcoded environment limits."""
    correct_username = secrets.compare_digest(credentials.username, "google")
    correct_password = secrets.compare_digest(credentials.password, "omni2026")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ==========================================
# 🧠 CORE AI ROUTER: API KEY ROTATION
# ==========================================
def get_valid_keys():
    keys = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2"), os.getenv("GEMINI_API_KEY_3")]
    return [k for k in keys if k and k.strip() != ""]

def call_gemini_with_rotation(model_name: str, contents: str, config: types.GenerateContentConfig):
    keys = get_valid_keys()
    if not keys:
        raise HTTPException(status_code=500, detail="No valid Gemini API keys found.")

    last_error = None
    for i, key in enumerate(keys):
        try:
            logger.info(f"Router: Attempting execution with API Key #{i+1}")
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            logger.info(f"Router: Execution successful with API Key #{i+1}")
            return response.text
        except Exception as e:
            logger.warning(f"Router: API Key #{i+1} failed. Switching to fallback...")
            last_error = e
            continue
            
    logger.error("Router: All API keys exhausted.")
    raise HTTPException(status_code=502, detail=f"All API keys exhausted. Last error: {str(last_error)}")

# ==========================================
# 🌐 ROUTES
# ==========================================
@app.get("/", response_class=HTMLResponse)
def read_root(username: str = Depends(verify_credentials)):
    """Serves the UI. Protected by Basic Auth."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/generate-concepts", response_model=ScriptwriterOutput)
def generate_concepts(request: ScriptwriterInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI Video Director. Analyze the user's video request and propose 3 creative concepts.\n"
        "CRITICAL VISUAL CONSTRAINT: Avoid small readable text or complex UI screens natively. Focus on cinematic metaphors.\n"
        "Output must be strictly JSON, containing exactly 3 concepts."
    )
    user_prompt = f"Format: {request.video_format}\nDuration: {request.total_duration}s\nGoal: {request.business_goal}\nAudience: {request.target_audience}\nTopic: {request.topic_idea}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=ScriptwriterOutput, temperature=0.7)
    return ScriptwriterOutput.model_validate_json(call_gemini_with_rotation('gemini-3.5-flash', user_prompt, config))

@app.post("/api/generate-storyboard", response_model=StoryboarderOutput)
def generate_storyboard(request: StoryboarderInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an expert AI Video Storyboarder.\n"
        "1. HARD CUTS ONLY: Every scene is an independent visual shot. DO NOT create continuous unbroken shots.\n"
        "2. The sum of all scene durations MUST exactly match the requested total duration.\n"
        "3. Omni video models ONLY generate clips of 4, 6, 8, or 10 seconds. Use ONLY these durations."
    )
    user_prompt = f"Title: {request.working_title}\nConcept: {request.logline}\nStyle: {request.visual_style}\nPacing: {request.pacing}\nDuration: {request.total_duration}s"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=StoryboarderOutput, temperature=0.4)
    return StoryboarderOutput.model_validate_json(call_gemini_with_rotation('gemini-3.5-flash', user_prompt, config))

@app.post("/api/generate-prompts", response_model=PromptEngineerOutput)
def generate_prompts(request: PromptEngineerInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI Video Prompt Engineer.\n"
        "DECISION: TEXT-TO-VIDEO vs IMAGE-TO-VIDEO\n"
        "- 'image-to-video': For strict character consistency or macro shots. Write an Image Prompt for Nano Bana Pro.\n"
        "- 'text-to-video': For highly dynamic scenes. Leave 'image_prompt' empty.\n"
        "OMNI TEMPLATE: Format strictly: Camera: []. Style: []. Lighting: []. Scene: []. Action & Audio: []. Text: [].\n"
        "DURATION CONSTRAINT: Set 'omni_duration' to 4, 6, 8, or 10."
    )
    scenes_text = "\n".join([f"Scene {s.scene_number}: {s.visual_description} | Target Duration: {s.duration}s" for s in request.scenes])
    user_prompt = f"Overall Style: {request.visual_style}\n\nScenes:\n{scenes_text}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=PromptEngineerOutput, temperature=0.4)
    return PromptEngineerOutput.model_validate_json(call_gemini_with_rotation('gemini-3.5-flash', user_prompt, config))

@app.post("/api/format-edit-prompt", response_model=VideoEditOutput)
def format_edit_prompt(request: VideoEditInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI VFX Supervisor.\n"
        "Rewrite the user's edit request following the 5 Omni Rules: 1. Anchor Rule, 2. Before->After, 3. Layering, 4. Time-tagging, 5. Audio sync.\n"
        "Output ONLY the final, polished edit prompt."
    )
    user_prompt = f"Original Prompt: {request.original_prompt}\nUser Edit Request: {request.user_request}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=VideoEditOutput, temperature=0.3)
    return VideoEditOutput.model_validate_json(call_gemini_with_rotation('gemini-3.5-flash', user_prompt, config))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
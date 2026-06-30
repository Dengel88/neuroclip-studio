import os
import time
import logging
import secrets
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("NeuroclipStudio")

load_dotenv()
app = FastAPI(title="Neuroclip Studio - AI Video Production Platform")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "google")
    correct_password = secrets.compare_digest(credentials.password, "omni2026")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def get_valid_keys():
    keys = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2"), os.getenv("GEMINI_API_KEY_3")]
    return [k for k in keys if k and k.strip() != ""]

def call_gemini_with_rotation(model_name: str, contents: str, config: types.GenerateContentConfig, max_retries: int = 2):
    keys = get_valid_keys()
    if not keys:
        raise HTTPException(status_code=500, detail="No valid Gemini API keys found.")

    last_error = None
    fallback_model = 'gemini-3.5-flash' if 'pro' in model_name else model_name

    for i, key in enumerate(keys):
        client = genai.Client(api_key=key)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Router: Attempting {model_name} on Key #{i+1} (Try {attempt+1}/{max_retries})")
                response = client.models.generate_content(model=model_name, contents=contents, config=config)
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                
                if "503" in error_msg or "429" in error_msg:
                    logger.warning(f"Router: {model_name} failed. WAITING 2 SECONDS before fallback...")
                    time.sleep(2)  
                    
                    if fallback_model != model_name:
                        try:
                            logger.info(f"Router: Firing fallback {fallback_model} on Key #{i+1}...")
                            response = client.models.generate_content(model=fallback_model, contents=contents, config=config)
                            return response.text
                        except Exception as fallback_e:
                            logger.error(f"Router: Fallback also failed: {str(fallback_e)}")
                            last_error = fallback_e
                            time.sleep(1)
                    else:
                        last_error = e
                else:
                    logger.error(f"Router: Fatal error on Key #{i+1}: {error_msg}")
                    last_error = e
                    break 
                    
    logger.error("Router: ALL keys and fallbacks exhausted.")
    raise HTTPException(status_code=502, detail=f"All API keys exhausted. Last error: {str(last_error)}")

def generate_image_with_retry(prompt_text: str, aspect_ratio: str, max_retries: int = 1):
    keys = get_valid_keys()
    last_error = None
    
    models_to_try = ['imagen-3.0-generate-001', 'gemini-3-pro-image', 'imagen-4.0-generate-001']

    for model_name in models_to_try:
        for i, key in enumerate(keys):
            client = genai.Client(api_key=key)
            for attempt in range(max_retries):
                try:
                    logger.info(f"🎨 Image Gen: Attempting {model_name} on Key #{i+1} with ratio {aspect_ratio}...")
                    result = client.models.generate_images(
                        model=model_name,
                        prompt=prompt_text,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            output_mime_type="image/jpeg",
                            aspect_ratio=aspect_ratio
                        )
                    )
                    
                    if hasattr(result, 'generated_images') and result.generated_images:
                         logger.info(f"🎨 Image Gen: Successfully generated using {model_name}!")
                         return result
                except Exception as e:
                    logger.warning(f"🎨 Image Gen ERROR on {model_name} ({type(e).__name__}): {e}")
                    time.sleep(1)
                    last_error = e
                    
    logger.error("🎨 Image Gen: ALL models and keys exhausted.")
    return None

@app.get("/", response_class=HTMLResponse)
def read_root(username: str = Depends(verify_credentials)):
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/generate-concepts", response_model=ScriptwriterOutput)
def generate_concepts(request: ScriptwriterInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI Video Director. Analyze the user's video request and generate the exact conceptual breakdown requested (e.g., specific number of episodes, formats, or ideas).\n"
        "CRITICAL VISUAL CONSTRAINT: Avoid small readable text or complex UI screens natively. Focus on cinematic metaphors.\n"
        "Output must be strictly JSON format, dynamically adapting to the requested output structure."
    )
    user_prompt = f"Format: {request.video_format}\nDuration: {request.total_duration}s\nGoal: {request.business_goal}\nAudience: {request.target_audience}\nTopic: {request.topic_idea}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=ScriptwriterOutput, temperature=0.7)
    
    return ScriptwriterOutput.model_validate_json(call_gemini_with_rotation('gemini-3.1-pro-preview', user_prompt, config))

@app.post("/api/generate-storyboard", response_model=StoryboarderOutput)
def generate_storyboard(request: StoryboarderInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an expert AI Video Storyboarder.\n"
        "1. HARD CUTS ONLY: Every scene is an independent visual shot. DO NOT create continuous unbroken shots.\n"
        "2. The sum of all scene durations MUST exactly match the requested total duration.\n"
        "3. Veo Lite video models ONLY generate clips of 4, 6, or 8 seconds. Use ONLY these durations.\n"
        "4. Output must be strictly valid JSON matching the response schema without any conversational text."
    )
    user_prompt = f"Title: {request.working_title}\nConcept: {request.logline}\nStyle: {request.visual_style}\nPacing: {request.pacing}\nDuration: {request.total_duration}s"
    config = types.GenerateContentConfig(
        system_instruction=system_instruction, 
        response_mime_type="application/json", 
        response_schema=StoryboarderOutput, 
        temperature=0.4
    )
    
    raw_response = call_gemini_with_rotation('gemini-3.1-pro-preview', user_prompt, config)
    
    if raw_response.strip().startswith("I") or raw_response.strip().startswith("Internal"):
        logger.error(f"Storyboarder generated invalid text response: {raw_response}")
        raise HTTPException(
            status_code=500, 
            detail="Agent 2 encountered an internal constraint error. Please try again or slightly simplify the scene requirements."
        )
        
    return StoryboarderOutput.model_validate_json(raw_response)

@app.post("/api/generate-prompts", response_model=PromptEngineerOutput)
def generate_prompts(request: PromptEngineerInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI Video Prompt Engineer.\n"
        "DECISION: TEXT-TO-VIDEO vs IMAGE-TO-VIDEO\n"
        "- 'image-to-video': For strict character consistency or macro shots. Write an Image Prompt for Nano Bana Pro.\n"
        "- 'text-to-video': For highly dynamic scenes. Leave 'image_prompt' empty.\n"
        "VEO LITE TEMPLATE: Format strictly: Camera: []. Style: []. Lighting: []. Scene: []. Action & Audio: []. Text: [].\n"
        "DURATION CONSTRAINT: Set generation duration strictly to 4, 6, or 8."
    )
    scenes_text = "\n".join([f"Scene {s.scene_number}: {s.visual_description} | Target Duration: {s.duration}s" for s in request.scenes])
    user_prompt = f"Target Video Format: {request.video_format}\nOverall Style: {request.visual_style}\n\nScenes:\n{scenes_text}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=PromptEngineerOutput, temperature=0.4)
    
    result_text = call_gemini_with_rotation('gemini-3.1-pro-preview', user_prompt, config)
    output_obj = PromptEngineerOutput.model_validate_json(result_text)
    
    for prompt in output_obj.prompts:
        if prompt.generation_type == 'image-to-video' and prompt.image_prompt:
            image_result = generate_image_with_retry(prompt.image_prompt, request.video_format)
            
            if image_result and hasattr(image_result, 'generated_images') and image_result.generated_images:
                img_filename = f"scene_{prompt.scene_number}_{secrets.token_hex(4)}.jpg"
                
                os_file_path = os.path.join("static", img_filename)
                url_path = f"/static/{img_filename}"
                
                try:
                    with open(os_file_path, "wb") as f:
                        f.write(image_result.generated_images[0].image.image_bytes)
                    
                    if os.path.exists(os_file_path):
                        prompt.image_url = url_path
                except Exception as e:
                    logger.error(f"❌ Failed to save image: {e}")

    return output_obj

@app.post("/api/format-edit-prompt", response_model=VideoEditOutput)
def format_edit_prompt(request: VideoEditInput, username: str = Depends(verify_credentials)):
    system_instruction = (
        "You are an elite AI VFX Supervisor.\n"
        "Rewrite the user's edit request following the 5 Veo Lite Rules: 1. Anchor Rule, 2. Before->After, 3. Layering, 4. Time-tagging, 5. Audio sync.\n"
        "Output ONLY the final, polished edit prompt."
    )
    user_prompt = f"Original Prompt: {request.original_prompt}\nUser Edit Request: {request.user_request}"
    config = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=VideoEditOutput, temperature=0.3)
    
    return VideoEditOutput.model_validate_json(call_gemini_with_rotation('gemini-3.1-pro-preview', user_prompt, config))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

# 🎬 Neuroclip Studio
**An Autonomous AI Multi-Agent Video Production Platform powered by Google Gemini.**

🚀 **Live Demo:** [neuroclipstudio.com](https://neuroclipstudio.com)  
🎥 **Demo Video:** [Presentetion](https://drive.google.com/file/d/1mUTNLedQR70cvlbA1jGIv-Qr3KV72h7U/view?usp=sharing)

🔒 **Testing Access Credentials:**
To prevent API abuse, the live demo is protected by Basic Auth.
- **Username:** `google`
- **Password:** `omni2026`

Developed for the **Google AI Agent Hackathon**, Neuroclip Studio is an enterprise-grade agentic pipeline that transforms a simple text brief into a production-ready, highly technical cinematic storyboard using an orchestrated team of 4 specialized AI Agents.

## 🏗️ Architecture Diagram

```mermaid
graph TD
    User([User Brief]) --> UI[Neuroclip Studio UI]
    
    UI -->|Step 1| A1[Agent 1: Scriptwriter<br>Gemini 1.5 Pro]
    A1 -->|Strict JSON Concepts| UI
    
    UI -->|Step 2| A2[Agent 2: Storyboarder<br>Gemini 1.5 Flash]
    A2 -->|Math Constraints & Hard Cuts| UI
    
    UI -->|Step 3| A3[Agent 3: Prompt Engineer<br>Gemini 1.5 Flash]
    A3 --> Router{Router Logic}
    Router -->|Text-to-Video| T2V[6-Dimension Omni Prompt]
    Router -->|Image-to-Video| I2V[Nano Bana Pro Image Prompt <br>+ Omni Video Prompt]
    T2V --> UI
    I2V --> UI
    
    UI -->|Step 4: User Edits| A4[Agent 4: VFX Supervisor<br>Gemini 1.5 Flash]
    A4 -->|5-Rule Optimized Edit Prompt| UI

---

## 🧠 Core Architecture & Agentic Workflow

Neuroclip Studio replaces the chaotic traditional video production pipeline with a sequential, human-in-the-loop Agentic Workflow. We leverage the blazing speed and reasoning capabilities of **Gemini 3.5 Flash**.

All inter-agent communication is strictly validated using **Pydantic Models** (Structured JSON Outputs) to ensure zero hallucination and perfect data flow across the pipeline.

### The 4 AI Agents:

1. **🎭 Agent 1: The Scriptwriter**
   - Ingests the user's brief (Format, Duration, Goal, Audience, Core Idea).
   - Generates 3 distinct, highly visual concepts formatted as strict JSON.
   - *Constraint applied:* Avoids text-heavy concepts that video diffusion models struggle to render, focusing on cinematic metaphors.

2. **🎞️ Agent 2: The Storyboarder**
   - Takes the selected concept and breaks it down into a precise shot-by-shot storyboard.
   - *Mathematical Constraint:* Ensures the sum of all scene durations perfectly matches the requested total duration, strictly using chunks supported by Omni (4, 6, 8, or 10 seconds).
   - *Directing Rule:* Applies "Hard Cuts Only" to prevent continuous motion breakage between separately generated video clips.

3. **🎥 Agent 3: The Prompt Engineer**
   - Translates raw scene descriptions into highly technical **6-dimension prompts** (Camera, Style, Lighting, Scene, Action & Audio, Text).
   - *Dynamic Router Logic:* Automatically decides between `text-to-video` and `image-to-video` depending on focal complexity. If high fidelity is needed, it dynamically writes an Image Prompt for **Nano Bana Pro** to be used as a reference frame.

4. **🪄 Agent 4: The VFX Supervisor**
   - Handles the Human-in-the-Loop editing system.
   - Takes raw user edit requests (e.g., "Make it rain") and reframes them using **Omni's strict 5 Rules of Editing** (Anchor Rule, Before->After, Layering, Time-tagging, Audio Sync) to prevent the diffusion model from hallucinating or breaking the original composition.

---

## 💎 Enterprise Features

- **API Key Rotation & Load Balancing:** Built-in router that catches `429 Resource Exhausted` or `503 Service Unavailable` errors and seamlessly falls back to backup API keys to ensure uninterrupted service.
- **Copy-to-Clipboard Flow:** UX optimized for rapid prompt transfer to external generative video engines.
- **Strict Data Validation:** 100% Pydantic schema enforcement on every LLM response.

---

## ⚠️ Hackathon Scope & Limitations (Honesty Badge)

**Note on Video Rendering:**
The core focus of our submission is the **Agentic Prompt Engineering pipeline** (Agents 1 through 4) powered by the Gemini API. 

Because generating high-fidelity video via native diffusion models takes significant time and requires background task queues (like Celery/Redis) which exceed the hackathon's prototyping scope, the final "Generate Video" button currently triggers a **simulated UI response**. The frontend mocks the asynchronous webhook flow and displays a pre-rendered placeholder video to demonstrate UX state handling. 

*The Agentic brain is 100% real and dynamic; the final rendering muscle is mocked for demonstration purposes.*

---

## 🚀 How to Run Locally

1. Clone the repository and navigate to the folder:
```bash
git clone https://github.com/your-username/neuroclip-studio.git

1.Create a virtual environment and install dependencies:

python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

1.Add your Google AI Studio keys:
Create a .env file in the root directory. You can add multiple keys to test the API rotation feature:
GEMINI_API_KEY_1=your_primary_api_key_here
GEMINI_API_KEY_2=your_backup_api_key_here

1.Run the application:
python main.py
Open your browser and navigate to http://127.0.0.1:8000.

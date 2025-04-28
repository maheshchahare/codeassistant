import json
import time
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
import re
import traceback
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Code Assistant Bot",
    description="An API that helps developers write clean, maintainable, modular code using Ollama LLMs",
    version="1.0.0",
)

# Configure Ollama endpoint
OLLAMA_API_BASE = "http://localhost:11434/api"
DEFAULT_MODEL = "codellama"  # Change this to a model you have installed


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Pydantic models for request/response
class CodeRequest(BaseModel):
    code: str
    language: str
    task: str = "improve"  # improve, document, refactor, optimize, debug
    context: Optional[str] = None
    detailed_response: bool = False
    model: Optional[str] = None


class NaturalLanguageQuery(BaseModel):
    query: str
    language: Optional[str] = None
    task: Optional[str] = None
    model: Optional[str] = None
    sessionId: Optional[str] = None
    includeHistory: bool = True  # New field to control history inclusion
    historyLimit: int = 10  # Limit number of previous messages


class CodeSuggestion(BaseModel):
    improved_code: str
    explanation: str
    suggestions: List[str]


class CodeResponse(BaseModel):
    original_code: str
    suggestions: CodeSuggestion


# Task-specific prompts
TASK_PROMPTS = {
    "improve": "Improve this code to make it cleaner, more maintainable, and follow best practices.",
    "document": "Add appropriate documentation to this code following best practices.",
    "refactor": "Refactor this code to improve its structure while preserving functionality.",
    "optimize": "Optimize this code for better performance while maintaining readability.",
    "debug": "Identify and fix potential bugs or issues in this code.",
    "query": "Answer the following programming question or help with the code task.",
}


# Check if Ollama is available
async def check_ollama_available():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_API_BASE}/tags", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


# async def query_ollama(prompt: str, model: Optional[str] = None) -> str:
#     """Send a prompt to Ollama API and return the response."""
#     if model is None or model.strip() == "":
#         model = DEFAULT_MODEL

#     async with httpx.AsyncClient() as client:
#         try:
#             print(f"Model: {model}")
#             print(f"Prompt: {prompt}")
#             print(f"")
#             response = await client.post(
#                 f"{OLLAMA_API_BASE}/generate",
#                 json={"model": model, "prompt": prompt, "stream": False},
#                 timeout=60.0,
#             )
#             response.raise_for_status()
#             result = response.json()
#             return result.get("response", "")
#         except httpx.HTTPStatusError as e:
#             if e.response.status_code == 404:
#                 # Model not found error
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Model '{model}' not found. Make sure to pull it with 'ollama pull {model}'",
#                 )
#             else:
#                 raise HTTPException(
#                     status_code=e.response.status_code,
#                     detail=f"Ollama API error: {str(e)}",
#                 )
#         except httpx.RequestError as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Error communicating with Ollama: {str(e)}. Make sure Ollama is running.",
#             )


async def query_ollama(prompt: str, model: str = "codellama") -> str:
    """Send a prompt to Ollama API and return the response."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_API_BASE}/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=f"Ollama API error: {e}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500, detail=f"Error communicating with Ollama: {e}"
            )


def save_sessions(sessions_data):
    """Save sessions data to storage"""
    try:
        with open("sessions.json", "w") as f:
            json.dump(sessions_data, f)
    except Exception as e:
        print(f"Error saving sessions: {e}")


def load_sessions():
    """Load sessions data from storage"""
    try:
        if not Path("sessions.json").exists():
            return []
        with open("sessions.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading sessions: {e}")
        return []


def get_chat_history(session_id: str, limit: int = 10) -> list:
    """Retrieve recent chat history for context"""
    try:
        # This implementation will depend on how you're storing sessions
        # For now we'll use a simple file-based approach
        sessions_file = Path("sessions.json")
        if not sessions_file.exists():
            return []

        with open(sessions_file, "r") as f:
            sessions = json.load(f)

        # Find the session
        session = next((s for s in sessions if s["id"] == session_id), None)
        if not session:
            print(f"Warning: Session ID {session_id} not found in sessions.json")
            return []

        if "messages" not in session:
            print(f"Warning: No messages in session {session_id}")
            return []

        # Format messages for LLM context - last 'limit' messages
        messages = session["messages"][-limit:]
        return [
            {
                "role": "user" if msg["isUser"] else "assistant",
                "content": strip_html_tags(msg["content"]),
            }
            for msg in messages
        ]

    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text to clean it for LLM input"""
    # Simple regex to remove HTML tags
    return re.sub(r"<[^>]+>", "", text)


def create_nl_prompt(
    query: str, language: str = None, task: str = None, history: list = None
) -> str:
    """Craft a natural language programming prompt including chat history"""
    task_description = ""
    history_context = ""

    if history and len(history) > 0:
        history_formatted = "\n".join(
            [
                f"User: {msg['content'] if msg['role'] == 'user' else ''}"
                f"Assistant: {msg['content'] if msg['role'] == 'assistant' else ''}"
                for msg in history
            ]
        )
        history_context = f"\nChat History:\n{history_formatted}\n"

    # Rest of function as before
    if task and task in TASK_PROMPTS:
        task_description = f"The user wants to {TASK_PROMPTS[task].lower()}"

    return f"""As an expert programmer, your task is to {task_description}

Focus on making the code more:
- Readable and maintainable
- Modular with proper separation of concerns
- Following best practices for {language} language
- Free of code smells and anti-patterns
- Well-documented when needed
{history_context}

Current Question:
{query}
Give code snippet to explain more clearly enclosed in ``````.

Please provide a clear, accurate, and helpful response.
"""


def create_code_prompt(
    code: str, language: str, task: str, context: Optional[str] = None
) -> str:
    """Create a prompt for the LLM to process code."""
    task_description = TASK_PROMPTS.get(task, TASK_PROMPTS["improve"])

    prompt = f"""As an expert programmer, your task is to {task_description}
    
Code Language: {language}

{f"Context: {context}" if context else ""}

Original Code:
```{language}
{code}
```

Please provide:
1. An improved version of the code
2. A clear explanation of the changes made
3. Three specific recommendations for further improvement

Format your response in JSON with the following structure:
{{
    "improved_code": "your improved code here",
    "explanation": "your explanation here",
    "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}

Focus on making the code more:
- Readable and maintainable
- Modular with proper separation of concerns
- Following best practices for {language}
- Free of code smells and anti-patterns
- Well-documented when needed
"""
    return prompt


def extract_json_from_text(text: str) -> dict:
    """Extract JSON object from text which might contain other content."""
    # Try to find JSON object in the text
    json_match = re.search(r"({[\s\S]*})", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback for when the model doesn't strictly follow JSON format
    try:
        # Try to extract sections using regex patterns
        improved_code_match = re.search(r"```[\w]*\n([\s\S]*?)\n```", text)
        improved_code = improved_code_match.group(1) if improved_code_match else ""

        explanation_pattern = r"(?:explanation|explanation:)([\s\S]*?)(?:suggestions|$)"
        explanation_match = re.search(explanation_pattern, text, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        suggestions = []
        suggestions_text = re.search(r"suggestions:?([\s\S]*?)$", text, re.IGNORECASE)
        if suggestions_text:
            suggestions_raw = suggestions_text.group(1)
            # Extract numbered or bulleted items
            suggestions = re.findall(
                r"(?:^\d+\.|\*)\s*(.*?)(?=^\d+\.|\*|$)", suggestions_raw, re.MULTILINE
            )
            if not suggestions:
                # Split by newlines as fallback
                suggestions = [
                    s.strip() for s in suggestions_raw.split("\n") if s.strip()
                ]

        return {
            "improved_code": improved_code,
            "explanation": explanation,
            "suggestions": suggestions[:3],  # Limit to 3 suggestions
        }
    except Exception:
        # Return error structure if parsing fails completely
        return {
            "improved_code": "Failed to parse code improvements",
            "explanation": "The AI response couldn't be properly parsed.",
            "suggestions": [
                "Try again with a simpler code snippet",
                "Ensure code is properly formatted",
                "Try a different task type",
            ],
        }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def api_info():
    """Root endpoint with API information."""
    return {
        "name": "Code Assistant Bot",
        "version": "1.0.0",
        "endpoints": {
            "/improve-code": "Get suggestions to improve your code",
            "/query": "Ask programming-related questions",
            "/models": "List available LLM models",
            "/health": "Check API health",
        },
    }


# In main.py - Add session management endpoints
@app.post("/sessions")
async def create_session():
    """Create a new chat session"""
    session_id = f"session_{int(time.time() * 1000)}"
    new_session = {
        "id": session_id,
        "name": f"Chat {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}",
        "messages": [],
    }

    sessions = load_sessions()
    sessions.insert(0, new_session)
    save_sessions(sessions)

    return {"id": session_id, "name": new_session["name"]}


@app.put("/sessions/{session_id}/messages")
async def save_session_messages(session_id: str, messages: List[dict]):
    """Save messages for a session"""
    sessions = load_sessions()
    session_index = next(
        (i for i, s in enumerate(sessions) if s["id"] == session_id), -1
    )

    if session_index == -1:
        raise HTTPException(status_code=404, detail="Session not found")

    sessions[session_index]["messages"] = messages
    save_sessions(sessions)

    return {"status": "success"}


@app.post("/improve-code", response_model=CodeResponse)
async def improve_code(request: CodeRequest):
    """Analyze code and provide suggestions for improvement."""
    try:
        # Check if Ollama is available
        if not await check_ollama_available():
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama. Make sure it's running on http://localhost:11434",
            )

        # Create prompt
        prompt = create_code_prompt(
            request.code, request.language, request.task, request.context
        )

        # Query the LLM
        llm_response = await query_ollama(prompt, request.model)

        # Process LLM response
        parsed_response = extract_json_from_text(llm_response)

        return CodeResponse(
            original_code=request.code,
            suggestions=CodeSuggestion(
                improved_code=parsed_response.get("improved_code", ""),
                explanation=parsed_response.get("explanation", ""),
                suggestions=parsed_response.get("suggestions", []),
            ),
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Log the full error
        error_details = traceback.format_exc()
        print(f"Error in improve_code: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.get("/models")
async def list_models():
    """List available models from Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_API_BASE}/tags", timeout=5.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            return {"models": [model.get("name") for model in models]}
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code, detail=f"Ollama API error: {str(e)}"
        )
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": f"Could not fetch models: {str(e)}"}


@app.post("/query")
async def natural_language_query(request: NaturalLanguageQuery):
    """Handle natural language queries from users."""
    try:
        if not await check_ollama_available():
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama. Make sure it's running on http://localhost:11434",
            )

        # Get chat history if requested
        history = []
        if request.includeHistory and request.sessionId:
            history = get_chat_history(request.sessionId, request.historyLimit)

        print(f"Session ID: {request.sessionId}")
        print(f"History: {history}")

        # Create prompt based on language and task preferences
        prompt = create_nl_prompt(
            request.query, request.language, request.task, history
        )

        print("Prompt Sent:", prompt)

        response = await query_ollama(prompt, request.model)

        print("Raw Response from Ollama:", response)
        # Add basic tone detection logic
        tone = "technical"
        query_lower = request.query.lower()
        if (
            query_lower in ["hi", "hello", "hey"]
            or "how are you" in query_lower
            or "what's up" in query_lower
            or query_lower.startswith("can you")
            or len(query_lower.split()) < 5
        ):
            tone = "conversation"

        if request.sessionId:
            sessions = load_sessions()

            # Check if session exists, create if not
            session_index = next(
                (i for i, s in enumerate(sessions) if s["id"] == request.sessionId), -1
            )

            # if session_index != -1:
            #     sessions[session_index]["messages"].append(
            #         {"isUser": True, "content": request.query}
            #     )
            #     sessions[session_index]["messages"].append(
            #         {"isUser": False, "content": response.strip()}
            #     )
            #     save_sessions(sessions)

            if session_index == -1:
                print(f"Warning: Session {request.sessionId} not found, creating it")
                # Create session if it doesn't exist
                new_session = {
                    "id": request.sessionId,
                    "name": f"Chat {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}",
                    "messages": [],
                }
                sessions.append(new_session)
                session_index = len(sessions) - 1

            # Add messages
            if "messages" not in sessions[session_index]:
                sessions[session_index]["messages"] = []

            sessions[session_index]["messages"].append(
                {"isUser": True, "content": request.query}
            )
            sessions[session_index]["messages"].append(
                {"isUser": False, "content": response.strip()}
            )
            save_sessions(sessions)

        print(f"Request: {request}")
        print(f"Response: {response}")

        return {
            "query": request.query,
            "response": response.strip(),
            "tone": tone,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in natural_language_query: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """Check if the API and Ollama service are working."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_API_BASE}/tags", timeout=5.0)
            response.raise_for_status()
            return {"status": "healthy", "ollama": "connected"}
    except Exception:
        return {"status": "healthy", "ollama": "disconnected"}


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    return JSONResponse(
        status_code=500, content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

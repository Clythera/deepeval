"""Main FastAPI application for the LLM support assistant."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import logging

from app.llm import ask_llm
from app.models import QuestionRequest, AnswerResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Support Assistant",
    description="A simple REST API for asking questions to an LLM",
    version="1.0.0"
)

# Add CORS middleware (optional, for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint that does not make an LLM call."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Ask a question to the LLM and return the answer."""
    try:
        # Get the answer from the LLM
        answer = await ask_llm(request.question)
        
        # Return the response with the original question and LLM answer
        return AnswerResponse(
            question=request.question,
            answer=answer
        )
        
    except Exception as e:
        # Log the error for debugging but don't expose internal details
        logger.error(f"Error processing question: {str(e)}")
        
        # Return appropriate HTTP error codes
        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM request timed out"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to get response from LLM"
            )


# For development, allow running the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
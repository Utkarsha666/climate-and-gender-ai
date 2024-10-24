from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from question_service import process_question, QuestionRequest
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/question")
def ask_question(request: QuestionRequest):
    return {"answer": process_question(request.user_input)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

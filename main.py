from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from huggingface_hub import InferenceClient
from pathlib import Path

app = FastAPI()

client = InferenceClient(
    provider="fireworks-ai",
    api_key="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",
)

templates = Jinja2Templates("templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1", 
        messages=messages, 
        max_tokens=500,
    )
    answer = completion.choices[0].message['content']
    return HTMLResponse(content=f"<h1>Question:</h1><p>{question}</p><h1>Answer:</h1><p>{answer}</p>")
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import google.generativeai as genai
import os

# Initialize app with metadata
app = FastAPI(
    title="PDF EXPLORER API",
    description="This API lets you upload PDF files and get AI-generated analysis.",
    version="1.0.0",
    contact={
        "name": "ETTYDEV",
        "email": "ettiolayinka4@gmail.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Enable CORS
origins = ["http://localhost:5175"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Max characters to avoid Gemini prompt limits
MAX_CHARS = 5000

@app.post("/analyze-pdf", tags=["File Upload"])
async def extract_text_from_pdf(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Read PDF file contents
        reader = PdfReader(file.file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Truncate text if needed
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        full_prompt = f"{prompt}\n\n{text}"

        # Generate response from Gemini
        response = model.generate_content(full_prompt)

        # Safely extract response text
        result = getattr(response, "text", None)
        if result:
            return {"result": result}
        else:
            return {"error": "No text returned from Gemini."}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/get-models", tags=["AI"])
async def get_models():
    try:
        models = genai.list_models()
        model_names = [m.name for m in models]
        return {"List Of Models": model_names}
    except Exception as e:
        return {"error": f"Failed to fetch models: {str(e)}"}

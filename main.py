from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
import numpy as np
import os

app = FastAPI()

# 🔑 PUT YOUR GEMINI API KEY HERE
import os
genai.configure(api_key=os.environ["gen-lang-client-0494493325"])

# Models
model = genai.GenerativeModel("gemini-1.5-flash")
embed_model = "models/embedding-001"

# Emotion keywords
EMOTIONS = ["anger", "happy", "satisfaction", "doubt", "frustration"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Precompute emotion vectors
emotion_vectors = {}
for emotion in EMOTIONS:
    emb = genai.embed_content(
        model=embed_model,
        content=emotion,
        task_type="semantic_similarity"
    )
    emotion_vectors[emotion] = np.array(emb["embedding"])

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    
    audio_bytes = await file.read()

    # Step 1 — Speech → Text
    response = model.generate_content(
        [
            {"mime_type": file.content_type, "data": audio_bytes},
            "Transcribe this audio clearly."
        ]
    )

    transcript = response.text

    # Step 2 — Convert text → vector
    text_embedding = genai.embed_content(
        model=embed_model,
        content=transcript,
        task_type="semantic_similarity"
    )

    text_vector = np.array(text_embedding["embedding"])

    # Step 3 — Compare with emotions
    scores = {}
    for emotion, vec in emotion_vectors.items():
        score = cosine_similarity(text_vector, vec)
        scores[emotion] = float(score)

    return {
        "transcript": transcript,
        "emotion_scores": scores

    }


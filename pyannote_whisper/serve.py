import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import uvicorn

app = FastAPI()

# Initialize the pipeline and model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=os.environ.get('HF_TOKEN'))
model = whisper.load_model("tiny.en")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the file temporarily
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Process the audio file
    asr_result = model.transcribe(temp_filename)
    diarization_result = pipeline(temp_filename)
    final_result = diarize_text(asr_result, diarization_result)

    # Format the results
    results = []
    for seg, spk, sent in final_result:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": spk,
            "text": sent
        })

    # Remove the temporary file
    os.remove(temp_filename)

    return JSONResponse(content=results)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

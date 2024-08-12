import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import ffmpeg

from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from pyannote_whisper.utils import diarize_text


# Initialize the diarization and transcription pipelines
diarization_pipeline = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global diarization_pipeline, model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=os.environ.get('HF_TOKEN'))

    # Load the local Whisper model
    # model_path = Path("/app/medium.en.pt")
    # if not model_path.exists():
    #     raise FileNotFoundError(f"Whisper model not found at {model_path}")
    # model = whisper.load_model(model_path)
    transcription_model_id = "distil-whisper/distil-large-v3"
    transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transcription_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    transcription_pipeline.to(device)

    transcription_processor = AutoProcessor.from_pretrained(transcription_model_id)

    transcription_pipeline = pipeline(
        "automatic-speech-recognition",
        model=transcription_model,
        tokenizer=transcription_processor.tokenizer,
        feature_extractor=transcription_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=25,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Assert that the `diarization_pipeline` and model are non-None
    assert diarization_pipeline is not None, "Failed to initialize the diarization pipeline"
    assert transcription_pipeline is not None, "Failed to initialize the transcription pipeline"

    yield

    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Create a temporary directory to store files
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)

    # Save the uploaded file temporarily
    temp_input = temp_dir / f"temp_input{Path(file.filename).suffix}"
    temp_output = temp_dir / "temp_output.wav"

    try:
        # Save the uploaded file
        temp_input.write_bytes(await file.read())

        # Convert the file to WAV format
        try:
            (
                ffmpeg
                .input(str(temp_input))
                .output(str(temp_output),
                        vn=None,  # No video
                        acodec='pcm_s16le',  # Audio codec
                        ar=44100,  # Audio sample rate
                        ac=2)  # Audio channels
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise HTTPException(status_code=400, detail=f"Error converting file: {e.stderr.decode()}")

        # Process the audio file
        asr_result = model.transcribe(str(temp_output))
        diarization_result = diarization_pipeline(str(temp_output))
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

        return JSONResponse(content=results)

    finally:
        # Clean up temporary files
        if temp_input.exists():
            temp_input.unlink()
        if temp_output.exists():
            temp_output.unlink()

        # Remove the temporary directory if it's empty
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty or already deleted

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

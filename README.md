# Pyannote-Whisper Transcription Service

This project provides a FastAPI-based web service for audio transcription and speaker diarization using the Pyannote and Distil-Whisper models. It processes audio files, transcribes the speech, and identifies different speakers in the audio.

## Features

- Audio file upload and processing
- Speech transcription using Distil-Whisper
- Speaker diarization using Pyannote
- FastAPI web service
- Dockerized application for easy deployment

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (for optimal performance)
- Hugging Face account and API token

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file in the project root with your Hugging Face token:
   ```
   HF_TOKEN=your_hugging_face_token_here
   ```

3. Ensure you have Docker installed and the NVIDIA Container Toolkit set up if you're using GPUs.

## Building the Docker Image

To build the Docker image, run the following command in the project root:

```
docker build . -t pyannote
```

This process may take some time as it downloads and installs all necessary dependencies and models.

## Running the Container

To run the container, use the following command:

```
docker run --gpus all --env-file .env -p 6677:8000 pyannote
```

This command:
- Uses all available GPUs (`--gpus all`)
- Loads environment variables from the `.env` file
- Maps port 8000 in the container to port 6677 on the host

The service will be available at `http://localhost:6677`.

## API Usage

The service exposes a single endpoint for transcription:

- **POST** `/transcribe`
  - Accepts a form-data with a file upload
  - Returns a JSON response with transcribed text and speaker diarization

Example using curl:
```
curl -X POST -F "file=@path/to/your/audio/file.mp3" http://localhost:6677/transcribe
```

## Project Structure

- `serve.py`: Main application file containing the FastAPI app and transcription logic
- `Dockerfile`: Instructions for building the Docker image
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not tracked in git)

## Notes

- The Distil-Whisper model is downloaded during the Docker build process to optimize startup time.
- Ensure your Hugging Face token has the necessary permissions to access the Pyannote model.
- Processing time may vary depending on the length of the audio file and your hardware capabilities.

## Troubleshooting

- If you encounter GPU-related issues, ensure that your NVIDIA drivers and CUDA toolkit are up to date and compatible with the Python version used in the Dockerfile.
- For any permission issues related to model downloads, check your Hugging Face token permissions.

## Contributing

Contributions to this project are welcome. Please ensure to follow the existing code style and add unit tests for any new features.

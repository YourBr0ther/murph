# server/src/murph_server/main.py
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from murph_server.config import ServerConfig
from murph_server.audio.stt import SpeechToText
from murph_server.audio.tts import TextToSpeech
from murph_server.llm.ollama import OllamaClient
from murph_server.intent.parser import parse_intent, IntentType


app = FastAPI(title="Murph Server")
config = ServerConfig()

stt = SpeechToText(model_size=config.whisper_model)
tts = TextToSpeech(voice=config.piper_voice)
llm = OllamaClient(model=config.ollama_model)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "model": config.ollama_model})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Received {len(data)} bytes of audio")

            try:
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                print(f"Audio array shape: {audio_array.shape}, duration: {len(audio_array)/16000:.2f}s")
                print(f"Audio stats: min={audio_array.min():.3f}, max={audio_array.max():.3f}, mean={abs(audio_array).mean():.4f}")

                # Normalize audio to improve transcription
                max_val = np.abs(audio_array).max()
                if max_val > 0.01:  # Only normalize if there's actual signal
                    audio_array = audio_array / max_val * 0.9
                    print(f"Normalized audio: max now {np.abs(audio_array).max():.3f}")

                # Debug: save audio to file
                import soundfile as sf
                sf.write("debug_audio.wav", audio_array, 16000)
                print("Saved debug audio to debug_audio.wav")

                text = stt.transcribe(audio_array)
                print(f"Transcribed: '{text}'")

                if not text.strip():
                    await websocket.send_json({"type": "no_speech"})
                    continue

                intent = parse_intent(text)
                print(f"Intent: {intent.type}, command: {intent.command}")

                if intent.type == IntentType.MOVEMENT:
                    await websocket.send_json({
                        "type": "command",
                        "command": intent.command,
                        "distance": intent.distance,
                    })
                elif intent.type == IntentType.STOP:
                    await websocket.send_json({"type": "command", "command": "stop"})
                else:
                    response_text = await llm.query(text)
                    print(f"LLM response: '{response_text}'")
                    audio_bytes = tts.synthesize(response_text)
                    print(f"TTS generated {len(audio_bytes)} bytes")
                    await websocket.send_json({"type": "response", "text": response_text})
                    await websocket.send_bytes(audio_bytes)

            except Exception as e:
                print(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")


def main():
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()

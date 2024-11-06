import asyncio
import base64
import io
import os
import ssl
import wave
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_def.SenseVoice import Transcriber
from protos.cosy_grpc import TTSClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(send_blink_command())
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
active_connections = set()
transcriber = Transcriber()
tts_client = TTSClient()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    try:
        while True:
            try:
                # 接收音频数据
                audio_base64 = await websocket.receive_text()
                audio_data = base64.b64decode(audio_base64)

                # 保存临时文件
                save_dir = "received_audio"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_wav = os.path.join(save_dir, f"temp_{timestamp}.wav")
                with open(temp_wav, "wb") as wav_file:
                    wav_file.write(audio_data)

                try:
                    # 转录音频
                    text = transcriber.transcribe(temp_wav)
                    print(f"转录文本: {text}")

                    # 如果转录文本为空或者太短，使用默认回复
                    if not text or len(text.strip()) < 2:
                        text = "抱歉，我没有听清楚，请再说一遍。"
                        print(f"使用默认回复: {text}")

                    # 合成音频
                    audio_data = b""
                    for response in tts_client.synthesize_speech_stream(
                        text, speaker_id="中文女"
                    ):
                        audio_data += response

                    # 创建完整的WAV文件
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
                        wav_file.writeframes(audio_data)

                    # 发送完整的WAV文件
                    wav_base64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")
                    await websocket.send_json({"type": "audio", "data": wav_base64})

                except Exception as e:
                    print(f"转录或合成错误: {e}")
                    # 发送一个默认的错误提示音频
                    error_text = "抱歉，我遇到了一些问题，请稍后再试。"
                    audio_data = b""
                    for response in tts_client.synthesize_speech_stream(
                        error_text, speaker_id="中文女"
                    ):
                        audio_data += response

                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
                        wav_file.writeframes(audio_data)

                    wav_base64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")
                    await websocket.send_json({"type": "audio", "data": wav_base64})

                # 发送眨眼信号
                await websocket.send_json({"type": "blink", "data": "blink"})

                # 清理临时文件
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    print(f"Error cleaning temp file: {e}")

            except Exception as e:
                print(f"WebSocket处理错误: {e}")
                break

    except Exception as e:
        print(f"WebSocket连接错误: {str(e)}")
    finally:
        active_connections.remove(websocket)


async def send_blink_command():
    while True:
        for connection in active_connections:
            try:
                await connection.send_json({"type": "blink", "data": "blink"})
            except Exception:
                continue
        await asyncio.sleep(10)


if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.pem", "key.pem")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        reload=False,
    )

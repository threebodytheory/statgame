import asyncio
import websockets
import json
import os
from funasr import AutoModel
from funasr import AutoModel

from funasr_model import SenseVoiceSmall
import wave
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
import ssl
import time
import re
import funasr_handler
from importlib import reload

# Deepseek API Key: sk-7147a348ab0e4a4c89db501a44e24bc2

# 初始化 SenseVoiceSmall 模型
# model_dir = "iic/SenseVoiceSmall"
# svs_model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, remote_code="./funasr_model.py", disable_update=True, device="cuda:0")
# svs_model.eval()


model = AutoModel(
    model="paraformer-zh-streaming",
    disable_update=True,
    device="cuda:0",
)

vad_model = AutoModel(
    model="fsmn-vad",
    max_single_segment_time=12000
)

punc_model = AutoModel(
    model="ct-punc"
)

# faudio = open("res.pcm", "wb")faudio.write(speech_chunk.tobytes())
#                     faudio.flush()
re_result = re.compile(r"<\|([^\|]+)\|>")

async def main():
    # 加载 SSL 上下文
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile="/etc/nginx/certs/tvt.im.crt", keyfile="/etc/nginx/certs/tvt.im.key")
    async with websockets.serve(lambda ws: funasr_handler.handle_audio(ws, model, vad_model, punc_model), "0.0.0.0", 8765, ssl=ssl_context) as server:
        await server.serve_forever()

(reload(funasr_handler), asyncio.run(main()))

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import threading

import websockets
import json
import numpy as np
import scipy.signal as sps
from typing import Dict, List, Optional, Any
import aiohttp
import traceback
import sys
#import device_base
import time

chunk_bytes = int(44100 * 2 * 0.6)
orig_samples = 44100 * 0.6
new_samples = 16000 * 0.6

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
is_final = False

class StreamASRDeepSeekServer:
    def __init__(self, deepseek_api_key: str, model, vad_model, punc_model):
        # 初始化FunASR流式模型
        self.asr_model = model
        self.vad_model = vad_model
        self.punc_model = punc_model

        # DeepSeek配置
        # self.deepseek_api_key = deepseek_api_key
        # self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        # self.deepseek_model = "deepseek-chat"
        self.deepseek_api_key = "sk-ec0bcd72b992413ab64ac6a7c7638057"
        self.deepseek_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.deepseek_model = "qwen2.5-32b-instruct"

        self.price_perk_input = 0.002
        self.price_perk_output = 0.006

        self.system_prompt = None

        # 音频参数
        self.sample_rate = 16000

        # 会话管理
        self.sessions: Dict[int, Dict] = {}


    async def initalize_sys(self):
        # traindata = []

        # for device in self.devices_db.values():
        #     action_types = set(device_base.action_lists.keys()) & set(device.type)
        #     if device.name is not None and device.name != "" and bool(action_types):
        #         traindata.append(json.dumps({"messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "有什么可以管理的设备?"}, {"role": "assistant", "content": f"{device.name} ({'|'.join(action_types)}, ID: {device.id})"}]}))

        # for device_cls in device_base.action_lists:
        #     traindata.append(json.dumps({"messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": f"设备{device_cls}支持什么样的命令?"}, {"role": "assistant", "content": f"设备{device_cls}支持以下命令：\n"+"\n".join([f"  - {d}" for d in device_base.action_lists[device_cls]])}]}))


        # print("\n".join(traindata))

        # 更新DeepSeek系统提示词
        self.system_prompt = """你是一个高级智能家居控制助手，可以管理以下设备：

{devices_list}

各个设备分类支持的命令如下：
{action_lists}

请根据用户指令生成响应，支持批量操作（如"关闭一楼所有的灯"）。
如果是控制指令，请用以下JSON格式回复，如果有任何附加信息：
{{
    "action": "指令类型",
    "targets": ["设备ID"],
    "params": {{参数}},
    "response": "给用户的回复"
}}

普通对话请按此格式返回:
{{
    "response": "给用户的回复"
}}""".format(
            devices_list=device_base.devices.generate_devices_list(),
            action_lists=device_base.devices.generate_action_list()
        )


#         # 更新DeepSeek系统提示词
#         self.system_prompt = """你是一个高级智能家居控制助手

# 请根据用户指令生成响应，支持批量操作（如"关闭一楼所有的灯"）。
# 如果是控制指令，请用以下JSON格式回复，如果有任何附加信息：
# {{
#     "action": "指令类型",
#     "targets": ["设备ID"],
#     "params": {{参数}},
#     "response": "给用户的回复"
# }}

# 普通对话请按此格式返回:
# {{
#     "response": "给用户的回复"
# }}"""
        print(self.system_prompt)

    async def _batch_device_control(self, targets: List[str], action: str, device_type: str = None) -> str:
        """批量控制设备"""
#        devices = device_base.devices.resolve_targets(targets, device_type)

        if not devices:
            return "没有找到匹配的设备"

        results = []
        for device in devices:
            try:
                if not device.controllable:
                    results.append(f"{device.name} 不可控制")
                    continue

                # 模拟控制操作
                if device.type == "light":
                    if action in ["on", "off"]:
                        device.status["power"] = action
                        results.append(f"{device.name} 已{action}")
                    elif action == "toggle":
                        new_state = "off" if device.status["power"] == "on" else "on"
                        device.status["power"] = new_state
                        results.append(f"{device.name} 已切换为{new_state}")
                    else:
                        results.append(f"{device.name} 不支持操作: {action}")

                elif device.type == "thermostat":
                    if action == "on":
                        device.status["power"] = "on"
                        results.append(f"{device.name} 已开启")
                    elif action == "off":
                        device.status["power"] = "off"
                        results.append(f"{device.name} 已关闭")
                    else:
                        results.append(f"{device.name} 不支持操作: {action}")

                # 模拟网络延迟
                await asyncio.sleep(0.1)

            except Exception as e:
                results.append(f"{device.name} 控制失败: {str(e)}")

        return "\n".join(results)

    def _query_devices(self, targets: List[str]) -> str:
        """查询设备状态"""
        devices = device_base.devices.resolve_targets(targets)

        if not devices:
            return "没有找到匹配的设备"

        result = []
        for device in devices:
            status_desc = ", ".join(f"{k}:{v}" for k, v in device.status.items())
            result.append(f"{device.name} ({device.location}): {status_desc}")

        return "当前状态:\n" + "\n".join(result)

    def ai_session_thread(self, stop_flag: threading.Event, session_id: any):
        asyncio.run(self.ai_call_async(stop_flag, session_id))

    async def handle_connection(self, websocket):
        """处理WebSocket连接"""
        session_id = id(websocket)
        self.sessions[session_id] = {
            'websocket': websocket,
            'speech_chunk_buf': b"",
            'vad_chunk_buf': b"",
            'vad_chunk_buf_ts': 0,
            'vad_start_pos': 0,
            'cache': {},
            'vad_cache': {},
            'dialog_history': [],
            'last_deepseek_time': 0,
            'request_flag': threading.Event(),
            'loading_flag': threading.Event()
        }
        stop_flag = threading.Event()
        exec_thread = threading.Thread(target=self.ai_session_thread, args=(stop_flag, session_id,))
        exec_thread.start()

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self.process_audio_chunk(session_id, message)
                elif isinstance(message, str):
                    await self.process_text_message(session_id, message)

        except websockets.exceptions.ConnectionClosed:
            print(f"连接关闭: {session_id}")
        finally:
            self.sessions[session_id]['request_flag'].set()
            stop_flag.set()
            exec_thread.join()
            self.sessions.pop(session_id, None)

    async def process_audio_chunk(self, session_id: int, pcm_data: bytes):
        """处理音频数据块"""
        session = self.sessions.get(session_id)
        if not session:
            return

        buf = np.copy(np.frombuffer(pcm_data, dtype=np.float32))
        pcm_data = (buf * 32768).astype('<i2').tobytes()
        session['speech_chunk_buf'] += pcm_data

        if len(session['speech_chunk_buf']) >= chunk_bytes:

            speech_chunk = sps.resample(np.frombuffer(session['speech_chunk_buf'][0:chunk_bytes], dtype=np.int16), int(chunk_bytes / 2 * (16000 / 44100))).astype(np.int16)
            session['speech_chunk_buf'] = session['speech_chunk_buf'][chunk_bytes:]

            # 使用线程池执行识别(避免阻塞事件循环)
            future = asyncio.get_event_loop().run_in_executor(
                None,  # 使用默认线程池
                self._recognize_audio,
                speech_chunk,
                session
            )

            try:
                text, is_final = await future

                if text and len(text) >= 5 and is_final:
                    try:
                        await self._send_to_client(session_id, {
                            "type": "partial_result",
                            "text": text,
                            "is_final": True
                        })

                        await self.process_complete_sentence(session_id, text)
                    except websockets.exceptions.ConnectionClosed:
                        print("连接已关闭，无法发送消息")
            except Exception as e:
                print(f"识别错误: {e.__class__.__name__}: {e}", "\n" + ''.join(traceback.format_tb(e.__traceback__)), file=sys.stderr, flush=True)



    def _recognize_audio(self, speech_chunk: bytes, session) -> tuple:
        """执行语音识别(在同步上下文中运行)"""
        # audio_data = np.frombuffer(pcm_data, dtype=np.int16)
        # audio_data = audio_data.astype(np.float32) / 32768.0

        vad_buf_start = int(session['vad_chunk_buf_ts'] - 1000 * len(session['vad_chunk_buf']) / 2 / self.sample_rate)
        session['vad_chunk_buf'] += speech_chunk.tobytes()
        session['vad_chunk_buf_ts'] += 1000 * len(speech_chunk) / self.sample_rate
        vad_res = self.vad_model.generate(
            input=speech_chunk,
            cache=session['vad_cache'],
            disable_pbar=True,
            sample_rate=self.sample_rate,
            chunk_size=600
        )
        content = ""
        for vad_result in vad_res:
            for vad_res in vad_result['value']:
                if vad_res[0] == -1:
                    start_sample = int((session['vad_start_pos'] - vad_buf_start) / 1000 * self.sample_rate)
                    end_sample = int((vad_res[1] - vad_buf_start) / 1000 * self.sample_rate)
                    vad_speech_chunk = np.frombuffer(session['vad_chunk_buf'][start_sample*2:end_sample*2], dtype=np.int16)
                    session['vad_chunk_buf'] = session['vad_chunk_buf'][end_sample*2:]

                    if vad_res[1] - session['vad_start_pos'] > 1000:
                        res = self.asr_model.generate(
                            input=vad_speech_chunk,
                            cache=session['cache'],
                            disable_pbar=True,
                            sample_rate=self.sample_rate,
                            chunk_size=chunk_size,
                            encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back,
                            is_final = True,
                            incremental=False  # 启用增量模式
                        )

                        for result in res:
                            if result['text'] != '':
                                has_ct_result = False
                                text = result['text']
                                ct_res = self.punc_model.generate(input=result['text'], disable_pbar=True)
                                for ct_result in ct_res:
                                    if ct_result['text'] != '':
                                        has_ct_result = True
                                        text = ct_result['text']

                                print(f"\033[32mRecognized: {text}\033[0m")
                                content += text+(" " if has_ct_result else "")

                elif vad_res[1] == -1:
                    session['vad_start_pos'] = vad_res[0]

        # print("Remain buffer: %.1f s" % (len(session['vad_chunk_buf']) / 2 / self.sample_rate))
        if content != "":
            return content, True
        else:
            return None, False

    async def ai_call_async(self, stop_flag, session_id):
        session = self.sessions.get(session_id)
        while not stop_flag.is_set():
            session['request_flag'].wait(timeout=0.1)
            if stop_flag.is_set():
                break
            if session['request_flag'].is_set():
                session['loading_flag'].set()
                session['request_flag'].clear()

                deepseek_response = await self.query_deepseek(session['dialog_history'], session, session_id)
                session['last_deepseek_time'] = asyncio.get_event_loop().time()

                # 处理DeepSeek响应
                if isinstance(deepseek_response, dict) and "action" in deepseek_response:
                    # 执行结构化指令
                    action_result = await self.execute_action(deepseek_response)
                    response = f"{deepseek_response.get('response', '')}\n\n{action_result}"
                elif isinstance(deepseek_response, dict) and "response" in deepseek_response:
                    response = deepseek_response.get('response', '')
                else:
                    response = deepseek_response

                print("Deepseek text: ", response)
                # 更新对话历史并发送响应
                session['dialog_history'].append({"role": "assistant", "content": response})
                try:
                    await self._send_to_client(session_id, {
                        "type": "ai_response",
                        "text": response,
                        "is_final": True
                    })
                except websockets.exceptions.ConnectionClosed:
                    print("连接已关闭，中止AI指令进程")
                    stop_flag.set()
            session['loading_flag'].clear()

        http_sess = session.get("http_session")
        if http_sess is not None:
            await http_sess.close()

    async def process_complete_sentence(self, session_id: int, sentence: str):
        """处理完整句子"""
        session = self.sessions.get(session_id)
        if not session:
            print(f"\033[31mError: Session {session_id} invalid\033[0m", file=sys.stderr)
            return

        if session['loading_flag'].is_set():
            print("\033[33mWarning: AI Process is loading..., ignore the sentence\033[0m", file=sys.stderr)
            return

        # 添加到对话历史
        session['dialog_history'].append({"role": "user", "content": sentence})

        # 简单命令本地处理
        local_response = self._handle_local_command(sentence)
        if local_response:
            await self._send_to_client(session_id, {
                "type": "command_response",
                "content": local_response
            })
            return

        # 发送到DeepSeek处理
        current_time = asyncio.get_event_loop().time()
        if current_time - session['last_deepseek_time'] < 1.0:
            await asyncio.sleep(1.0 - (current_time - session['last_deepseek_time']))

        print("sending to deepseek")
        session['request_flag'].set()
        # print(session['dialog_history'])

    async def query_deepseek(self, dialog_history: List[Dict], session, session_id) -> str:
        """查询DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }

        messages = [{
            "role": "system",
            "content": self.system_prompt
        }]
        messages.extend(dialog_history[-3:])  # 只保留最近3轮对话

        payload = {
            "model": self.deepseek_model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
            # "response_format": {'type': 'json_object'},
            "max_tokens": 8192
        }

        if session.get("http_session") is None:
            session['http_session'] = aiohttp.ClientSession()
        t1 = time.time()
        try:
            async with session['http_session'].post(
                self.deepseek_url,
                headers=headers,
                json=payload
            ) as response:
                # 检查响应状态
                if response.status != 200:
                    error = await response.text()
                    return f"请求失败，状态码: {response.status}，{error}"

                buffer = ""
                content = ""
                data_finished = False
                isFirstToken = True
                async for chunk in response.content.iter_chunked(1024):  # 1KB 为块大小
                    data = chunk.decode('utf-8')
                    buffer += data

                    # 处理完整的事件（假设使用 Event Stream 格式）
                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)

                        # 跳过心跳包（如果有）
                        if event.startswith(":"):
                            continue

                        if event == "data: [DONE]":
                            data_finished = True
                            break

                        # 解析事件数据
                        if event.startswith("data: "):
                            try:
                                result = json.loads(event[6:])  # 移除 "data: " 前缀
                                # print(result)
                                # 处理返回数据（根据 API 响应格式调整）
                                if "choices" in result:
                                    delta_content = result["choices"][0].get("delta", {}).get("content", "")
                                    if isFirstToken:
                                        isFirstToken = False
                                        print("first token latency: %.1f ms" % (1000.*(time.time() - t1)))
                                    content += delta_content

                                    if len(content) >= 18:
                                        await self._send_to_client(session_id, {
                                            "type": "ai_response",
                                            "text": content[18:] if content.startswith("{\n    \"response\": \"") else content,
                                            "is_final": False
                                        })

                                    print(delta_content, end="", flush=True)
                                    if result["choices"][0].get("delta", {}).get("finish_reason", None) is not None:
                                        data_finished = True

                                if result.get("usage", None):
                                    if 'prompt_tokens' in result['usage']:
                                        print("Cost: input %.4f  output: %.4f" % (
                                            result['usage']['prompt_tokens'] / 1000. * self.price_perk_input,
                                            result['usage']['completion_tokens'] / 1000. * self.price_perk_output
                                        ))
                                    print(result['usage'])
                            except websockets.exceptions.ConnectionClosed:
                                print("连接已关闭，执行进程中止")
                                data_finished = True
                            except Exception as e:
                                print(f"处理分块数据失败：{e.__class__.__name__}: {str(e)}\n{event}", file=sys.stderr, flush=True)

                    if data_finished:
                        break

                print("\ntoken take: %.1f ms" % (1000.*(time.time() - t1)))

                try:
                    json_resp = json.loads(content)
                    return json_resp
                except json.JSONDecodeError:
                    return content
        except Exception as e:
            return f"请求失败: {e.__class__.__name__}: {str(e)}"

    def _handle_local_command(self, text: str) -> Optional[str]:
        """处理本地命令"""
        text = text.lower()
        if "当前时间" in text:
            from datetime import datetime
            return f"现在时间是: {datetime.now().strftime('%H:%M:%S')}"
        return None

    async def execute_action(self, command: Dict) -> str:
        """执行结构化指令"""
        # 这里实现具体的设备控制逻辑
        # 示例实现：
        print("执行action", command)
        if command["action"] == "control_light":
            return f"执行灯光控制: {command.get('targets', [])} -> {command.get('params', {})}"
        return "指令执行完成"

    async def _send_to_client(self, session_id: int, message: Dict):
        """发送消息给客户端"""
        session = self.sessions.get(session_id)
        if session and session['websocket']:
            await session['websocket'].send(json.dumps(message))

    async def process_text_message(self, session_id: int, message: str):
        """处理文本消息"""
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            pass

async def handle_audio(websocket, model, vad_model, punc_model):
    global rec, DSAudioSrv

    DSAudioSrv = StreamASRDeepSeekServer("sk-7147a348ab0e4a4c89db501a44e24bc2", model, vad_model, punc_model)
    await DSAudioSrv.initalize_sys()

    print("Client connected")

    await DSAudioSrv.handle_connection(websocket)

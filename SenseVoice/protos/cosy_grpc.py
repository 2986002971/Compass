import queue
import threading

import grpc
import numpy as np
import pygame
import torch

from . import cosyvoice_pb2, cosyvoice_pb2_grpc


class TTSClient:
    def __init__(self, host="localhost", port=50000):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = cosyvoice_pb2_grpc.CosyVoiceStub(self.channel)
        self.prompt_sr = 16000
        self.target_sr = 22050
        self.audio_queue = queue.Queue()
        self.is_playing = False

        # 初始化 pygame mixer，使用目标采样率
        pygame.mixer.init(frequency=self.target_sr, channels=1)
        pygame.init()

    def synthesize_speech(self, text, speaker_id="中文女", stream=False):
        """基础的 TTS 合成
        Args:
            text: 要合成的文本
            speaker_id: 说话人ID
            stream: 是否流式播放
        """
        request = cosyvoice_pb2.Request()
        sft_request = cosyvoice_pb2.sftRequest()
        sft_request.spk_id = speaker_id
        sft_request.tts_text = text
        request.sft_request.CopyFrom(sft_request)

        if not stream:
            # 非流式模式：返回完整音频
            audio_data = b""
            for response in self.stub.Inference(request):
                audio_data += response.tts_audio

            # 保持一维数组
            audio_tensor = torch.from_numpy(
                np.frombuffer(audio_data, dtype=np.int16)
            ).float() / (2**15)

            return audio_tensor
        else:
            # 流式模式：启动播放线程
            self.is_playing = True
            play_thread = threading.Thread(target=self._play_audio_stream)
            play_thread.start()

            # 将音频数据放入队列
            try:
                for response in self.stub.Inference(request):
                    if not self.is_playing:
                        break
                    audio_chunk = np.array(
                        np.frombuffer(response.tts_audio, dtype=np.int16)
                    ).astype(np.float32) / (2**15)
                    self.audio_queue.put(audio_chunk)
            finally:
                self.audio_queue.put(None)  # 发送结束信号

    def _play_audio_stream(self):
        """使用 pygame 播放音频流"""
        while self.is_playing:
            chunk = self.audio_queue.get()
            if chunk is None:  # 结束信号
                break

            # 转换为 pygame 可以播放的格式
            audio_data = (chunk * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(audio_data)

            # 播放音频块
            sound.play()
            # 等待当前块播放完成
            while pygame.mixer.get_busy() and self.is_playing:
                pygame.time.wait(10)

    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
        pygame.mixer.stop()  # 停止所有声音
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def __del__(self):
        """清理 pygame"""
        pygame.mixer.quit()
        pygame.quit()

    def synthesize_speech_stream(self, text, speaker_id="中文女"):
        """流式返回音频数据
        Args:
            text: 要合成的文本
            speaker_id: 说话人ID
        Returns:
            音频数据流
        """
        request = cosyvoice_pb2.Request()
        sft_request = cosyvoice_pb2.sftRequest()
        sft_request.spk_id = speaker_id
        sft_request.tts_text = text
        request.sft_request.CopyFrom(sft_request)

        for response in self.stub.Inference(request):
            yield response.tts_audio


# 使用示例
if __name__ == "__main__":
    client = TTSClient()

    # 流式播放示例
    print("开始流式合成播放...")
    client.synthesize_speech(
        "这是一段测试音频，我们正在测试流式播放功能。让我们看看效果如何！",
        speaker_id="中文女",
        stream=True,
    )

    # 等待用户输入以停止播放
    input("按回车键停止播放...")
    client.stop_playback()

    # 非流式播放示例
    print("\n开始普通合成...")
    audio = client.synthesize_speech("这是非流式播放测试。", speaker_id="中文女")

    # 转换为 pygame 可播放的格式并播放
    audio_data = (audio.numpy() * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(audio_data)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.wait(100)

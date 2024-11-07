import os
import subprocess


def run_program(cmd, cwd=None, env=None, silent=False):
    kwargs = {
        "cwd": cwd,
        "env": env,
        "shell": True,
        "text": True,
    }

    if silent:
        kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})

    return subprocess.Popen(cmd, **kwargs)  # cmd 作为第一个位置参数


# 获取当前环境变量
env = os.environ.copy()
# 取消设置 PIXI_PROJECT_MANIFEST
env.pop("PIXI_PROJECT_MANIFEST", None)

# 定义四个程序的配置
programs = [
    {
        "cwd": "Arm",
        "cmd": "pixi run python person_follow.py",
        "env": env,
        "silent": True,
    },
    {
        "cwd": "SenseVoice",
        "cmd": "pixi run python server.py",
        "env": env,
        "silent": False,
    },
    {
        "cwd": "CosyVoice",
        "cmd": "pixi run python runtime/python/grpc/server.py",
        "env": env,
        "silent": False,
    },
    {
        "cwd": "llama.cpp",
        "cmd": "./llama-server -m models/minicpm3-4b-q4_k_m.gguf -c 2048",
        "env": None,
        "silent": False,
    },
]

# 启动所有程序
processes = []
for prog in programs:
    p = run_program(prog["cmd"], prog["cwd"], prog["env"], prog.get("silent", False))
    processes.append(p)
    print(f"Started process with command: {prog['cmd']}")

# 等待所有程序结束
try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n正在关闭所有程序...")
    for p in processes:
        p.terminate()
    for p in processes:
        p.wait()

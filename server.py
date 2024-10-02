import ssl

from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("客户端已连接")


@socketio.on("disconnect")
def handle_disconnect():
    print("客户端已断开连接")


@socketio.on("audio")
def handle_audio(data):
    print("收到音频数据")
    # 这里可以处理接收到的音频数据


def send_blink_command():
    while True:
        socketio.emit("blink", {"data": "blink"})
        socketio.sleep(10)


if __name__ == "__main__":
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain("cert.pem", "key.pem")
    socketio.start_background_task(send_blink_command)
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, ssl_context=context)

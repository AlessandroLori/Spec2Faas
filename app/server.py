import asyncio
import json
import logging
import os
import threading

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock

from app.Utils import get_config_data, load_env_variables, set_logging_config
from app.main import ROOT


class Server:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.sock = Sock(self.app)
        self.connected_clients = set()
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/start", "start_processing", self.start_processing, methods=["POST"])
        self.sock.route("/stream")(self.stream)

    def send_chunk(self, text: str, sender: str, message_type: str = "text"):
        message = json.dumps({"type": message_type, "text": text, "sender": sender})
        for ws in list(self.connected_clients):
            try:
                ws.send(message)
            except Exception as exc:
                print(f"Error: {exc}")
                self.connected_clients.discard(ws)

    def stream(self, ws):
        self.connected_clients.add(ws)
        try:
            while True:
                ws.receive()
        except Exception:
            print(f"Client disconnected: {ws}")
        finally:
            self.connected_clients.discard(ws)

    def index(self):
        return "Server WebSocket active."

    def start_processing(self):
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        load_env_variables()
        log_path = os.getenv("LOG_PATH")
        if log_path:
            set_logging_config(log_path)
        llm = get_config_data(str(ROOT / "config.yaml"))
        thread = threading.Thread(
            target=self.run_main_in_background,
            args=(llm, data["text"]),
        )
        thread.start()
        return jsonify({
            "status": "ok", "message": "Processing started in background",
            "received": data["text"],
        })

    def run_main_in_background(self, llm, user_text):
        from app.main_RAG import main

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main(llm, self, user_text))
        except Exception as exc:
            print(f"Error during background execution: {exc}")
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            logging.shutdown()

    def run(self, host="127.0.0.1", port=5000, debug=True, use_reloader=False):
        self.app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)


if __name__ == "__main__":
    os.environ["UI"] = "True"
    Server().run()

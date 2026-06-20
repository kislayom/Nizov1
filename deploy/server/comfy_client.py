"""Minimal ComfyUI HTTP client for Nizo's video backend.

Submits an API-format workflow, polls /history to completion, returns the produced media file paths,
and frees VRAM via /free. Deliberately stdlib-only (urllib, no websockets/httpx) so it can be imported
into the voice sidecar without adding deps. The caller wraps queue()+wait() in llama_paused() and calls
free() afterwards so ComfyUI releases VRAM back to Qwen between jobs.
"""
import json
import os
import time
import urllib.parse
import urllib.request


class ComfyClient:
    def __init__(self, base="http://127.0.0.1:8188"):
        self.base = base.rstrip("/")

    def _post(self, path, obj):
        data = json.dumps(obj).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get(self, path, timeout=30):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def object_info(self):
        """Full node catalogue — used to validate a workflow's node names before submitting."""
        return self._get("/object_info")

    def queue(self, workflow, client_id=None):
        """workflow: API-format dict {node_id: {class_type, inputs}}. Returns prompt_id."""
        payload = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        return self._post("/prompt", payload)["prompt_id"]

    def run_with_progress(self, workflow, client_id, on_progress, timeout=1800):
        """Queue the workflow and stream live sampler progress over the ComfyUI WebSocket, calling
        on_progress(step, total, node) as it advances. Returns the /history entry once complete.
        Best-effort: if the WS import/connection fails, falls back to a plain queue+poll (no steps)."""
        import json as _json
        try:
            import websocket  # websocket-client
        except Exception:
            pid = self.queue(workflow, client_id=client_id)
            return self.wait(pid, timeout=timeout)
        ws_url = self.base.replace("http://", "ws://").replace("https://", "wss://") + "/ws?clientId=" + client_id
        ws = websocket.create_connection(ws_url, timeout=15)
        try:
            pid = self.queue(workflow, client_id=client_id)
            ws.settimeout(timeout)
            while True:
                raw = ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue  # binary preview frames — ignore
                msg = _json.loads(raw)
                mtype = msg.get("type")
                data = msg.get("data") or {}
                mpid = data.get("prompt_id")
                if mtype == "progress" and mpid in (None, pid):
                    try:
                        on_progress(int(data.get("value", 0)), int(data.get("max", 0)), data.get("node"))
                    except Exception:
                        pass
                elif mtype == "executing" and data.get("node") is None and mpid == pid:
                    break  # this prompt finished
                elif mtype == "execution_error" and mpid == pid:
                    raise RuntimeError("comfy execution error: " + _json.dumps(data)[:400])
        finally:
            try: ws.close()
            except Exception: pass
        return self.wait(pid, timeout=60)

    def wait(self, prompt_id, timeout=1800, poll=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                h = hist[prompt_id]
                st = h.get("status", {})
                if st.get("completed") or st.get("status_str") == "success":
                    return h
                if st.get("status_str") == "error":
                    raise RuntimeError("comfy workflow error: " + json.dumps(st)[:800])
            time.sleep(poll)
        raise TimeoutError(f"comfy job {prompt_id} timed out after {timeout}s")

    def outputs(self, history_entry):
        """[(filename, subfolder, type), ...] for produced video/gif/image media."""
        files = []
        for _node_id, out in (history_entry.get("outputs") or {}).items():
            for key in ("gifs", "videos", "images"):
                for f in out.get(key, []):
                    files.append((f["filename"], f.get("subfolder", ""), f.get("type", "output")))
        return files

    def fetch(self, filename, subfolder, ftype, dst, timeout=300):
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": ftype})
        with urllib.request.urlopen(self.base + "/view?" + q, timeout=timeout) as r:
            data = r.read()
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        return dst

    def free(self, unload_models=True, free_memory=True):
        """Unload models + free VRAM so Qwen can reload. Best-effort."""
        try:
            self._post("/free", {"unload_models": unload_models, "free_memory": free_memory})
        except Exception:
            pass

import http.server
import socketserver
import sys
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import download_dataset as dd  # noqa: E402


def _serve(blob: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            body = blob[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(blob)-1}/{len(blob)}")
            self.end_headers()
            self.wfile.write(body)

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_resume_verify_and_extract(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w", zipfile.ZIP_STORED) as z:
        z.writestr("3D/frames/frame_00001.png", b"a" * 3_000_000)
        z.writestr("3D/masks/frame_00001.png", b"b" * 10)
    blob = src_zip.read_bytes()
    srv = _serve(blob)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/f"
        dest = tmp_path / "raw" / "3D.zip"
        dest.parent.mkdir()
        dest.write_bytes(blob[:1_234_567])  # simulated interrupted download
        dd.download(url, dest, len(blob))   # must RESUME, not restart
        assert dest.read_bytes() == blob
        dd.download(url, dest, len(blob))   # already complete -> no-op
        out = tmp_path / "data"
        dd.extract(dest, out, force=False)
        assert (out / "3D" / "frames" / "frame_00001.png").stat().st_size == 3_000_000
        assert (out / ".extracted").exists()
    finally:
        srv.shutdown()


def test_size_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    srv = _serve(b"x" * 1000)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/f"
        try:
            dd.download(url, tmp_path / "z.zip", 2000)
            assert False, "expected size mismatch"
        except RuntimeError as e:
            assert "Size mismatch" in str(e)
    finally:
        srv.shutdown()

# 검수 서버 — review.html 서빙 + 브라우저에서 서버로 직접 저장/반영(POST)
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/review_server.py [PORT]
#   GET  /review.html        검수 UI
#   POST /save   (body=jsonl) → data/elements_hidpi/reviewed.jsonl 저장(서버)
#   POST /apply  (body=jsonl) → reviewed.jsonl 저장 + ingest_reviewed.py 실행(라벨 반영, 제외=삭제)
import http.server, os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HID = os.path.join(REPO, "data/elements_hidpi"); ING = os.path.join(HERE, "ingest_reviewed.py")
os.chdir(HID)  # review.html 등 서빙 루트
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

class H(http.server.SimpleHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        rp = os.path.join(HID, "reviewed.jsonl")
        open(rp, "w", encoding="utf-8").write(body)
        cnt = sum(1 for l in body.splitlines() if l.strip())
        if self.path.startswith("/save"):
            self._send(200, {"ok": True, "saved": cnt, "path": rp})
        elif self.path.startswith("/apply"):
            r = subprocess.run([sys.executable, ING, rp], capture_output=True, text=True)
            self._send(200, {"ok": r.returncode == 0, "saved": cnt,
                             "out": r.stdout.strip(), "err": r.stderr.strip()[-600:]})
        else:
            self._send(404, {"ok": False, "err": "unknown path"})

print(f"검수 서버: http://0.0.0.0:{PORT}/review.html  (POST /save·/apply 지원)  루트={HID}")
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

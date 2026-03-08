#!/usr/bin/env python3
"""
Pleez Growth Engine — Web Service
REST API + static frontend
"""

import json
import os
import sys
import io
import cgi
import html
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure engine is importable from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import analyse

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ─────────────────────────────────────────────────────────────────────────────
# MULTIPART PARSER (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def parse_multipart(headers, body: bytes):
    """Extract uploaded file bytes from a multipart/form-data body."""
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data")

    # Extract boundary
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip('"')
            break
    if not boundary:
        raise ValueError("No boundary found in Content-Type")

    boundary_bytes = ("--" + boundary).encode()
    parts = body.split(boundary_bytes)

    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        # Split headers and body
        if b"\r\n\r\n" in part:
            header_section, file_body = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_section, file_body = part.split(b"\n\n", 1)
        else:
            continue

        header_str = header_section.decode("utf-8", errors="ignore")
        # Check it's a file part (has filename=)
        if "filename=" not in header_str:
            continue

        # Strip trailing boundary markers
        file_body = file_body.rstrip(b"\r\n--")
        if file_body.endswith(b"--"):
            file_body = file_body[:-2]
        file_body = file_body.rstrip(b"\r\n")
        return file_body

    raise ValueError("No file found in multipart body")


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class PleezHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, body: str, status: int = 200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_static(self, path: str):
        filepath = os.path.join(STATIC_DIR, path.lstrip("/"))
        if not os.path.isfile(filepath):
            self.send_json({"error": "Not found"}, 404)
            return
        with open(filepath, "rb") as f:
            content = f.read()
        ext = os.path.splitext(filepath)[1]
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".ico":  "image/x-icon",
        }
        ct = content_types.get(ext, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_static("/index.html")
        elif path == "/api/health":
            self.send_json({"status": "ok", "service": "Pleez Growth Engine", "version": "1.0.0"})
        elif path == "/api/schema":
            self.send_json(SCHEMA_DOC)
        elif path.startswith("/static/"):
            self.send_static(path[len("/static"):])
        else:
            self.send_json({"error": f"Route {path} not found. See /api/schema for API docs."}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path not in ("/api/analyse", "/api/analyze"):
            self.send_json({"error": f"Unknown POST endpoint: {path}"}, 404)
            return

        # Handle Expect: 100-continue (curl sends this for large files)
        expect = self.headers.get("Expect", "")
        if expect.lower() == "100-continue":
            self.send_response(100)
            self.end_headers()

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_json({"error": "Empty request body. POST an XLSX file as multipart/form-data with field name 'file'."}, 400)
            return

        body = self.rfile.read(content_length)

        try:
            file_bytes = parse_multipart(self.headers, body)
        except Exception as e:
            self.send_json({"error": f"Could not parse file upload: {str(e)}. Send as multipart/form-data with field 'file'."}, 400)
            return

        try:
            report = analyse(file_bytes)
            self.send_json({"success": True, "report": report})
        except ValueError as e:
            self.send_json({"success": False, "error": str(e)}, 422)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Analysis error: {tb}")
            self.send_json({"success": False, "error": f"Analysis failed: {str(e)}"}, 500)


# ─────────────────────────────────────────────────────────────────────────────
# API DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_DOC = {
    "service": "Pleez Growth Engine API",
    "version": "1.0.0",
    "description": "Analyses a restaurant XLSX file and returns top 5 high-impact promotion recommendations.",
    "endpoints": {
        "GET /api/health": {
            "description": "Service health check",
            "response": {"status": "ok"}
        },
        "GET /api/schema": {
            "description": "This documentation"
        },
        "POST /api/analyse": {
            "description": "Analyse a restaurant XLSX and return top 5 action recommendations",
            "request": {
                "content_type": "multipart/form-data",
                "field": "file",
                "accepted_formats": ["xlsx"],
                "required_sheets": ["metrics"],
                "optional_sheets": ["promotions", "restaurant_tags", "restaurants", "orders", "promotion_daily_metrics"],
                "metrics_sheet_columns": [
                    "restaurant_id", "start_datetime", "end_datetime",
                    "avg_rating", "uptime", "total_orders", "new_users",
                    "returning_users", "conversion_rate", "rejected_orders", "cancelled_orders"
                ]
            },
            "response": {
                "success": True,
                "report": {
                    "restaurant_id": "string",
                    "cuisine": "string",
                    "tags": ["list of cuisine tags"],
                    "avg_rating": "float",
                    "avg_uptime_pct": "float",
                    "avg_daily_orders": "float",
                    "predicted_wins": "int (actions with predicted ROI > 35%)",
                    "avg_predicted_roi": "float",
                    "top_actions": [
                        {
                            "rank": "int (1–5)",
                            "promo_type": "string (free-delivery | save | two-for-one | rewards)",
                            "label": "string",
                            "predicted_roi": "float (%)",
                            "confidence_pct": "int (%)",
                            "is_predicted_win": "bool",
                            "headline": "string",
                            "rationale": "string",
                            "implementation": "string",
                            "discount_guidance": "string | null",
                            "target_guidance": "string",
                            "timing_guidance": "string",
                            "supporting_data_points": ["list of strings"]
                        }
                    ],
                    "hidden_truths": [
                        {
                            "tag": "string",
                            "claim": "string",
                            "evidence": "string"
                        }
                    ],
                    "data_quality_notes": ["list of warnings about missing data"],
                    "model_notes": "string"
                }
            },
            "curl_example": (
                "curl -X POST http://localhost:8000/api/analyse "
                "-F 'file=@restaurant_data.xlsx'"
            )
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), PleezHandler)
    print(f"""
╔══════════════════════════════════════════════════════╗
║         Pleez Growth Engine  — v1.0.0                ║
╠══════════════════════════════════════════════════════╣
║  Frontend:   http://localhost:{PORT}/                  ║
║  API docs:   http://localhost:{PORT}/api/schema        ║
║  Health:     http://localhost:{PORT}/api/health        ║
║  Analyse:    POST http://localhost:{PORT}/api/analyse  ║
╠══════════════════════════════════════════════════════╣
║  curl -X POST http://localhost:{PORT}/api/analyse \\   ║
║       -F 'file=@restaurant_data.xlsx'                ║
╚══════════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()

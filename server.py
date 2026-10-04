"""
server.py — HTTP server for the DiverseTune recommendation engine.
Serves the web frontend and exposes two JSON APIs:
  GET /api/search?q=<query>     → search tracks by name or artist
  GET /api/recommend?id=<idx>   → get recommendations for a track index

Column names updated for the new dataset:
  name (was track_name), genre (was track_genre), artists (unchanged)
"""

import http.server
import socketserver
import json
import urllib.parse
from recommendation_engine import (
    load_and_preprocess,
    greedy_recommend,
    content_filtering_recommend,
    graph_dpp_rerank_recommend,
    mmr_recommend,
    mmr_floor_recommend,
)

PORT = 8000
tracks = load_and_preprocess("data/song_track.csv")


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # ── API: Search tracks ──────────────────────────────────────
        if self.path.startswith("/api/search"):
            parsed_path = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_path.query).get("q", [""])[0].lower()

            results = []
            for i, t in enumerate(tracks):
                # Search in both name and artists (new column names)
                if query in t["name"].lower() or query in t["artists"].lower():
                    results.append({
                        "id":         i,
                        "name":       t["name"],
                        "artist":     t["artists"],
                        "genre":      t.get("genre", ""),
                        "popularity": round(t["popularity"], 1),
                    })

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(results[:10]).encode())
            return

        # ── API: Recommendations ────────────────────────────────────
        elif self.path.startswith("/api/recommend"):
            parsed_path = urllib.parse.urlparse(self.path)
            try:
                track_id = int(urllib.parse.parse_qs(parsed_path.query).get("id", [0])[0])
            except (ValueError, IndexError):
                track_id = 0

            track_id = max(0, min(track_id, len(tracks) - 1))

            greedy         = greedy_recommend(tracks, K=5, liked_indices=[track_id])
            similar        = content_filtering_recommend(tracks, [track_id], K=5)
            graph_dpp_recs = graph_dpp_rerank_recommend(tracks, [track_id], K=5, min_niche_pct=0.20)
            mmr_recs       = mmr_recommend(tracks, [track_id], K=5)
            mmr_floor_recs = mmr_floor_recommend(tracks, [track_id], K=5)

            def format_recs(recs):
                return [
                    {
                        "name":       r["name"],
                        "artist":     r["artists"],
                        "genre":      r.get("genre", ""),
                        "popularity": round(r["popularity"], 1),
                    }
                    for r in recs
                ]

            response = {
                "greedy":          format_recs(greedy),
                "content_filtering": format_recs(similar),
                "mmr": format_recs(mmr_recs),
                "mmr_floor": format_recs(mmr_floor_recs),
                "graph_dpp_rerank": format_recs(graph_dpp_recs),
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return

        # ── Default: serve static files ─────────────────────────────
        else:
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def log_message(self, format, *args):
        # Suppress default per-request log noise; comment out to restore
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving DiverseTune at http://localhost:{PORT}")
        httpd.serve_forever()

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import sys
import os

# Add parent directory to path to import recommendation_engine
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from recommendation_engine import load_and_preprocess, greedy_recommend, content_filtering_recommend, graph_dpp_rerank_recommend

CSV_PATH = os.path.join(ROOT_DIR, "data", "spotify_tracks.csv")
tracks = load_and_preprocess(CSV_PATH)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/search'):
            parsed_path = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_path.query).get('q', [''])[0].lower()
            
            results = []
            for i, t in enumerate(tracks):
                if query in t['track_name'].lower() or query in t['artists'].lower():
                    results.append({"id": i, "name": t['track_name'], "artist": t['artists'], "popularity": t['popularity']})
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results[:10]).encode())
            return
            
        elif self.path.startswith('/api/recommend'):
            parsed_path = urllib.parse.urlparse(self.path)
            try:
                track_id = int(urllib.parse.parse_qs(parsed_path.query).get('id', [0])[0])
            except ValueError:
                track_id = 0
                
            greedy = greedy_recommend(tracks, K=5)
            similar = content_filtering_recommend(tracks, [track_id], K=5)
            hybrid = graph_dpp_rerank_recommend(tracks, [track_id], K=5, min_niche_pct=0.20)
            
            def format_recs(recs):
                return [{"name": r["track_name"], "artist": r["artists"], "popularity": r["popularity"]} for r in recs]
                
            response = {
                "greedy": format_recs(greedy),
                "content_filtering": format_recs(similar),
                "graph_dpp_rerank": format_recs(hybrid)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

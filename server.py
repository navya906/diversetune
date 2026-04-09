import http.server
import socketserver
import json
import urllib.parse
from recommendation_engine import load_and_preprocess, greedy_recommend, content_filtering_recommend, graph_dpp_rerank_recommend

PORT = 8000
tracks = load_and_preprocess("data/spotify_tracks.csv")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API: Search tracks
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
            # return top 10 matches
            self.wfile.write(json.dumps(results[:10]).encode())
            return
            
        # API: Get recommendations for a track ID
        elif self.path.startswith('/api/recommend'):
            parsed_path = urllib.parse.urlparse(self.path)
            try:
                track_id = int(urllib.parse.parse_qs(parsed_path.query).get('id', [0])[0])
            except ValueError:
                track_id = 0
                
            greedy = greedy_recommend(tracks, K=5)
            similar = content_filtering_recommend(tracks, [track_id], K=5)
            graph_dpp_rerank = graph_dpp_rerank_recommend(tracks, [track_id], K=5, min_niche_pct=0.20)
            
            def format_recs(recs):
                return [{"name": r["track_name"], "artist": r["artists"], "popularity": r["popularity"]} for r in recs]
                
            response = {
                "greedy": format_recs(greedy),
                "content_filtering": format_recs(similar),
                "graph_dpp_rerank": format_recs(graph_dpp_rerank)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return
            
        # Default behavior: serve files (index.html, JS, CSS)
        else:
            # Special route for base URL
            if self.path == '/':
                self.path = '/index.html'
            super().do_GET()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

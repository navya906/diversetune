import os

files_to_update = ['index.html', 'search.html', 'index.css', 'app.js', 'server.py']

replacements = {
    'Hybrid DPP': 'Graph DPP Rerank',
    'hybrid': 'graph_dpp_rerank',
    'Hybrid': 'Graph DPP Rerank',
    'similarity': 'content_filtering',
    'Similarity': 'Content Filtering',
    'Gini Index': 'Popularity Concentration Index',
    'Gini': 'Popularity Concentration',
    'gini': 'popularity_concentration'
}

for file_name in files_to_update:
    if not os.path.exists(file_name):
        continue
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)

print("Renaming completed successfully.")

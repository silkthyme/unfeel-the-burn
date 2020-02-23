from flask import Flask, send_from_directory
import os

static_path = os.path.join('client')

app = Flask(__name__)

@app.route('/', methods=['GET'], defaults={'path': 'index.html'})  # redirect initial homepage requests to static files controller
@app.route('/<path:path>', methods=['GET'])
def static_file(path):
    """Return any requested static files"""
    full_path = os.path.join(static_path, path)
    if not os.path.exists(full_path):
        return send_from_directory(static_path, 'index.html')

    return send_from_directory(static_path, path)
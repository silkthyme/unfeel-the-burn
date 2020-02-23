import base64
import csv
from flask import Flask, make_response, request, send_from_directory
import io
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread

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


@app.route('/images/transform', methods=['POST'])
def transform():
    # imgdata = base64.b64decode()
    img = imread(request.files['image'])
    smart_img = resize(img, (100, 100, 3), preserve_range=True)
    smart_img_np_arr = np.array(smart_img).astype(int)
    flattened_img_np_arr = smart_img_np_arr.flatten()
    img_arr = flattened_img_np_arr.tolist()
    si = io.StringIO()
    csv_writer = csv.writer(si)
    csv_writer.writerow(img_arr)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=image.csv"
    output.headers["Content-type"] = "text/csv"
    return output

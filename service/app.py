import base64
import csv
from flask import Flask, make_response, request, send_from_directory
import io
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

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
    si = io.StringIO()
    csv_writer = csv.writer(si, delimiter=',')

    imgs = request.files.getlist("images")
    for img in imgs:
        parsed_img = imread(img)
        smart_img = resize(parsed_img, (100, 100, 3), preserve_range=True)
        smart_img_np_arr = np.array(smart_img).astype(int)
        flattened_img_np_arr = smart_img_np_arr.flatten()
        img_arr = flattened_img_np_arr.tolist()
        csv_writer.writerow(img_arr)

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=image.csv"
    output.headers["Content-type"] = "text/csv"

    return classify(output)

@app.route('/images/classify', methods=['POST'])
def transform2():
    imgdata = base64.b64decode(request.json['image'])
    with open('preimage.jpg', 'wb') as f:
        f.write(imgdata)
    img = imread('preimage.jpg')
    smart_img = resize(img, (100, 100, 3), preserve_range=True)
    smart_img_np_arr = np.array(smart_img).astype(int)
    flattened_img_np_arr = smart_img_np_arr.flatten()

    return classify(flattened_img_np_arr.reshape(1, -1))

def classify(table) :
    f = open('model2.py', 'rb')
    classifier = pickle.load(f)
    prediction = classifier.predict(table)
    print(prediction[0])
    return prediction[0]

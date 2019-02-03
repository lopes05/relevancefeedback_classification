import sys
from image import *
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import requests
import cv2
import logging
from image import CBIR

logging.basicConfig(filename='backend.log', level=logging.DEBUG)
logger = logging.getLogger('backend')

UPLOAD_FOLDER = 'queryfolder/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
from flask_cors import CORS
app = Flask(__name__, static_url_path='/corel1000')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


cbir = CBIR()

@app.route("/")
def hello():
    global cbir
    cbir = CBIR()
    return render_template('homepage.html')

import json as js

@app.route("/imagesearch", methods=['POST'])
def search():
    try:
        image = request.files['imgurl']
        
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        global cbir
        hists = cbir.ranking(f"{app.config['UPLOAD_FOLDER']}{filename}")[:20]
        hists = list(map(lambda k: k[1], hists))
        response = app.response_class(
            response=js.dumps(hists),
            status=200,
            mimetype='application/json'
        )
        return response


    except Exception as e:
        response = app.response_class(
            response=str(e),
            status=400
        )
        return response

@app.route('/refilter', methods=['POST'])
def refilter():
    
    try:
        global cbir
        dados = request.get_json()
        hists = cbir.refilter(dados)

        response = app.response_class(
            response=js.dumps(hists),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=str(e),
            status=400
        )
        return response


  
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000)
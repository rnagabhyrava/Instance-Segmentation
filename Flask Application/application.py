from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
import imageio

import sys
import os
import matplotlib.pyplot as plt
from mrcnn import visualize
sys.path.append(os.path.abspath("./model"))
from load import *
application = app = Flask(__name__)
global model, graph
model = init()

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        exists = os.path.isfile('./output.jpg')
        if exists:
            os.remove("./output.jpg")
        os.rename('./'+filename,'./'+'output.jpg')

        def get_ax(rows=1, cols=1, size=16):
            _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
            return ax

        image = imageio.imread('output.jpg')
        results = model.detect([image], verbose=1)
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=ax,title="Predictions")
        exists = os.path.isfile('./static/img/prediction.jpg')
        if exists:
            os.remove("./static/img/prediction.jpg")
        plt.savefig('./static/img/prediction.jpg',bbox_inches='tight', pad_inches=-0.5,orientation= 'landscape')

        plt.close()
        return render_template("index2.html")

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	app.run(debug=True)

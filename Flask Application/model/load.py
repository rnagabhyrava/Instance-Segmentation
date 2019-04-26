import os
import sys

ROOT_DIR = os.path.abspath("./assets/")

sys.path.append(ROOT_DIR)

import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

from samples.pcdscript import main

class_names = ['BG', 'person', 'cat', 'dog']

config = main.PCDConfig()

class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

def init():

	model = modellib.MaskRCNN(mode="inference", model_dir=ROOT_DIR,config=config)
	weights_path = "mask_rcnn_pcd_0001.h5"
	print("Loading weights ", weights_path)
	model.load_weights(weights_path, by_name=True)
	model.keras_model._make_predict_function()

	return model

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#log directory
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class PCDConfig(Config):
    NAME = "pcd" 
    
    GPU_COUNT = 1

    IMAGES_PER_GPU = 2

    # Number of classes
    NUM_CLASSES = 1 + 3 # 3 + 1 background 
    
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 75 #steps = total images/no of images per gpu

    # Skipping the detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.6


#  Dataset

class PCDDataset(utils.Dataset):

    def load_PCD(self, dataset_dir, subset):
        # Add classes. We have only three class to add.
        self.add_class("pcd", 1, "person")
        self.add_class("pcd", 2, "cat")
        self.add_class("pcd", 3, "dog")

        # Train or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

       
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes'] for s in a['regions']]
            num_ids = [int(n['class']) for n in objects]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "pcd", 
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,num_ids=num_ids)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "pcd":  
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "pcd":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):    # Training dataset.
    dataset_train = PCDDataset()
    dataset_train.load_PCD(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PCDDataset()
    dataset_val.load_PCD(args.dataset, "val")
    dataset_val.prepare()
    
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    #augmentation = iaa.SomeOf((0, 2), [
    #    iaa.Fliplr(0.5),
    #    iaa.Flipud(0.5),
    #    iaa.OneOf([iaa.Affine(rotate=90),
    #               iaa.Affine(rotate=180),
    #               iaa.Affine(rotate=270)]),
    #    iaa.Multiply((0.8, 1.5)),
    #    iaa.GaussianBlur(sigma=(0.0, 5.0))
    #])

#     print("Training network heads")
#     model.train(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE,
#                #augmentation=augmentation,
#                 layers='heads')
    print("Training network layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                #augmentation=augmentation,
                layers='4+')
#     print("Training network All")
#     model.train(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE,
#                 epochs=20,
#                 #augmentation=augmentation,
#                 layers='all')



#  Training
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the PCD dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PCDConfig()
    else:
        class InferenceConfig(PCDConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load coco or imagenet or last for training with previous weights
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

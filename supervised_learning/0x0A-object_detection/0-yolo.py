#!/usr/bin/env python3
"""This module contains a class that uses the Yolo ve algorithm, to perform
object detection"""

import tensorflow.keras as K


class Yolo:
    """This class uses the Yolo ve algorithm, to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """This function defines the attributes of the Yolo class"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.nms_t = nms_t
        self.anchors = anchors
        self.class_t = class_t

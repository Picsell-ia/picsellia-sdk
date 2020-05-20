import cv2
import cvxpy
import io
import json
import numpy as np
import os
import requests
import time
from PIL import Image, ImageDraw
from picsellia.exceptions import *
import sys
import easygui
import random
import logging
from picsellia.Client import Client

class FileProcessor(Client):
    """
        The Picsell.ia FileProcessor
        It provides top-level functions to :
                                            - format data for training

        """

    def generate_labelmap(self):
        """ /!\ THIS FUNCTION IS MAINTAINED FOR TENSORFLOW 1.X /!\
        ----------------------------------------------------------

        Genrate the labelmap.pbtxt file needed for Tensorflow training at:

            - project_id/
                network_id/
                    training_id/
                        label_map.pbtxt



        Raises:
            ResourceNotFoundError : If no annotations in the Picsell.ia Client yet or images can't be downloaded
                                    If no directories have been created first.

        """
        print("Generating labelmap ...")
        if not hasattr(self, "dict_annotations") or not hasattr(self, "base_dir"):
            raise ResourceNotFoundError("Please client.init model() and client.dl_annotation()")

        self.label_path = os.path.join(self.base_dir, "label_map.pbtxt")

        if not "categories" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation() first")

        categories = self.dict_annotations["categories"]
        labels_Network = {}
        try:
            with open(self.label_path, "w+") as labelmap_file:
                for k, category in enumerate(categories):
                    name = category["name"]
                    labelmap_file.write("item {\n\tname: \"" + name + "\"" + "\n\tid: " + str(k + 1) + "\n}\n")
                    if self.project_type == 'classification':
                        labels_Network[str(k)] = name
                    else:
                        labels_Network[str(k + 1)] = name
                labelmap_file.close()
            print("Label_map.pbtxt crée @ {}".format(self.label_path))

        except:
            raise ResourceNotFoundError("No directory found, please call init_model() function first")

        self.label_map = labels_Network

        return self.label_path

    def send_labelmap(self, label_path=None):
        """Attach to network, it allow nicer results visualisation on hub playground
        """

        if label_path is not None:
            if not os.path.isfile(label_path):
                raise FileNotFoundError("label map @ %s doesn't exists" % label_path)
            with open(label_path, 'r') as f:
                label_map = json.load(f)
            label = {}
            for k,v in label_map.items():
                if len(k) < 3 and not all(map(str.isdigit, k)):
                    label[v] = k

        if not hasattr(self, "label_map") and label_path is None:
            raise ValueError("Please Generate label map first")

        if label_path is not None:
            to_send = {"project_token": self.project_token, "labels": label, "network_id": self.network_id}
        else:
            to_send = {"project_token": self.project_token, "labels": self.label_map, "network_id": self.network_id}

        try:
            r = requests.get(self.host + 'attach_labels', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError("Could not connect to picsellia backend")
        if r.status_code != 201:
            print(r.text)
            raise ValueError("Could not upload label to server")

    def tf_vars_generator(self, label_map, ensemble='train', annotation_type="polygon"):
        """ /!\ THIS FUNCTION IS MAINTAINED FOR TENSORFLOW 1.X /!\

        Generator for variable needed to instantiate a tf example needed for training.

        Args :
            label_map (tf format)
            ensemble (str) : Chose between train & test
            annotation_type: "polygon", "rectangle" or "classification"

        Yields :
            (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                   encoded_jpg, image_format, classes_text, classes, masks)

        Raises:
            ResourceNotFoundError: If you don't have performed your trained test split yet
                                   If images can't be opened

        """

        if annotation_type not in ["polygon", "rectangle", "classification"]:
            raise InvalidQueryError("Please select a valid annotation_type")

        if ensemble == "train":
            path_list = self.train_list
            id_list = self.train_list_id
        else:
            path_list = self.eval_list
            id_list = self.eval_list_id

        for path, ID in zip(path_list, id_list):
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            masks = []

            internal_picture_id = ID

            with open(path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size
            filename = path.encode('utf8')
            image_format = '{}'.format(path.split('.')[-1])
            image_format = bytes(image_format.encode('utf8'))

            if annotation_type=="polygon":
                for image_annoted in self.dict_annotations["annotations"]:
                    if internal_picture_id == image_annoted["internal_picture_id"]:
                        for a in image_annoted["annotations"]:
                            try:
                                if "polygon" in a.keys():
                                    geo = a["polygon"]["geometry"]
                                    poly = []
                                    for coord in geo:
                                        poly.append([[coord["x"], coord["y"]]])

                                    poly = np.array(poly, dtype=np.float32)
                                    mask = np.zeros((height, width), dtype=np.uint8)
                                    mask = Image.fromarray(mask)
                                    ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
                                    maskByteArr = io.BytesIO()
                                    mask.save(maskByteArr, format="PNG")
                                    maskByteArr = maskByteArr.getvalue()
                                    masks.append(maskByteArr)

                                    x, y, w, h = cv2.boundingRect(poly)
                                    xmins.append(x / width)
                                    xmaxs.append((x + w) / width)
                                    ymins.append(y / height)
                                    ymaxs.append((y + h) / height)
                                    classes_text.append(a["label"].encode("utf8"))
                                    label_id = label_map[a["label"]]
                                    classes.append(label_id)

                            except:
                                pass
                yield (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                encoded_jpg, image_format, classes_text, classes, masks)

            if annotation_type=="rectangle":
                for image_annoted in self.dict_annotations["annotations"]:
                    if internal_picture_id == image_annoted["internal_picture_id"]:
                        for a in image_annoted["annotations"]:
                            try:
                                if 'rectangle' in a.keys():
                                    xmin = a["rectangle"]["top"]
                                    xmax = xmin + a["rectangle"]["width"]
                                    ymin = a["rectangle"]["left"]
                                    ymax = ymin + a["rectangle"]["height"]
                                    xmins.append(xmin/width)
                                    xmaxs.append(xmax/width)
                                    ymins.append(ymin/height)
                                    ymaxs.append(ymax/height)
                                    classes_text.append(a["label"].encode("utf8"))
                                    label_id = label_map[a["label"]]
                                    classes.append(label_id)
                            except:
                                pass

                yield (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                    encoded_jpg, image_format, classes_text, classes)

            if annotation_type=="classification":
                for image_annoted in self.dict_annotations["annotations"]:
                    if internal_picture_id == image_annoted["internal_picture_id"]:
                        for a in image_annoted["annotations"]:
                            classes_text.append(a["label"].encode("utf8"))
                            label_id = label_map[a["label"]]
                            classes.append(label_id)

                yield (width, height, filename, encoded_jpg, image_format,
                    classes_text, classes)
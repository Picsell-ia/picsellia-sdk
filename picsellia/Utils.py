import cv2
import io
import json
import numpy as np
import os
import requests
import time
from PIL import Image, ImageDraw
from picsellia.exceptions import *
import sys
import random
import logging




def display_project_state(project_name, project_type, project_infos=None, network_names=None):
    if project_infos is not None and network_names is not None:
        print(
            "Current stateof project : {}\nThis is a {} project".format(
                project_name, project_type))
        for i, col in enumerate(project_infos):

            print("-" * 15)
            print("{} training version(s) for Network named : {}".format(len(col), network_names[i]))
            print("-" * 15)
            for training in col:
                print("\t For training id {}:".format(training["training_id"]))
                if training["is_datasplit"]:
                    print("\t\t Train Test Set repartition : DONE")
                else:
                    print("\t\t Train Test Set repartition : NOT DONE")
                if training["is_examples"]:
                    print("\t\t Visual results uploaded to Picsell.ia : DONE")
                else:
                    print("\t\t Visual results uploaded to Picsell.ia : NOT DONE")
                if training["is_metrics"]:
                    print("\t\t Training logs uploaded to Picsell.ia : DONE")
                else:
                    print("\t\t Training logs uploaded to Picsell.ia : NOT DONE")

                print("\t\t Model usable from Picsell.ia : DONE")

    elif project_infos is None and network_names is not None:
        print("-" * 80)
        print("Welcome to Picsell.ia Client, this project_token is linked to your project : {}".format(
            project_name))
        print("-" * 80)
        print("You don't have any Network trained for this project yet.\n")
        print("{} Network(s) attached to your project:".format(len(network_names)))
        for e in network_names:
            print("\t - {}".format(e))
        print("\nTo initialise a training session, please run create_network(network_name)\n")

    elif project_infos is None and network_names is None:
        print("Welcome to Picsell.ia Client, this project_token is linked to your project : {}\n".format(
            project_name))
        print(
            "You don't have any Network attache to this project yet.\nIf you want to continue without an attached model, please initialise it with init_model(YOUR NAME)")



def train_valid_split_obj_simple(dict_annotations, prop=0.8):
    """Perform Optimized train test split for Object Detection.
       Uses optimization to find the optimal split to have the desired repartition of instances by set.

    Args:
        prop (float) : Percentage of Instances used for training.
        dict_annotations (dict) : annotation from dl_annotations

    Raises:
        ResourceNotFoundError: If not annotations in the Picsell.ia Client yet.
    """
    if dict_annotations is None:
        raise ResourceNotFoundError("No dict_annotations passed")

    list_im = np.linspace(0, len(dict_annotations['images']) - 1, len(dict_annotations['images'])).astype('int')
    random.shuffle(list_im)
    nb_im = int(prop * len(dict_annotations['images']))
    train_list = list_im[:nb_im]
    test_list = list_im[nb_im:]

    index_url = []
    for e in range(len(list_im)):
        if e in train_list:
            index_url.append(1)
        elif e in test_list:
            index_url.append(0)

    return index_url


def get_labels_repartition_obj_detection(dict_annotations, index_url):
    """Perform train test split scanning for Object Detection.

    Returns:
        cate (array[str]) : Array of the classes names
        cnt_train (array[int]) : Array of the number of object per class for the training set.
        cnt_eval (array[int]) : Array of the number of object per class for the evaluation set.

    Raises:
        ResourceNotFoundError: If not annotations in the Picsell.ia Client yet.
    """

    if dict_annotations is None:
        raise ResourceNotFoundError("No dict_annotations passed")

    cate = [v["name"] for v in dict_annotations["categories"]]
    cnt_train = [0] * len(cate)
    cnt_eval = [0] * len(cate)

    for img, index in zip(dict_annotations['images'], index_url):

        internal_picture_id = img["internal_picture_id"]
        for ann in dict_annotations["annotations"]:
            if internal_picture_id == ann["internal_picture_id"]:
                for an in ann['annotations']:
                    idx = cate.index(an['label'])
                    if index == 1:
                        cnt_train[int(idx)] += 1
                    else:
                        cnt_eval[int(idx)] += 1

    return cnt_train, cnt_eval, cate

def is_checkpoint(path, project_type):

    index_path = ""
    data_path = ""
    config_path = ""

    try:
        if not os.path.isdir(path):
            return False
    except:
        return False

    for f in os.listdir(path):
        print(f)
        if ".index" in f:
            index_path = f
        if ".data" in f:
            data_path = f
        if project_type != "classification":
            if "pipeline" in f:
                config_path = f


    if index_path == "" or data_path == "":
        return False

    elif config_path == "" and project_type != "classification":
        return False

    return True

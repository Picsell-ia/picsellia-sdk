import json
import os
import sys
import io
import picsellia.Utils as utils
import requests
from PIL import Image
from picsellia.exceptions import *
import numpy as np
import cv2
import sys
from multiprocessing.pool import ThreadPool

if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

class Client:
    """
    The Picsell.ia Client contains info necessary for connecting to the Picsell.ia Platform.
    It provides top-level functions to :
                                        - format data for training
                                        - dl annotations & images
                                        - send training logs
                                        - send examples
                                        - save weights and SavedModel to Picsell.ia server.

    """

    def __init__(self, api_token=None, host="https://app.picsellia.com/sdk/"):
        """ Creates and initializes a Picsell.ia Client.
        Args:
            project_token (str): project_token key, given on the platform.
            host (str): URL of the Picsell.ia server to connect to.
        Raises:
            InvalidQueryError: If no 'project_token' provided as an argument.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
        """

        if api_token is None:
            raise InvalidQueryError("api_token argument not provided")

        self.auth = {"Authorization": "Bearer " + api_token}
        self.host = host
        try:
            r = requests.get(self.host + 'ping', headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")
        self.project_name_list = r.json()["project_list"]
        self.username = r.json()["username"]
        self.supported_img_types = ("png", "jpg", "jpeg")
        self.label_path = ""
        print(
            "Welcome {}, Glad to have you back".format(
                self.username))

    def checkout_project(self, project_token=None, png_dir=None):
        """ Attach the Picsell.ia Client to the desired project.
                Args:
                    project_token (str): project_token key, given on the platform.
                    png_dir (str): path to your images, if None you can dl pic
                Raises:
                    InvalidQueryError: If no 'project_token' provided as an argument.
                    NetworkError: If Picsell.ia server not responding or host is incorrect.
                """
        if project_token is None:
            raise InvalidQueryError("project_token argument not provided")

        to_send = {"project_token": project_token}
        try:
            r = requests.get(self.host + 'init_project', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")
        if r.status_code != 200:
            raise AuthenticationError(
                'The project_token provided does not match any of the known project_token for profile.')

        self.project_token = project_token
        self.project_id = r.json()["project_id"]
        self.project_infos = r.json()["infos"]
        self.project_name = r.json()["project_name"]
        self.project_type = r.json()["project_type"]
        self.network_names = r.json()["network_names"]

        if png_dir is None:
            self.png_dir = os.path.join(self.project_name, 'images')
        else:
            self.png_dir = png_dir
            print("Looking for images @ %s ..." % self.png_dir)
            if not len(os.listdir(self.png_dir)) != 0:
                raise ResourceNotFoundError("Can't find images at %s" % (self.png_dir))

            for filename in os.listdir(self.png_dir):
                ext = filename.split('.')[-1]
                if not ext in self.supported_img_types:
                    raise ResourceNotFoundError(
                        "Found a non supported filetype (%s) in your png_dir " % (filename.split('.')[-1]))

    def create_network(self, network_name, orphan=False):
        """ Initialise the NeuralNet instance on Picsell.ia server.
            If the model name exists on the server for this project, you will create a new version of your training.

            Create all the repositories for your training with this architecture :

              your_code.py
              - project_id
                    - images/
                    - network_id/
                        - training_version/
                            - logs/
                            - checkpoints/
                            - records/
                            - config/
                            - results/
                            - exported_model/

        Args:
            network_name (str): It's simply the name you want to give to your NeuralNet
                              For example, SSD_Picsellia

        Raises:
            AuthenticationError: If `project_token` does not match the provided project_token on the platform.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
        """

        assert isinstance(network_name, str), "model name must be string, got %s" % type(network_name)

        if self.network_names is None:
            self.network_names = []

        if network_name in self.network_names:
            raise InvalidQueryError("The Network name you provided already exists for this project")

        to_send = {"model_name": network_name, "project_token": self.project_token}

        try:
            r = requests.get(self.host + 'init_model', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")

        if r.status_code == 400:
            raise AuthenticationError(
                'The project_token provided does not match any of the known project_token for profile.')

        self.network_id = r.json()["network_id"]
        self.training_id = r.json()["training_id"]
        self.network_name = network_name
        self.dict_annotations = {}

        if orphan is not True:
            self.setup_dirs()
        else:
            self.base_dir = os.path.join(self.project_name, self.network_name, str(self.training_id))
            self.metrics_dir = os.path.join(self.base_dir, 'metrics')
            self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')
            self.record_dir = os.path.join(self.base_dir, 'records')
            self.config_dir = os.path.join(self.base_dir, 'config')
            self.results_dir = os.path.join(self.base_dir, 'results')
            self.exported_model_dir = os.path.join(self.base_dir, 'exported_model')


        print("New Network have been created")
        return None

    def checkout_network(self, network_name, training_id=None):
        """ Attach the Picsell.ia Client to the desired Network.
            If the model name exists on the server for this project, you will create a new version of your training.

            Create all the repositories for your training with this architecture :

              your_code.py
              - project_id
                    - images/
                    - network_id/
                        - training_version/
                            - logs/
                            - checkpoints/
                            - records/
                            - config/
                            - results/
                            - exported_model/

        Args:
            network_name (str): It's simply the name you want to give to your NeuralNet
                              For example, SSD_Picsellia

        Raises:
            AuthenticationError: If `project_token` does not match the provided project_token on the platform.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
        """

        assert isinstance(network_name, str), "model name must be string, got %s" % type(network_name)

        if network_name not in self.network_names:
            raise ResourceNotFoundError("The Network name you provided does not exists for this project")

        to_send = {"model_name": network_name, "project_token": self.project_token}

        try:
            r = requests.get(self.host + 'init_model', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")

        if r.status_code == 400:
            raise AuthenticationError(
                'The project_token provided does not match any of the known project_token for profile.')

        response = r.json()
        self.network_id = response["network_id"]
        self.training_id = response["training_id"]
        if training_id is not None:
            self.training_id=training_id
        self.network_name = network_name
        self.dict_annotations = {}
        if "index_object_name" in response["checkpoints"].keys():
            self.checkpoint_index =  response["checkpoints"]["index_object_name"]
        else:
            self.checkpoint_index = None

        if "data_object_name" in response["checkpoints"].keys():
            self.checkpoint_data =  response["checkpoints"]["data_object_name"]

        else:
            self.checkpoint_data = None

        if "config_file" in response["checkpoints"].keys():
            self.config_file =  response["checkpoints"]["config_file"]
        else:
            self.config_file = None
        self.setup_dirs()
        self.model_selected = self.dl_checkpoints()
        print("You already have some checkpoints on your machine, we'll start training from there.")
        return self.model_selected

    def configure_network(self, project_type=None):

        if project_type is None:
            raise InvalidQueryError("project type not provided")

        supported_type = ["classification", "detection", "segmentation"]
        if project_type not in supported_type:
            ok = False
            while not ok:
                a = input("Please provide a supported project type : {}".format(supported_type))
                if a in supported_type:
                    if a == "classification" and self.project_type != "classification":
                        print("You tried to configure you project with an incompatible project type, you project type is {}".format(self.project_type))
                    else:
                        ok = True

            to_send = {"network_id": self.network_id, "type": a}

        else:
            to_send = {"network_id": self.network_id, "type": project_type}

        try:
            r = requests.post(self.host + 'configure_network', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")

        if not r.status_code == 200:
            raise ResourceNotFoundError("Invalid network to configure")

        if project_type == "classification":
            self.annotation_type = "label"
        elif project_type == "detection":
            self.annotation_type = "rectangle"
        elif project_type == "segmentation":
            self.annotation_type = "polygon"



    def reset_network(self, network_name):
        """ Reset your training checkpoints to the origin.

        Args:
            network_name (str): It's simply the name you want to give to your NeuralNet
                              For example, SSD_Picsellia

        Raises:
            AuthenticationError: If `project_token` does not match the provided project_token on the platform.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
        """

        assert isinstance(network_name, str), "model name must be string, got %s" % type(network_name)

        print("Ok.. We'll reset your project to the origin checkpoint")
        if network_name not in self.network_names:
            raise ResourceNotFoundError("The Network name you provided does not exists for this project")

        to_send = {"model_name": network_name, "project_token": self.project_token, "reset": True}

        try:
            r = requests.get(self.host + 'init_model', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")

        if r.status_code == 400:
            raise AuthenticationError(
                'The project_token provided does not match any of the known project_token for profile.')

        response = r.json()
        self.network_id = response["network_id"]
        self.training_id = response["training_id"]
        self.network_name = network_name
        self.dict_annotations = {}
        self.setup_dirs()
        if "index_object_name" in response["checkpoints"].keys():
            self.checkpoint_index =  response["checkpoints"]["index_object_name"]
        else:
            self.checkpoint_index = None

        if "data_object_name" in response["checkpoints"].keys():
            self.checkpoint_data =  response["checkpoints"]["data_object_name"]

        else:
            self.checkpoint_data = None

        if "config_file" in response["checkpoints"].keys():
            self.config_file =  response["checkpoints"]["config_file"]
        else:
            self.config_file = None
        self.model_selected = self.dl_checkpoints(reset=True)
        print("You already have some checkpoints on your machine, we'll start training from there.")
        return self.model_selected

    def setup_dirs(self):

        self.base_dir = os.path.join(self.project_name, self.network_name, str(self.training_id))
        self.metrics_dir = os.path.join(self.base_dir, 'metrics')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')
        self.record_dir = os.path.join(self.base_dir, 'records')
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.exported_model_dir = os.path.join(self.base_dir, 'exported_model')

        if not os.path.isdir(self.project_name):
            print("First time using Picsell.ia on this project, initializing directories ...")
            os.mkdir(self.project_name)

        if not os.path.isdir(os.path.join(self.project_name, self.network_name)):
            os.mkdir(os.path.join(self.project_name, self.network_name))

        if not os.path.isdir(self.base_dir):
            print("Creating directory for project {}".format(self.base_dir))
            os.mkdir(self.base_dir)

        if not os.path.isdir(self.png_dir):
            print("Creating directory for PNG Images of project {}".format(self.base_dir))
            os.mkdir(self.png_dir)

        if not os.path.isdir(self.checkpoint_dir):
            print("Creating directory for checkpoints project {}".format(self.base_dir))
            os.mkdir(self.checkpoint_dir)

        if not os.path.isdir(self.metrics_dir):
            print("Creating directory for logs of project {}".format(self.metrics_dir))
            os.mkdir(self.metrics_dir)

        if not os.path.isdir(self.record_dir):
            print("Creating directory for records of project {}".format(self.base_dir))
            os.mkdir(self.record_dir)

        if not os.path.isdir(self.config_dir):
            print("Creating directory for config of project {}".format(self.base_dir))
            os.mkdir(self.config_dir)

        if not os.path.isdir(self.results_dir):
            print("Creating directory for results of project {}".format(self.results_dir))
            os.mkdir(self.results_dir)

        if not os.path.isdir(self.exported_model_dir):
            print("Creating directory for results of project {}".format(self.exported_model_dir))
            os.mkdir(self.exported_model_dir)

    def dl_checkpoints(self, checkpoint_path=None, reset=False):

        if checkpoint_path is not None:
            if not os.path.isdir(checkpoint_path):
                raise ResourceNotFoundError("No directory @ %s" % checkpoint_path)
            return checkpoint_path






        if (self.checkpoint_index is None) or (self.checkpoint_data is None):
            print("You are working with a Custom model, you should start from a pre-trained network.")
            return None


        # list all existing training id
        if not reset:
            list_training = os.listdir(os.path.join(self.project_name, self.network_name))
            training_ids = sorted([int(e) for e in list_training], reverse=True)
            print("available training id , ", training_ids)

            index_path = ""
            data_path = ""
            config_path = ""
            a = 0
            for id in training_ids:
                path = os.path.join(self.project_name, self.network_name, str(id), "checkpoint")
                if utils.is_checkpoint(path, self.project_type):

                    while (a not in ["y", "yes", "n", "no"]):
                        a = input("Found checkpoints files for training %s  , do you want to use this checkpoints ? [y/n]" % (str(id)))

                    if a.lower() == 'y' or a.lower() == 'yes':
                        print("Your next training will use checkpoint stored @ %s " % os.path.join(self.project_name,
                                                                                                   self.network_name,
                                                                                                   str(self.training_id),
                                                                                               "checkpoint"))
                        return path

                    else:
                        continue

                else:
                    path_deep = os.path.join(self.project_name, self.network_name, str(id), "checkpoint", "origin")

                    if utils.is_checkpoint(path_deep, self.project_type):

                        while (a not in ["y", "yes", "n", "no"]):
                            a = input("Found original checkpoints files from training %s  , do you want to use this checkpoints ? [y/n]" % (str(id)))

                        if a.lower() == 'y' or a.lower()=='yes':
                            print("Your next training will use checkpoint stored @ %s " % os.path.join(self.project_name,
                                                                                                       self.network_name,
                                                                                                       str(self.training_id),
                                                                                                   "checkpoint"))
                            return path_deep

                        else:
                            continue
                    else:
                        continue

        else:
            try:
                path_to_look_0 = os.path.join(self.project_name, self.network_name, "0", "checkpoint", "origin")
                path_to_look_1 = os.path.join(self.project_name, self.network_name, "1", "checkpoint", "origin")
                if os.path.isdir(path_to_look_0):
                    if utils.is_checkpoint(path_to_look_0, self.project_type):
                        print("Found the originals checkpoints @ %s " % path_to_look_0)
                        return path_to_look_0
                if os.path.isdir(path_to_look_1):
                    if utils.is_checkpoint(path_to_look_1, self.project_type):
                        print("Found the originals checkpoints @ %s " % path_to_look_1)
                        return path_to_look_1
            except:
                pass



        path_to_origin = os.path.join(self.checkpoint_dir, "origin")


        if not os.path.isdir(path_to_origin):
            os.makedirs(path_to_origin)

        for fpath in os.listdir(path_to_origin):
            os.remove(os.path.join(path_to_origin,fpath))
        url_index = self._get_presigned_url('get', self.checkpoint_index, bucket_model=True)
        checkpoint_file = os.path.join(path_to_origin, self.checkpoint_index.split('/')[-1])


        with open(checkpoint_file, 'wb') as handler:
            response = requests.get(url_index, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                print("couldn't download checkpoint index file")
                self.checkpoint_index = None
            else:
                print("Downloading %s" % self.checkpoint_index)
                dl = 0
                total_length = len(response.content)
                for data in response.iter_content(chunk_size=1024):
                    dl += len(data)
                    handler.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * (done - 1) + '>', ' ' * (50 - done)))
                    sys.stdout.flush()
                print('Checkpoint Index downloaded')

        url_config = self._get_presigned_url('get', self.config_file, bucket_model=True)
        config_file = os.path.join(path_to_origin, self.config_file.split('/')[-1])


        with open(config_file, 'wb') as handler:
            print("Downloading %s" % self.config_file)
            response = requests.get(url_config, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:
                dl = 0
                total_length = int(total_length)
                print("couldn't download config file")
            else:
                print("Downloading %s" % self.checkpoint_index)
                dl = 0
                total_length = len(response.content)
                for data in response.iter_content(chunk_size=1024):
                    dl += len(data)
                    handler.write(data)
                    done = int(50 * dl / total_length)
                print('Config downloaded')

        url_data = self._get_presigned_url('get', self.checkpoint_data, bucket_model=True)
        checkpoint_file = os.path.join(path_to_origin, self.checkpoint_data.split('/')[-1])

        with open(checkpoint_file, 'wb') as handler:
            print("Downloading %s" % self.checkpoint_data)
            response = requests.get(url_data, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                print("couldn't download checkpoint data file")
                self.checkpoint_data = None
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    handler.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
                print('Checkpoint Data downloaded')

        return path_to_origin

    def dl_annotations(self, option="train"):
        """ Pull all the annotations made on Picsell.ia Platform for your project.

            Args:
                option (str): Define what time of annotation to export (accepted or all)

            Raises:
                AuthenticationError: If `project_token` does not match the provided project_token on the platform.
                NetworkError: If Picsell.ia server not responding or host is incorrect.
                ResourceNotFoundError: If we can't find any annotations for that project.
            """

        print("Downloading annotations of project {} ...".format(self.project_token))

        try:
            to_send = {"project_token": self.project_token, "type": option}
            r = requests.get(self.host + 'annotations', data=json.dumps(to_send), headers=self.auth)

            if r.status_code != 200:
                return ResourceNotFoundError("There is no annotations found for this project")

            print("Annotations pulled ...")
            self.dict_annotations = r.json()

            if len(self.dict_annotations.keys()) == 0:
                raise ResourceNotFoundError("You don't have any annotations")

        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")

    def dl_latest_saved_model(self, path_to_save=None):
        """ Pull the latest  Picsell.ia Platform for your project.

                    Args:
                        option (str): Define what time of annotation to export (accepted or all)

                    Raises:
                        AuthenticationError: If `project_token` does not match the provided project_token on the platform.
                        NetworkError: If Picsell.ia server not responding or host is incorrect.
                        ResourceNotFoundError: If we can't find any annotations for that project.
                    """
        if path_to_save is None:
            raise InvalidQueryError("Please precise where you want to save .pb file.")
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        if not hasattr(self, "training_id"):
            raise ResourceNotFoundError("Please init model first")

        if not hasattr(self, "auth"):
            raise ResourceNotFoundError("Please init client first")

        to_send = {"project_token": self.project_token, "network_id": self.network_id}
        try:
            r = requests.get(self.host + 'get_saved_model_object_name', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError("Could not connect to Picsell.ia Backend")

        object_name = r.json()["object_name"]
        if object_name == 0:
            raise ValueError("There is no saved model on our backend for this project")

        url = self._get_presigned_url("get", object_name, bucket_model=True)

        with open(os.path.join(path_to_save, 'saved_model.pb'), 'wb') as handler:
            print("Downloading exported model..")
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                handler.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
            print('Exported model downloaded @ %s' % path_to_save)

    def dl_pictures(self):
        """Download your training set on the machine (Use it to dl images to Google Colab etc.)
           Save it to /project_id/images/*
           Perform train_test_split & send the repartition to Picsell.ia Platform

        Raises:
            ResourceNotFoundError : If no annotations in the Picsell.ia Client yet or images can't be downloaded
            ProcessingError: If the train test split can't be performed.

        """

        if not hasattr(self, "dict_annotations"):
            raise ResourceNotFoundError("Please dl_annotation model with dl_annotation()")

        if not "images" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")

        cnt = 0

        print("Downloading PNG images to your machine ...")

        dl = 0
        total_length = len(self.dict_annotations["images"])
        for info in self.dict_annotations["images"]:

            pic_name = os.path.join(self.png_dir, info['external_picture_url'].split('/')[-1])
            if not os.path.isdir(self.png_dir):
                os.makedirs(self.png_dir)
            if not os.path.isfile(pic_name):
                try:
                    response = requests.get(info["signed_url"], stream=True)
                    with open(pic_name, 'wb') as handler:
                        for data in response.iter_content(chunk_size=1024):
                            handler.write(data)
                    cnt += 1
                except:
                    print("Image %s can't be downloaded" % pic_name)
                    pass

            dl += 1
            done = int(50 * dl / total_length)
            sys.stdout.flush()
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))



        print("{} files were already on your machine".format(total_length - cnt))
        print(" {} PNG images have been downloaded to your machine".format(cnt))






    def train_test_split(self, prop=0.8):

        if not hasattr(self, "dict_annotations"):
            raise ResourceNotFoundError("Please dl_annotation model with dl_annotation()")

        if not "images" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")

        self.train_list = []
        self.eval_list = []
        self.train_list_id = []
        self.eval_list_id = []
        self.index_url = utils.train_valid_split_obj_simple(self.dict_annotations, prop)

        total_length = len(self.dict_annotations["images"])
        for info, idx in zip(self.dict_annotations["images"], self.index_url):
            pic_name = os.path.join(self.png_dir, info['external_picture_url'])
            if idx == 1:
                self.train_list.append(pic_name)
                self.train_list_id.append(info["internal_picture_id"])
            else:
                self.eval_list.append(pic_name)
                self.eval_list_id.append(info["internal_picture_id"])

        print("{} Images used for training, {} Images used for validation".format(len(self.train_list_id),
                                                                                  len(self.eval_list_id)))

        label_train, label_test, cate = utils.get_labels_repartition_obj_detection(self.dict_annotations, self.index_url)

        to_send = {"project_token": self.project_token,
                   "train": {"train_list_id": self.train_list_id, "label_repartition": label_train, "labels": cate},
                   "eval": {"eval_list_id": self.eval_list_id, "label_repartition": label_test, "labels": cate},
                   "network_id": self.network_id, "training_id": self.training_id}

        try:
            r = requests.post(self.host + 'post_repartition', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 201:
                raise NetworkError('Can not send repartition to Picsell.ia Backend')
            print("Repartition sent ..")

        except:
            raise NetworkError('Can not send repartition to Picsell.ia Backend')


    def send_logs(self, logs=None, logs_path=None):
        """Send training logs to Picsell.ia Platform

        Args:
            logs (dict): Dict of the training metric (Please find Getting Started Picsellia Docs to see how to get it)
        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no saved_model saved

        """

        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self,
                                                                                              "host") or not hasattr(
                self, "project_token"):
            raise ResourceNotFoundError("Please initialize model with init_model()")

        if logs_path is not None:
            if not os.path.isfile(logs_path):
                raise FileNotFoundError("Logs file not found")
            with open(logs_path, 'r') as f:
                logs = json.load(f)

        if logs is None and logs_path is None:
            raise ResourceNotFoundError("No log dict or path to logs .json given")

        try:
            to_send = {"project_token": self.project_token, "training_id": self.training_id, "logs": logs,
                       "network_id": self.network_id}
            r = requests.post(self.host + 'post_logs', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 201:
                raise NetworkError("The logs have not been sent because %s" % (r.text))

            print(
                "Training logs have been sent to Picsell.ia Platform...\nYou can now inspect and showcase results on the platform.")

        except:
            raise NetworkError("Could not connect to Picsell.ia Server")

    def send_metrics(self, metrics=None, metrics_path=None):
        """Send evalutation metrics to Picsell.ia Platform

        Args:
            metrics (dict): Dict of the evaluation metrics (Please find Getting Started Picsellia Docs to see how to get it)
        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no saved_model saved

        """
        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self,
                                                                                              "host") or not hasattr(
                self, "project_token"):
            raise ResourceNotFoundError("Please initialize model first")

        if metrics_path is not None:
            if not os.path.isfile(metrics_path):
                raise FileNotFoundError("Metrics file not found")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        if metrics is None and metrics_path is None:
            raise ResourceNotFoundError("No metrics dict or path to metrics.json given")

        try:
            to_send = {"project_token": self.project_token, "training_id": self.training_id, "metrics": metrics,
                       "network_id": self.network_id}
            r = requests.post(self.host + 'post_metrics', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 201:
                raise NetworkError("The evaluation metrics have not been sent because %s" % (r.text))

            print(
                "Evaluation metrics have been sent to Picsell.ia Platform...\nYou can now inspect and showcase results on the platform.")

        except:
            raise NetworkError("Could not connect to Picsell.ia Server")

    def send_examples(self, id=None, example_path_list=None):
        """Send Visual results to Picsell.ia Platform

        Args:
            id (int): Id of the training
        Raises:
            NetworkError: If it impossible to initialize upload
            FileNotFoundError:
            ResourceNotFoundError:

        """
        if id is None and example_path_list is None:
            try:
                results_dir = self.results_dir
                list_img = os.listdir(results_dir)
                assert len(list_img) != 0, 'No example have been created'
            except:
                raise ResourceNotFoundError("You didn't init_model(), please call this before sending examples")

        elif id is not None and example_path_list is None:
            base_dir = os.path.join(self.project_name, self.network_name)
            if str(id) in os.listdir(base_dir):
                results_dir = os.path.join(base_dir, str(id), 'results')
                list_img = os.listdir(results_dir)
                assert len(list_img) != 0, 'No example have been created'
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        os.path.join(base_dir, str(id) + '/results'))

        elif (id is None and example_path_list is not None) or (id is not None and example_path_list is not None):
            for file in example_path_list:
                if not os.path.isfile(file):
                    raise FileNotFoundError("file not found @ %s" % file)
            list_img = example_path_list
            results_dir = ""

        object_name_list = []
        for img_path in list_img:
            file_path = os.path.join(results_dir, img_path)
            if not os.path.isfile(file_path):
                raise FileNotFoundError("Can't locate file @ %s" % (file_path))
            if id is None and example_path_list is not None:
                OBJECT_NAME = os.path.join(self.project_id, self.network_id, str(self.training_id), "results",
                                           file_path.split('/')[-1])
            elif id is not None and example_path_list is not None:
                OBJECT_NAME = os.path.join(self.project_id, self.network_id, str(id), "results",
                                           file_path.split('/')[-1])
            else:
                OBJECT_NAME = file_path

            response = self._get_presigned_url('post', OBJECT_NAME)
            to_send = {"project_token": self.project_token, "object_name": OBJECT_NAME}

            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (OBJECT_NAME, f)}
                    http_response = requests.post(response['url'], data=response['fields'], files=files)
                    print('http:', http_response.status_code)
                if http_response.status_code == 204:
                    object_name_list.append(OBJECT_NAME)
            except:
                raise NetworkError("Could not upload examples to s3")

        to_send2 = {"project_token": self.project_token, "network_id": self.network_id,
                    "training_id": self.training_id, "urls": object_name_list}
        try:
            r = requests.post(self.host + 'post_preview', data=json.dumps(to_send2), headers=self.auth)
            if r.status_code != 201:
                print(r.text)
                raise ValueError("Errors.")
            print("A snapshot of results has been saved to the platform")
        except:
            raise NetworkError("Could not upload to Picsell.ia Backend")

    def send_model(self, file_path=None):

        """Send frozen graph for inference to Picsell.ia Platform


        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no visual results saved in /project_id/network_id/training_id/results/

        """

        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self,
                                                                                              "host") or not hasattr(
                self, "project_token"):
            raise ResourceNotFoundError("Please initialize model with init_model()")

        if file_path is not None:
            if not os.path.isdir(file_path):
                raise FileNotFoundError("You have not exported your model")

            ok = False
            for fp in os.listdir(file_path):
                if fp.endswith('.pb'):
                    ok=True
                    break
            if not ok:
                raise InvalidQueryError("wrong file type, please send a .pb file")

            file_path = self._zipdir(file_path)
            self.OBJECT_NAME = os.path.join(self.network_id, str(self.training_id), file_path.split('/')[-1])


        else:

            if os.path.isdir(os.path.join(self.exported_model_dir, 'saved_model')):
                file_path = os.path.join(self.exported_model_dir, 'saved_model')
                ok = False
                for fp in os.listdir(file_path):
                    if fp.endswith('.pb'):
                        ok=True
                        break
                if not ok:
                    raise InvalidQueryError("wrong file type, please send a .pb file")


                self.OBJECT_NAME = os.path.join(self.network_id, str(self.training_id),'saved_model.zip')
                file_path = self._zipdir(file_path)
            else:
                file_path = self.exported_model_dir
                ok = False
                liste = os.listdir(file_path)

                if "variables" in liste and "saved_model.pb" in liste:
                    ok = True

                if not ok:
                    raise InvalidQueryError("wrong file type, please send a .pb file")


                self.OBJECT_NAME = os.path.join(self.network_id, str(self.training_id),'saved_model.zip')
                file_path = self._zipdir(file_path)


        self._init_multipart()
        parts = self._upload_part(file_path)

        if self._complete_part_upload(parts, self.OBJECT_NAME, 'model'):
            print("Your exported model have been uploaded successfully to our cloud.")

    def send_checkpoints(self, index_path=None, data_path=None, config_path=None):

        """Send frozen graph for inference to Picsell.ia Platform


        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no visual results saved in /project_id/network_id/training_id/results/

        """

        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self,
                                                                                              "host") or not hasattr(
                self, "project_token"):
            raise ResourceNotFoundError("Please initialize model with init_model()")
        max_size = 5 * 1024 * 1024
        urls = []
        file_list = os.listdir(self.checkpoint_dir)
        if (index_path is not None) and (data_path is not None) and (config_path is not None):
            if not os.path.isfile(index_path):
                raise FileNotFoundError("{}: no such file".format(index_path))
            if not os.path.isfile(data_path):
                raise FileNotFoundError("{}: no such file".format(data_path))
            if not os.path.isfile(config_path):
                raise FileNotFoundError("{}: no such file".format(config_path))

            # index_name = index_path.split('/')[-1]
            # data_path = data_path.split('/')[-1]
            # config_path = config_path.split('/')[-1]

            ckpt_index_object = os.path.join(self.checkpoint_dir, index_path.split('/')[-1])
            ckpt_data_object = os.path.join(self.checkpoint_dir, data_path.split('/')[-1])
            self.OBJECT_NAME = ckpt_data_object
            if self.project_type != "classification":
                config_object = os.path.join(self.checkpoint_dir, config_path.split('/')[-1])


        elif (index_path is None) and (data_path is None) and (config_path is None):
            ckpt_id = max([int(p.split('-')[1].split('.')[0]) for p in file_list if 'index' in p])
            ckpt_index = "model.ckpt-{}.index".format(str(ckpt_id))
            ckpt_index_object = os.path.join(self.checkpoint_dir, ckpt_index)
            index_path = ckpt_index_object

            ckpt_data = None
            for e in file_list:
                if "{}.data".format(ckpt_id) in e:
                    ckpt_data = e

            if ckpt_data is None:
                raise ResourceNotFoundError("Could not fin matching data file with index")

            ckpt_data_object = os.path.join(self.checkpoint_dir, ckpt_data)
            self.OBJECT_NAME = ckpt_data_object
            data_path = ckpt_data_object
            if self.project_type != "classification":
                if not os.path.isfile(os.path.join(self.checkpoint_dir, "pipeline.config")):
                    raise FileNotFoundError("No config file found")
                config_object = os.path.join(self.checkpoint_dir, "pipeline.config")
                config_path = config_object
        else:
            raise ValueError("checkpoints' index and data  and config files must be sent together to ensure \
                              compatibility")

        self.send_checkpoint_index(index_path, ckpt_index_object)
        print("Checkpoint index saved")

        if self.project_type != "classification":
            self.send_config_file(config_path, config_object)
        print("Config file saved")

        self._init_multipart()
        parts = self._upload_part(data_path)

        if self._complete_part_upload(parts, ckpt_data_object, 'checkpoint'):
            print("Your index checkpoint have been uploaded successfully to our cloud.")

    def send_checkpoint_index(self, filename, object_name):
        response = self._get_presigned_url(method='post', object_name=object_name,
                                           bucket_model=True)
        try:
            with open(filename, 'rb') as f:
                files = {'file': (filename, f)}
                http_response = requests.post(response['url'], data=response['fields'], files=files)
                print('http:', http_response.status_code)
            if http_response.status_code == 204:
                index_info = {"project_token": self.project_token, "object_name": object_name,
                              "network_id": self.network_id}
                r = requests.post(self.host + 'post_checkpoint_index', data=json.dumps(index_info), headers=self.auth)
                if r.status_code != 201:
                    print(r.text)
                    raise ValueError("Errors.")
        except:
            raise NetworkError("Could not upload checkpoint to s3")

    def send_config_file(self, filename, object_name):
        response = self._get_presigned_url('post', object_name, bucket_model=True)
        try:
            with open(filename, 'rb') as f:
                files = {'file': (filename, f)}
                http_response = requests.post(response['url'], data=response['fields'], files=files)
                print('http:', http_response.status_code)
            if http_response.status_code == 204:
                index_info = {"project_token": self.project_token, "object_name": object_name,
                              "network_id": self.network_id}
                r = requests.post(self.host + 'post_config', data=json.dumps(index_info), headers=self.auth)
                if r.status_code != 201:
                    print(r.text)
                    raise ValueError("Errors.")
        except:
            raise NetworkError("Could not upload config to s3")

    def get_dataset_list(self):
        r = requests.get(self.host + 'get_dataset_list', headers=self.auth)
        dataset_names = r.json()["dataset_names"]
        for e in dataset_names:
            print("\t", e)
        return dataset_names

    def create_dataset(self, dataset_name):
        if not isinstance(dataset_name, str):
            raise ValueError('dataset_name must be a string not {}'
                             .format(type(dataset_name)))
        dataset_info = {"dataset_name": dataset_name}
        r = requests.get(self.host + 'create_dataset', data=json.dumps(dataset_info), headers=self.auth)
        new_dataset = r.json()["new_dataset"]
        if not new_dataset:
            dataset_names = r.json()["dataset_names"]
            raise ValueError("You already have a dataset with this name, \
                please pick another one")
        else:
            dataset_id = r.json()["dataset_id"]
            print("Dataset {} created successfully".format(dataset_name))
            return dataset_id

    def send_dataset_thumbnail(self, dataset_name, img_path):
        if not isinstance(dataset_name, str):
            raise ValueError('dataset_name must be a string not {}'
                             .format(type(dataset_name)))
        data = {
            "dataset_name": dataset_name
        }
        with open(img_path, 'rb') as f:
            files = {'file': (img_path, f)}
            http_response = requests.post(self.host + 'send_dataset_thumbnail', data=data, files=files,
                                          headers=self.auth)
            if http_response.status_code == 200:
                print(http_response.text)
            else:
                raise NetworkError("Could not upload thumbnail")

    def create_and_upload_dataset(self, dataset_name, path_to_images):
        """ Create a dataset and upload the images to Picsell.ia

        Args :
            dataset_name (str)
            path_to_images (str)

        Raises:
            ValueError
            NetworkError: If impossible to upload to Picsell.ia server

        """
        if not isinstance(dataset_name, str):
            raise ValueError('dataset_name must be a string not {}'
                             .format(type(dataset_name)))

        if not isinstance(path_to_images, str):
            raise ValueError('path_to_images must be a string not {}'
                             .format(type(path_to_images)))

        if not os.path.isdir(path_to_images):
            raise FileNotFoundError('{} is not a directory'
                                    .format(path_to_images))

        dataset_id = self.create_dataset(dataset_name)
        print("Dataset created, starting file upload...")
        image_list = os.listdir(path_to_images)
        thumb_path = os.path.join(path_to_images, image_list[0])
        self.send_dataset_thumbnail(dataset_name, thumb_path)
        if len(image_list) > 0:
            object_name_list = []
            image_name_list = []
            sizes_list = []
            for img_path in image_list:
                file_path = os.path.join(path_to_images, img_path)
                if not os.path.isfile(file_path):
                    raise FileNotFoundError("Can't locate file @ %s" % (file_path))
                OBJECT_NAME = os.path.join(dataset_id, img_path)

                response = self._get_presigned_url(method='post', object_name=OBJECT_NAME)
                to_send = {"object_name": OBJECT_NAME}

                try:
                    im = Image.open(file_path)
                    width, height = im.size
                    with open(file_path, 'rb') as f:
                        files = {'file': (OBJECT_NAME, f)}
                        http_response = requests.post(response['url'], data=response['fields'], files=files)
                        print('http:', http_response.status_code)
                    if http_response.status_code == 204:
                        object_name_list.append(OBJECT_NAME)
                        image_name_list.append(img_path)
                        sizes_list.append([width, height])
                except:
                    raise NetworkError("Could not upload examples to s3")

            to_send2 = {"dataset_id": dataset_id,
                        "object_list": object_name_list, "image_list": image_list,
                        "sizes_list": sizes_list}
            try:
                r = requests.post(self.host + 'create_pictures_for_dataset', data=json.dumps(to_send2),
                                  headers=self.auth)
                if r.status_code != 200:
                    raise ValueError("Errors.")
                print("A snapshot of results has been saved to the platform")
            except:
                raise NetworkError("Could not upload to Picsell.ia Backend")

    def _send_chunk_custom(self, chunk_annotations):
        to_send = {
            'format': 'custom',
            'annotations': chunk_annotations,
            'project_token': self.project_token
        }

        try:
            r = requests.post(self.host + 'upload_annotations', data=json.dumps(to_send), headers=self.auth)
            if r.status_code == 400:
                raise NetworkError("Impossible to upload annotations to Picsell.ia backend because \n%s" % (r.text))
            print("300 annotations uploaded")
        except:
            raise NetworkError("Impossible to upload annotations to Picsell.ia backend")

    def _send_chunk_picsell(self, chunk_annotations):
        to_send = {
            'format': 'picsellia',
            'annotations': chunk_annotations,
            'project_token': self.project_token
        }

        try:
            r = requests.post(self.host + 'upload_annotations', data=json.dumps(to_send), headers=self.auth)
            if r.status_code == 400:
                raise NetworkError("Impossible to upload annotations to Picsell.ia backend because \n%s" % (r.text))
            print("300 annotations uploaded")
        except:
            raise NetworkError("Impossible to upload annotations to Picsell.ia backend")

    def upload_annotations(self, annotations, format='picsellia'):
        """ Upload annotation to Picsell.ia Backend

        Please find in our Documentation the annotations format accepted to upload

        Args :
            annotation (dict)
            format (str) : Chose between train & test

        Raises:
            ValueError
            NetworkError: If impossible to upload to Picsell.ia server

        """
        if not isinstance(format, str):
            raise ValueError('format must be a string not {}'
                             .format(type(format)))

        if format != 'picsellia':
            if not isinstance(annotations, dict):
                raise ValueError('dict of annotations in images must be a dict_annotations not {}'
                                 .format(type(annotations)))
            total_img = len(annotations["annotations"])
            #steps = total_img // 300
            print("Chunking your annotations ...")
            print("Upload starting ..")

            all_chunk = []

            for im in annotations["images"]:
                chunk_tmp = []
                for ann in annotations["annotations"]:
                    if ann["image_id"] == im["id"]:
                        chunk_tmp.append(ann)
                all_chunk.append({
                    "images": [im],
                    "annotations": chunk_tmp,
                    "categories": annotations["categories"]
                })

            pool = ThreadPool(processes=8)
            pool.map(self._send_chunk_custom, all_chunk)


    def _get_presigned_url(self, method, object_name, bucket_model=False):

        to_send = {"object_name": object_name, "bucket_model": bucket_model}

        if method == 'post':
            r = requests.get(self.host + 'get_post_url_preview', data=json.dumps(to_send), headers=self.auth)
        if method == 'get':
            r = requests.get(self.host + 'generate_get_presigned_url', data=json.dumps(to_send), headers=self.auth)
        if r.status_code != 200:
            raise ValueError("Errors.")

        return r.json()["url"]

    def _init_multipart(self):
        """Initialize the upload to saved Checkpoints or SavedModel

        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no saved_model saved

        """

        try:
            to_send = {"object_name": self.OBJECT_NAME}
            r = requests.get(self.host + 'init_upload', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 200:
                print(r.text)
                return False
            self.uploadId = r.json()["upload_id"]

        except:
            raise NetworkError('Impossible to initialize Upload')

    def _get_url_for_part(self, no_part):
        """Get a pre-signed url to upload a part of Checkpoints or SavedModel

        Raises:
            NetworkError: If it impossible to initialize upload

        """
        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self,
                                                                                              "OBJECT_NAME") or not hasattr(
                self, "uploadId"):
            raise ResourceNotFoundError("Please initialize upload with _init_multipart()")
        try:
            to_send = {"project_token": self.project_token, "object_name": self.OBJECT_NAME,
                       "upload_id": self.uploadId, "part_no": no_part}
            r = requests.get(self.host + 'get_post_url', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 200:
                raise NetworkError("Impossible to get an url.. because :\n%s" % (r.text))
            return r.json()["url"]
        except:
            raise NetworkError("Impossible to get an url..")

    def _upload_part(self, file_path):
        try:
            max_size = 5 * 1024 * 1024
            urls = []
            file_size = os.path.getsize(file_path)
            upload_by = int(file_size / max_size) + 1
            with open(file_path, 'rb') as f:
                for part in range(1, upload_by + 1):
                    signed_url = self._get_url_for_part(part)
                    urls.append(signed_url)
                parts = []
                for num, url in enumerate(urls):
                    part = num + 1
                    done = int(50 * num / len(urls))
                    try:
                        file_data = f.read(max_size)
                        res = requests.put(url, data=file_data)
                        if res.status_code != 200:
                            raise NetworkError("Impossible to put part no {}\n because {}".format(num + 1, res.text))
                        etag = res.headers['ETag']
                        parts.append({'ETag': etag, 'PartNumber': part})
                        sys.stdout.write("\r[%s%s]" % ('=' * (done - 1) + '>', ' ' * (50 - done)))
                        sys.stdout.flush()
                    except:
                        raise NetworkError("Impossible to put part no {}".format(num + 1))
                return parts
        except:
            raise NetworkError("Impossible to upload frozen graph to Picsell.ia backend")

    def _complete_part_upload(self, parts, object_name, file_type):

        """Complete the upload a part of Checkpoints or SavedModel

        Raises:
            NetworkError: If it impossible to initialize upload

        """
        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self, "OBJECT_NAME"):
            raise ResourceNotFoundError("Please initialize upload with _init_multipart()")
        try:
            to_send = {"project_token": self.project_token, "object_name": object_name, "file_type": file_type,
                       "upload_id": self.uploadId, "parts": parts, "network_id": self.network_id,
                       "training_id": self.training_id}

            r = requests.get(self.host + 'complete_upload', data=json.dumps(to_send), headers=self.auth)
            if r.status_code != 201:
                NetworkError("Impossible to get an url.. because :\n%s" % (r.text))
            return True
        except:
            raise NetworkError("Impossible to get an url..")

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
            print("Label_map.pbtxt created @ {}".format(self.label_path))

        except:
            raise ResourceNotFoundError("No directory found, please call checkout_network() or create_network() function first")

        self.label_map = labels_Network




    def send_labelmap(self, label_path=None):
        """Attach to network, it allow nicer results visualisation on hub playground
        """

        if label_path is not None:
            if not os.path.isfile(label_path):
                raise FileNotFoundError("label map @ %s doesn't exists" % label_path)
            with open(label_path, 'r') as f:
                label_map = json.load(f)
            # label = {}
            # for k,v in label_map.items():
            #     if len(k) < 3 and not all(map(str.isdigit, k)):
            #         label[v] = k
            #     else:
            #         label[]

        if not hasattr(self, "label_map") and label_path is None:
            raise ValueError("Please Generate label map first")

        if label_path is not None:
            to_send = {"project_token": self.project_token, "labels": label_map, "network_id": self.network_id}
        else:
            to_send = {"project_token": self.project_token, "labels": self.label_map, "network_id": self.network_id}

        try:
            r = requests.get(self.host + 'attach_labels', data=json.dumps(to_send), headers=self.auth)
        except:
            raise NetworkError("Could not connect to picsellia backend")
        if r.status_code != 201:
            print(r.text)
            raise ValueError("Could not upload label to server")

    def _zipdir(sef, path):
        zipf = zipfile.ZipFile(path.split('.')[0]+'.zip', 'w', zipfile.ZIP_DEFLATED)
        for filepath in os.listdir(path):
            zipf.write(os.path.join(path, filepath), filepath)

            if os.path.isdir(os.path.join(path,filepath)):
                for fffpath in os.listdir(os.path.join(path,filepath)):
                    zipf.write(os.path.join(path, filepath, fffpath), os.path.join(filepath,fffpath))

        zipf.close()
        return path.split('.')[0]+'.zip'


    def tf_vars_generator(self, label_map=None, ensemble='train', annotation_type="polygon"):
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

        if label_map==None and annotation_type != "classification":
            raise ValueError("Please provide a label_map dict loaded from a protobuf file when working with object detection")

        if annotation_type == "classification":
            label_map = {v:int(k) for k,v in self.label_map.items()}

        if ensemble == "train":
            path_list = self.train_list
            id_list = self.train_list_id
        else:
            path_list = self.eval_list
            id_list = self.eval_list_id

        if annotation_type == "rectangle":
            for ann in self.dict_annotations["annotations"]:
                for an in ann["annotations"]:
                    if "polygon" in an.keys():
                        annotation_type = "rectangle from polygon"
                        break

        print(annotation_type)

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

            if annotation_type == "rectangle from polygon":
                for image_annoted in self.dict_annotations["annotations"]:
                    if internal_picture_id == image_annoted["internal_picture_id"]:
                        for a in image_annoted["annotations"]:
                            # try:
                            if "polygon" in a.keys():
                                geo = a["polygon"]["geometry"]
                                poly = []
                                for coord in geo:
                                    poly.append([[coord["x"], coord["y"]]])

                                poly = np.array(poly, dtype=np.float32)

                                x, y, w, h = cv2.boundingRect(poly)
                                xmins.append(x / width)
                                xmaxs.append((x + w) / width)
                                ymins.append(y / height)
                                ymaxs.append((y + h) / height)
                                classes_text.append(a["label"].encode("utf8"))
                                label_id = label_map[a["label"]]
                                classes.append(label_id)
                            elif 'rectangle' in a.keys():
                                xmin = a["rectangle"]["left"]
                                ymin = a["rectangle"]["top"]
                                w = a["rectangle"]["width"]
                                h = a["rectangle"]["height"]
                                
                                xmax = xmin + w
                                ymax = ymin + h

                                # ymin = a["rectangle"]["top"]
                                # xmax = xmin + a["rectangle"]["height"]
                                # ymin = a["rectangle"]["left"]
                                # ymax = ymin + a["rectangle"]["width"]

                                ymins.append(ymin/height)
                                ymaxs.append(ymax/height)
                                xmins.append(xmin/width)
                                xmaxs.append(xmax/width)

                                classes_text.append(a["label"].encode("utf8"))
                                label_id = label_map[a["label"]]
                                classes.append(label_id)

                            # except:
                            #     pass
                yield (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                encoded_jpg, image_format, classes_text, classes)

            elif annotation_type=="rectangle":
                for image_annoted in self.dict_annotations["annotations"]:
                    if internal_picture_id == image_annoted["internal_picture_id"]:
                        for a in image_annoted["annotations"]:
                            try:
                                if 'rectangle' in a.keys():
                                    xmin = a["rectangle"]["left"]
                                    ymin = a["rectangle"]["top"]
                                    w = a["rectangle"]["width"]
                                    h = a["rectangle"]["height"]
                                    
                                    xmax = xmin + w
                                    ymax = ymin + h

                                    # ymin = a["rectangle"]["top"]
                                    # xmax = xmin + a["rectangle"]["height"]
                                    # ymin = a["rectangle"]["left"]
                                    # ymax = ymin + a["rectangle"]["width"]

                                    ymins.append(ymin/height)
                                    ymaxs.append(ymax/height)
                                    xmins.append(xmin/width)
                                    xmaxs.append(xmax/width)
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


if __name__ == '__main__':
    print("Trying to create a new model ")
    client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
    client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
    client.dl_annotations()
    client.dl_pictures()
    #client.checkout_network('test_up')
    #client.dl_annotations()
    #client.dl_pictures(prop=0.5)

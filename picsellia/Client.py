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
    def __init__(self, token=None, png_dir=None, host="http://127.0.0.1:8000/sdk/"):
        """ Creates and initializes a Picsell.ia Client.
        Args:
            token (str): TOKEN key, given on the platform.
            host (str): URL of the Picsell.ia server to connect to.
            png_dir (str): path/to/your/images If you are working locally and don't want to dl images from our server
        Raises:
            InvalidQueryError: If no 'token' provided as an argument.
            AuthenticationError: If `token` does not match the provided token on the platform.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
            ResourceNotFoundError: If the png_dir provided doesn't point to images
        """

        if token is None:
            raise InvalidQueryError("Token argument not provided")

        to_send = {"token": token}
        self.host = host





        print("Initializing Picsell.ia Client at {} ...".format(host))

        try:
            r = requests.get(self.host + 'check_connection', data=json.dumps(to_send))
        except:
            raise NetworkError(
                "Server is not responding, please check your host or Picsell.ia server status on twitter")
        if r.status_code == 400:
            raise AuthenticationError('The token provided does not match any of the known token for profile.')
        self.token = token
        self.project_id = r.json()["project_id"]

        print("Connection established at %s" % (host))


        if png_dir is None:
            self.png_dir = self.project_id + '/images/'
        else:
            self.png_dir = png_dir
            print("Looking for images @ %s ..." % self.png_dir)
            if not len(os.listdir(self.png_dir)) != 0:
                raise ResourceNotFoundError("Can't find images at %s" % (self.png_dir))

            for filename in os.listdir(self.png_dir):
                if filename.split('.')[-1] not in ["png", "jpg", "jpeg"]:
                    raise ResourceNotFoundError("Found a non supported filetype (%s) in your png_dir " % (filename.split('.')[-1]))
    def init_model(self, model_name):
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
            model_name (str): It's simply the name you want to give to your NeuralNet
                              For example, SSD_Picsellia

        Raises:
            AuthenticationError: If `token` does not match the provided token on the platform.
            NetworkError: If Picsell.ia server not responding or host is incorrect.
        """

        to_send = {"model_name": model_name, "token": self.token}

        try:
            r = requests.get(self.host + 'init_model', data=json.dumps(to_send))
            if r.status_code == 400:
                raise AuthenticationError('The token provided does not match any of the known token for profile.')

            print("Connection Established")

            self.network_id = r.json()["network_id"]
            self.training_id = r.json()["training_id"]

        except:
            raise NetworkError("Server is not responding, please check your host or Picsell.ia server status on twitter")


        if self.training_id == 0:
            print("It's your first training for this project")
        else:
            print("It's the training number {} for this project".format(self.training_id))


        self.dict_annotations = {}
        self.base_dir = "{}/{}/{}/".format(self.project_id, self.network_id, self.training_id)
        self.log_dir = self.base_dir + "logs/"
        self.checkpoint_dir = self.base_dir + 'checkpoint/'
        self.record_dir = self.base_dir + 'records/'
        self.config_dir = self.base_dir + 'config/'
        self.results_dir = self.base_dir + 'results/'
        self.exported_model = self.base_dir + 'exported_model/'

        if not os.path.isdir(self.project_id):
            print("First time using Picsell.ia on this project, initializing directories ...")
            os.mkdir(self.project_id)

        if not os.path.isdir(os.path.join(self.project_id, self.network_id)):
            os.mkdir(os.path.join(self.project_id, self.network_id))

        if not os.path.isdir(self.base_dir):
            print("Creating directory for project {}".format(self.base_dir))
            os.mkdir(self.base_dir)

        if not os.path.isdir(self.png_dir):
            print("Creating directory for PNG Images of project {}".format(self.base_dir))
            os.mkdir(self.png_dir)

        if not os.path.isdir(self.checkpoint_dir):
            print("Creating directory for checkpoints project {}".format(self.base_dir))
            os.mkdir(self.checkpoint_dir)

        if not os.path.isdir(self.log_dir):
            print("Creating directory for logs of project {}".format(self.log_dir))
            os.mkdir(self.log_dir)

        if not os.path.isdir(self.record_dir):
            print("Creating directory for records of project {}".format(self.base_dir))
            os.mkdir(self.record_dir)

        if not os.path.isdir(self.config_dir):
            print("Creating directory for config of project {}".format(self.base_dir))
            os.mkdir(self.config_dir)

        if not os.path.isdir(self.results_dir):
            print("Creating directory for results of project {}".format(self.results_dir))
            os.mkdir(self.results_dir)

    def dl_annotations(self, option="train"):
        """ Pull all the annotations made on Picsell.ia Platform for your project.

            Args:
                option (str): Define what time of annotation to export (accepted or all)

            Raises:
                AuthenticationError: If `token` does not match the provided token on the platform.
                NetworkError: If Picsell.ia server not responding or host is incorrect.
                ResourceNotFoundError: If we can't find any annotations for that project.
            """

        print("Downloading annotations of project {} ...".format(self.token))

        try:
            to_send = {"token": self.token, "type": option}
            r = requests.get(self.host + 'annotations', data=json.dumps(to_send))

            if r.status_code != 200:
                return ResourceNotFoundError("There is no annotations found for this project")

            print("Annotations pulled ...")
            self.dict_annotations = r.json()

        except:
            raise NetworkError("Server is not responding, please check your host or Picsell.ia server status on twitter")

    def _train_valid_split_obj_detection(self, prop=0.8):
        """Perform Optimized train test split for Object Detection.
           Uses optimization to find the optimal split to have the desired repartition of instances by set.

        Args:
            prop (float) : Percentage of Instances used for training.

        Raises:
            ResourceNotFoundError: If not annotations in the Picsell.ia Client yet.
        """
        if not hasattr(self, "dict_annotations") :
            raise ResourceNotFoundError("Please dl_annotation model with dl_annotation()")

        if not "categories" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")

        if not "images" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")


        final_mat = []
        cate = [v["name"] for v in self.dict_annotations["categories"]]
        print(cate)
        for img in self.dict_annotations['images']:
            cnt = [0] * len(cate)
            internal_picture_id = img["internal_picture_id"]
            for ann in self.dict_annotations["annotations"]:
                if internal_picture_id == ann["internal_picture_id"]:
                    for an in ann['annotations']:
                        idx = cate.index(an['label'])
                        cnt[int(idx)] += 1
            final_mat.append(cnt)

        L = np.array(final_mat).T
        train_total = np.array([sum(e) for e in L])
        nc, n = L.shape
        train_mins = prop * train_total
        train_mins = train_mins.astype('int')

        valid_mins = (1 - prop) * train_total
        valid_mins = valid_mins.astype('int')

        x = cvxpy.Variable(n, boolean=True)
        lr = cvxpy.Variable(nc, nonneg=True)
        ur = cvxpy.Variable(nc, nonneg=True)

        lb = (L @ x >= train_mins.T - lr)
        ub = (L @ x <= (sum(L.T) - valid_mins).T + ur)
        constraints = [lb, ub]

        objective = (sum(lr) + sum(ur)) ** 2
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        problem.solve()
        result = x.value
        self.index_url = [int(round(i)) for i in result]

    def _get_and_send_labels_repartition_obj_detection(self):
        """Perform train test split scanning for Object Detection.
           Uses optimization to find the optimal split to have the desired repartition of instances by set.

        Returns:
            cate (array[str]) : Array of the classes names
            cnt_train (array[int]) : Array of the number of object per class for the training set.
            cnt_eval (array[int]) : Array of the number of object per class for the evaluation set.

        Raises:
            ResourceNotFoundError: If not annotations in the Picsell.ia Client yet.
        """

        if not hasattr(self, "dict_annotations") :
            raise ResourceNotFoundError("Please dl_annotation model with dl_annotation()")

        if not "categories" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")

        if not "images" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")


        cate = [v["name"] for v in self.dict_annotations["categories"]]
        cnt_train = [0] * len(cate)
        cnt_eval = [0] * len(cate)

        for img, index in zip(self.dict_annotations['images'], self.index_url):

            internal_picture_id = img["internal_picture_id"]
            for ann in self.dict_annotations["annotations"]:
                if internal_picture_id == ann["internal_picture_id"]:
                    for an in ann['annotations']:
                        idx = cate.index(an['label'])
                        if index == 1:
                            cnt_train[int(idx)] += 1
                        else:
                            cnt_eval[int(idx)] += 1

        return cnt_train, cnt_eval, cate




    def local_pic_save(self, prop=0.8):
        """Download your training set on the machine (Use it to dl images to Google Colab etc.)
           Save it to /project_id/images/*

           Perform train_test_split_obj_detection & send the repartition to Picsell.ia Platform

        Args :
            prop (float) Percentage of Instances used for training.

        Raises:
            ResourceNotFoundError : If no annotations in the Picsell.ia Client yet or images can't be downloaded
            ProcessingError: If the train test split can't be performed.

        """

        if not hasattr(self, "dict_annotations") :
            raise ResourceNotFoundError("Please dl_annotation model with dl_annotation()")

        if not "images" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation function first")

        images_infos = self.dict_annotations["images"]
        self.train_list = []
        self.eval_list = []
        self.train_list_id = []
        self.eval_list_id = []
        cnt = 0

        print("Downloading PNG images to your machine ...")

        try:
            self._train_valid_split_obj_detection(prop)
        except:
            raise ProcessingError("Error during Train Test Split optimization, do you have classes with 0 instances ?")


        for info, idx in zip(images_infos, self.index_url):
            pic_name = os.path.join(self.png_dir, info['external_picture_url'].split('.')[0] + '.png')
            if idx == 1:
                self.train_list.append(pic_name)
                self.train_list_id.append(info["internal_picture_id"])
            else:
                self.eval_list.append(pic_name)
                self.eval_list_id.append(info["internal_picture_id"])

            if not os.path.isfile(pic_name):
                try:
                    img_data = requests.get(info["signed_url"]).content
                    with open(pic_name, 'wb') as handler:
                        handler.write(img_data)
                    cnt += 1
                except:
                    raise ResourceNotFoundError("Image %s can't be downloaded" % (pic_name))

        print("{} Images used for training, {} Images used for validation".format(len(self.train_list_id),
                                                                                  len(self.eval_list_id)))
        print("{} files were already on your machine".format(len(images_infos) - cnt))
        print(" {} PNG images have been downloaded to your machine".format(cnt))

        print("Sending repartition to Picsell.ia backend")

        label_train, label_test, cate = self._get_and_send_labels_repartition_obj_detection()

        to_send = {"token": self.token, "train": {"train_list_id": self.train_list_id, "label_repartition": label_train, "labels": cate},
                   "eval": {"eval_list_id": self.eval_list_id, "label_repartition": label_test, "labels": cate},
                   "network_id": self.network_id,"training_id": self.training_id}


        try:
            r = requests.post(self.host + 'post_repartition', data=json.dumps(to_send))
            if r.status_code != 201:
                raise NetworkError('Can not send repartition to Picsell.ia Backend')

            print("Repartition send ..")
        except:
            raise NetworkError('Can not send repartition to Picsell.ia Backend')

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
            raise ResourceNotFoundError("Please init model and dl_annotation()")

        self.label_path = '{}/label_map.pbtxt'.format(self.base_dir)


        if not "categories" in self.dict_annotations.keys():
            raise ResourceNotFoundError("Please run dl_annotation() first")

        categories = self.dict_annotations["categories"]

        try:
            with open(self.label_path, "w+") as labelmap_file:
                for k, category in enumerate(categories):
                    name = category["name"]
                    labelmap_file.write("item {\n\tname: \"" + name + "\"" + "\n\tid: " + str(k + 1) + "\n}\n")
                labelmap_file.close()
            print("Label_map.pbtxt crée @ {}".format(self.label_path))

        except:
            raise ResourceNotFoundError("No directory found, please call init_model() function first")

    def _init_multipart(self):
        """Initialize the upload to saved Checkpoints or SavedModel

        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no saved_model saved

        """
        date = time.strftime("%Y%m%d-%H%M%S")

        self.OBJECT_NAME = '{}saved_model/saved_model.pb'.format(self.exported_model)  # Get the actual timestamp

        if not os.path.isfile(self.OBJECT_NAME):
            raise ResourceNotFoundError('Saved Model does not exists')

        try:
            to_send = {"object_name": self.OBJECT_NAME}
            r = requests.get(self.host + 'init_upload', data=json.dumps(to_send))
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
        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self, "OBJECT_NAME") or not hasattr(self, "uploadId"):
            raise ResourceNotFoundError("Please initialize upload with _init_multipart()")
        try:
            to_send = {"token": self.token, "object_name": self.OBJECT_NAME,
                       "upload_id": self.uploadId, "part_no": no_part}
            r = requests.get(self.host + 'get_post_url', data=json.dumps(to_send))
            if r.status_code != 200:
                raise NetworkError("Impossible to get an url.. because :\n%s" % (r.text))
            return r.json()["url"]
        except:
            raise NetworkError("Impossible to get an url..")

    def _complete_part_upload(self, parts):

        """Complete the upload a part of Checkpoints or SavedModel

        Raises:
            NetworkError: If it impossible to initialize upload

        """
        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self, "OBJECT_NAME") or not hasattr(self, "parts"):
            raise ResourceNotFoundError("Please initialize upload with _init_multipart()")
        try:
            to_send = {"token": self.token, "object_name": self.OBJECT_NAME,
                       "upload_id": self.uploadId, "parts": parts, "network_id": self.network_id, "training_id": self.training_id}

            r = requests.get(self.host + 'complete_upload', data=json.dumps(to_send))
            if r.status_code != 201:
                NetworkError("Impossible to get an url.. because :\n%s" % (r.text))
            return True
        except:
            raise NetworkError("Impossible to get an url..")


    # def get_weights(self, version='latest'):
    #
    #     """
    #     Method used to dl the wanted weights to start training
    #     :arg version default = "latest"
    #                         = str::version printed in list_weights method
    #
    #     """
    #
    #     print("Downloading weights ...")
    #     to_send = {"token": self.token, "version": version}
    #     r = requests.get(self.host + 'get_checkpoint', data=json.dumps(to_send))
    #     date = time.strftime("%Y%m%d-%H%M%S")
    #
    #     if r.status_code != 200:
    #         print(r.text)
    #         return False
    #
    #     self.url_weights = r.json()["url"]
    #     self.weight_path = self.checkpoint_dir + date + '.h5'
    #     with requests.get(self.url_weights, stream=True) as r:
    #         r.raise_for_status()
    #         with open(self.weight_path, 'wb') as f:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 if chunk:  # filter out keep-alive new chunks
    #                     f.write(chunk)
    #                     # f.flush()
    #
    #     print("Weights pulled to your machine ...")

    def send_logs(self, logs):
        """Send training logs to Picsell.ia Platform

        Args:
            logs (dict): Dict of the training metric (Please find Getting Started Picsellia Docs to see how to get it)
        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no saved_model saved

        """

        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self, "host") or not hasattr(self, "token"):
            raise ResourceNotFoundError("Please initialize model with init_model()")

        try:
            to_send = {"token": self.token, "training_id": self.training_id, "logs": logs,  "network_id": self.network_id}
            r = requests.post(self.host + 'post_logs', data=json.dumps(to_send))
            if r.status_code != 201:
                raise NetworkError("The logs have not been send because %s" %(r.text))

            print(
                "Training logs have been send to Picsell.ia Platform...\nYou can now inspect and showcase results on the platform.")

        except:
            raise NetworkError("Could not connect to Picsell.ia Server")


    def send_examples(self,id=None):
        """Send Visual results to Picsell.ia Platform

        Args:
            id (str): Id of the training
        Raises:
            NetworkError: If it impossible to initialize upload
            FileNotFoundError:
            ResourceNotFoundError:

        """



        if id==None:
            try:
                results_dir = self.results_dir
                list_img = os.listdir(results_dir)
                assert len(list_img) != 0, 'No example have been created'
            except:
                raise ResourceNotFoundError("You didn't init_model(), please call this before sending examples")
        else:
            base_dir = '{}/{}/'.format(self.project_id,self.network_id)
            if str(id) in os.listdir(base_dir):
                results_dir = os.path.join(base_dir,str(id)+'/results')
                list_img = os.listdir(results_dir)
                assert len(list_img) != 0, 'No example have been created'
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                    os.path.join(base_dir,str(id)+'/results'))
        object_name_list = []
        for img_path in list_img:
            file_path = os.path.join(results_dir,img_path)
            if not os.path.isfile(file_path):
                raise FileNotFoundError("Can't locate file @ %s" % (file_path))
            OBJECT_NAME = file_path
            to_send = {"token": self.token, "object_name": OBJECT_NAME}
            try:
                r = requests.get(self.host + 'get_post_url_preview', data=json.dumps(to_send))
                if r.status_code != 200:
                    print(r.text)
                    raise ValueError("Errors.")
                response = r.json()["url"]
                print('response:',response)
            except:
                raise NetworkError("Could not get an url to post annotations")

            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (OBJECT_NAME, f)}
                    headers = {'content-type': 'image/png'}
                    http_response = requests.put(response, data=f,headers=headers)
                    print('http:',http_response.status_code)
                if http_response.status_code == 200:
                    object_name_list.append(OBJECT_NAME)
            except:
                raise NetworkError("Could not upload examples to s3")

        to_send2 = {"token": self.token,"network_id": self.network_id,
                    "training_id": self.training_id, "urls": object_name_list}
        try:
            r = requests.post(self.host + 'post_preview', data=json.dumps(to_send2))
            if r.status_code != 201:
                print(r.text)
                raise ValueError("Errors.")
            print("A snapshot of results has been saved to the platform")
        except:
            raise NetworkError("Could not upload to Picsell.ia Backend")

    def send_weights(self):

        """Send frozen graph for inference to Picsell.ia Platform


        Raises:
            NetworkError: If it impossible to initialize upload
            ResourceNotFoundError: If no visual results saved in /project_id/network_id/training_id/results/

        """

        if not hasattr(self, "training_id") or not hasattr(self, "network_id") or not hasattr(self, "host") or not hasattr(self, "token"):
            raise ResourceNotFoundError("Please initialize model with init_model()")
        max_size = 5 * 1024 * 1024
        urls = []
        self._init_multipart()
        file_size = os.path.getsize(self.OBJECT_NAME)
        upload_by = int(file_size / max_size) + 1

        try:
            with open(self.OBJECT_NAME, 'rb') as f:
                for part in range(1, upload_by + 1):
                    signed_url = self._get_url_for_part(part)
                    urls.append(signed_url)
                parts = []
                for num, url in enumerate(urls):
                    print('*'*num)
                    part = num + 1
                    try:
                        file_data = f.read(max_size)
                        res = requests.put(url, data=file_data)

                        if res.status_code != 200:
                            raise NetworkError("Impossible to put part no {}\n because {}".format(num+1, res.text))

                        etag = res.headers['ETag']
                        parts.append({'ETag': etag, 'PartNumber': part})
                    except:
                        raise NetworkError("Impossible to put part no {}".format(num+1))
        except:
            raise NetworkError("Impossible to upload frozen graph to Picsell.ia backend")

        if self._complete_part_upload(parts):
            print("Your exported model have been uploaded successfully to our cloud.")


    # def list_weights(self):
    #
    #     """
    #     Method to list all available weights, copy paste the version you want to start training from this checkpoints
    #
    #     """
    #     print("------------------------------------------")
    #     to_send = {"token": self.token}
    #     r = requests.get(self.host + 'get_checkpoints_list', data=json.dumps(to_send))
    #
    #     if r.status_code != 200:
    #         print(r.text)
    #         return False
    #
    #     resp = r.json()
    #     for k, v in resp.items():
    #         print("Checkpoint version @ {}\nCheckpoint stored @ {}\n".format(v["date"], v["key"]))
    #         print("------------------------------------------")

    def tf_vars_generator(self, label_map, ensemble='train', annotation_type="polygon"):
        """ /!\ THIS FUNCTION IS MAINTAINED FOR TENSORFLOW 1.X /!\

        Generator for variable needed to instantiate a tf example needed for training.

        Args :
            label_map (tf format)
            ensemble (str) : Chose between train & test
            annotation_type: "polygon" or "rectangle"

        Yields :
            (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                   encoded_jpg, image_format, classes_text, classes, masks)

        Raises:
            ResourceNotFoundError: If you don't have performed your trained test split yet
                                   If images can't be opened

        """

        if annotation_type not in ["polygon", "rectangle"]:
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
                for a in self.dict_annotations["annotations"]:
                    if internal_picture_id == a["internal_picture_id"]:
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

            elif annotation_type=="rectangle":
                for a in self.dict_annotations["annotations"]:
                    if internal_picture_id==a["internal_picture_id"]:
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

    def upload_annotations(self, annotations,format='picsellia'):
        """ Upload annotation to Picsell.ia Backend

        Please find in our Documentation the annotations format acceoted to upload

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
            if not isinstance(annotations, list):
                raise ValueError('list of annotations in images must be a list not {}'
                    .format(type(annotations)))

            annotation_list = []
            for i, image_dict in enumerate(annotations):
                if not isinstance(image, dict):
                    raise ValueError('annotations by image must be a dict not {}, in image n°{}'
                        .format(type(image),i))

                image = json.loads(image_dict)
                image_annotations = []
                if not isinstance(image["image_id"], str):
                    raise ValueError('image_id must be a string not {}'
                        .format(type(image["image_id"])))

                if not isinstance(image["image_name"], str):
                    raise ValueError('image_name must be a string not {}'
                        .format(type(image["image_name"])))

                for j,annotation_dict in enumerate(image["annotations"]):
                    if not isinstance(annotation, list):
                        raise ValueError('annotation must be a dict not {}, in annotation n°{}'
                            .format(type(annotation),j))

                    annotation = json.loads(annotation_dict)
                    if not isinstance(annotation["coordinates"], list):
                        raise ValueError('coordinates must be a list not {}, in annotation n°{}'
                            .format(type(annotation["coordinates"]),j))

                    if not isinstance(annotation["type"], str):
                        raise ValueError('type of annotation must be a list not {}, in annotation n°{}'
                            .format(type(annotation["type"]),j))

                    if not isinstance(annotation["label"], str):
                        raise ValueError('label of annotation must be a list not {}, in annotation n°{}'
                            .format(type(annotation["label"]),j))

                    annotation_json = {
                        'type': annotation["type"],
                        'label': annotation["label"],
                    }

                    if annotation["type"]=="polygon":
                        geometry = annotation["coordinates"]
                        annotation_json['polygon'] = {
                            'geometry': geometry
                        }

                    image_annotations.append(annotation_json)

                image_json = {
                    'image_id': image["image_id"],
                    'image_name': image["image_name"],
                    'nb_labels': len(image_annotations),
                    'annotations': image_annotations
                }

                try:
                    image_json["is_accepted"] = image["is_accepted"]
                except:
                    pass
                try:
                    image_json["is_reviewed"] = image["is_reviewed"]
                except:
                    pass
                annotation_list.append(image_json)
        else:
            if not isinstance(annotations, dict):
                raise ValueError('Picsellia annotations are a dict not {}'
                    .format(type(annotations)))
            annotation_list = annotations["annotations"]

        to_send = {
            "token": self.token,
            'format': format,
            'annotations': annotation_list
        }

        try:
            r = requests.post(self.host + 'upload_annotations', data=json.dumps(to_send))
            if r.status_code != 201:
                raise NetworkError("Impossible to upload annotations to Picsell.ia backend because \n%s" % (r.text))
            print("Your annotations has been uploaded, you can now see them in the platform")
        except:
            raise NetworkError("Impossible to upload annotations to Picsell.ia backend")

if __name__ == '__main__':
    client = Client(token="8054234f-408e-4aee-ad4f-5346ff572d46", host="https://backstage.picsellia.com/sdk/", png_dir='/home/batman/Documents/PicsellAL/PennFudanPed/PNGImages')
    client.init_model("first_model")
    client.dl_annotations()

import cv2
import io
import json
import os
import requests
import time
from PIL import Image, ImageDraw
import numpy as np
import cvxpy


class Client:

    def __init__(self, token, host="http://127.0.0.1:8000/sdk/"):
        print("Initializing Picsell.ia Client")
        to_send = {"token": token}
        self.host = host
        r = requests.get(self.host + 'check_connection', data=json.dumps(to_send))
        if r.status_code == 400:
            print(r.text)
            raise ValueError('Token is not ok.')
        print("Connection Established")
        self.token = token
        self.dict_annotations = {}
        self.base_dir = "{}/".format(self.token)
        self.png_dir = self.base_dir + "images/"

        self.checkpoint_dir = self.base_dir + 'checkpoint/'
        self.record_dir = self.base_dir + 'records/'
        self.config_dir = self.base_dir + 'config/'

        if not os.path.isdir(self.base_dir):
            print("Creating directory for project {}".format(self.base_dir))
            os.mkdir(self.base_dir)

        if not os.path.isdir(self.png_dir):
            print("Creating directory for PNG Images of project {}".format(self.base_dir))
            os.mkdir(self.png_dir)

        if not os.path.isdir(self.checkpoint_dir):
            print("Creating directory for checkpoints project {}".format(self.base_dir))
            os.mkdir(self.checkpoint_dir)

        if not os.path.isdir(self.record_dir):
            print("Creating directory for records of project {}".format(self.base_dir))
            os.mkdir(self.record_dir)

        if not os.path.isdir(self.config_dir):
            print("Creating directory for config of project {}".format(self.base_dir))
            os.mkdir(self.config_dir)

    def dl_annotations(self, option="train"):
        """
        Call this method to retrieve all the annotations accepted or not of your project

        input :
        :arg str::<option> train, accepted

        output :  - type : dict

        """
        print("Downloading annotations of project {} ...".format(self.token))
        to_send = {"token": self.token, "type": option}
        r = requests.get(self.host + 'annotations', data=json.dumps(to_send))

        if r.status_code != 200:
            print(r.text)
            return False

        print("Annotations pulled ...")
        self.dict_annotations = r.json()

    def _train_valid_split(self, prop):

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
        print(L)
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
        print(result)
        self.index_url = [int(round(i)) for i in result]

    def local_pic_save(self, prop=0.8):
        """
        Call this method to retrieve all the images annotated

        input :

        output :  - all png pictures uploaded to token/png_images/*.png

        """
        images_infos = self.dict_annotations["images"]
        self.train_list = []
        self.eval_list = []
        self.train_list_id = []
        self.eval_list_id = []
        print("Downloading PNG images to your machine ...")
        cnt = 0
        self._train_valid_split(prop)
        for info, idx in zip(images_infos, self.index_url):
            pic_name = os.path.join(self.png_dir, info['external_picture_url'].split('.')[0] + '.png')
            if idx == 1:
                self.train_list.append(pic_name)
                self.train_list_id.append(info["internal_picture_id"])
            else:
                self.eval_list.append(pic_name)
                self.eval_list_id.append(info["internal_picture_id"])

            if not os.path.isfile(pic_name):
                img_data = requests.get(info["signed_url"]).content
                with open(pic_name, 'wb') as handler:
                    handler.write(img_data)
                cnt += 1

        print("{} Images used for training, {} Images used for validation".format(len(self.train_list_id),
                                                                                  len(self.eval_list_id)))
        print("{} files were already on your machine".format(len(images_infos) - cnt))
        print(" {} PNG images have been downloaded to your machine".format(cnt))

        print("Sending repartition to Picsell.ia backend")

        to_send = {"token": self.token, "train": {"train_list_id": self.train_list_id},
                   "val": {"eval_list_id": self.eval_list_id}}
        r = requests.post(self.host + 'post_repartition', data=json.dumps(to_send))

        if r.status_code != 201:
            print(r.text)
            return False

        print("Repartition send ..")

    def generate_labelmap(self):

        """
                Call this method to generate the labelmap needed to initialize a training in tf

                output :  save in token/label_map.pbtxt

        """
        print("Generating labelmap ...")
        self.label_path = '{}/label_map.pbtxt'.format(self.token)
        categories = self.dict_annotations["categories"]

        with open(self.label_path, "w+") as labelmap_file:
            for k, category in enumerate(categories):
                name = category["name"]
                labelmap_file.write("item {\n\tname: \"" + name + "\"" + "\n\tid: " + str(k + 1) + "\n}\n")
            labelmap_file.close()
        print("Label_map.pbtxt crée @ {}".format(self.label_path))

    def _init_multipart(self):
        """
        hidden method used to communicate with Picsell.ia backend.

        """
        date = time.strftime("%Y%m%d-%H%M%S")
        self.OBJECT_NAME = '{}/checkpoint/{}.h5'.format(self.token, date)  # Get the actual timestamp
        to_send = {"object_name": self.OBJECT_NAME}
        r = requests.get(self.host + 'init_upload', data=json.dumps(to_send))
        if r.status_code != 200:
            print(r.text)
            return False
        self.uploadId = r.json()["upload_id"]

    def _get_url_for_part(self, no_part):
        """
        hidden method used to generate pre signed url of chunks for .h5 upload to Picssell.ia Backend

        """
        to_send = {"token": self.token, "object_name": self.OBJECT_NAME,
                   "upload_id": self.uploadId, "part_no": no_part}
        print(to_send)
        r = requests.get(self.host + 'get_post_url', data=json.dumps(to_send))
        if r.status_code != 200:
            print(r.text)
            return False
        return r.json()["url"]

    def _complete_part_upload(self, parts):

        """
        Hidden method used to complete the streamed upload of .h5
        """
        to_send = {"token": self.token, "object_name": self.OBJECT_NAME,
                   "upload_id": self.uploadId, "parts": parts}

        print(to_send)
        r = requests.get(self.host + 'complete_upload', data=json.dumps(to_send))
        if r.status_code != 201:
            print(r.text)
            return False
        return True

    def get_weights(self, version='latest'):

        """
        Method used to dl the wanted weights to start training
        :arg version default = "latest"
                            = str::version printed in list_weights method

        """

        print("Downloading weights ...")
        to_send = {"token": self.token, "version": version}
        r = requests.get(self.host + 'get_checkpoint', data=json.dumps(to_send))
        date = time.strftime("%Y%m%d-%H%M%S")

        if r.status_code != 200:
            print(r.text)
            return False

        self.url_weights = r.json()["url"]
        self.weight_path = self.checkpoint_dir + date + '.h5'
        with requests.get(self.url_weights, stream=True) as r:
            r.raise_for_status()
            with open(self.weight_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()

        print("Weights pulled to your machine ...")

    def send_weights(self, path_h5):

        """
        :arg path_h5 path to h5 file
        Method used to send h5 file to Picsell.ia backend
        """
        print("Initializing connection to our cloud")
        max_size = 5 * 1024 * 1024
        urls = []
        file_size = os.path.getsize(path_h5)
        upload_by = int(file_size / max_size) + 1
        with open(path_h5, 'rb') as f:
            for part in range(1, upload_by + 1):
                signed_url = self._get_url_for_part(part)
                urls.append(signed_url)
            parts = []
            for num, url in enumerate(urls):
                part = num + 1
                file_data = f.read(max_size)
                res = requests.put(url, data=file_data)
                if res.status_code != 200:
                    return
                etag = res.headers['ETag']
                parts.append({'ETag': etag, 'PartNumber': part})

        if self._complete_part_upload(parts):
            print("Your weights have been uploaded successfully to our cloud.")
        else:
            print("There has been an error during the upload of your h5 file,\nmaybe consider upolading it manually "
                  "on the platform.")

    def list_weights(self):

        """
        Method to list all available weights, copy paste the version you want to start training from this checkpoints

        """
        print("------------------------------------------")
        to_send = {"token": self.token}
        r = requests.get(self.host + 'get_checkpoints_list', data=json.dumps(to_send))

        if r.status_code != 200:
            print(r.text)
            return False

        resp = r.json()
        for k, v in resp.items():
            print("Checkpoint version @ {}\nCheckpoint stored @ {}\n".format(v["date"], v["key"]))
            print("------------------------------------------")

    def tf_vars_generator(self, label_map, ensemble='train'):
        """

        Generator for variable needed to instantiate a tf example needed for training.

        input : label_map
        :return: (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                   encoded_jpg, image_format, classes_text, classes, masks)
        """
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
            image_format = '{}'.filename.split('.')[-1]
            image_format = bytes(image_format.encode('utf8'))
            for a in self.dict_annotations["annotations"]:
                if internal_picture_id == a["internal_picture_id"]:
                    print("ok")
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

            yield (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                   encoded_jpg, image_format, classes_text, classes, masks)

import cv2
import io
import json
import os
import requests
import time
from PIL import Image, ImageDraw
from progressbar import ProgressBar

class Client:

    def __init__(self, token, host="http://127.0.0.1:8000/sdk/"):
        print("Initializing Picsell.ia Client")
        to_send = {"token": token}
        self.host = host
        r = requests.get(self.host + 'check_connection', data=json.dumps(to_send))
        if r.status_code == 400:
            print(r.text)
            return
        print("Connection Established")
        self.token = token
        self.dict_annotations = {}
        self.base_dir = "{}/".format(self.token)
        self.png_dir = self.base_dir + "PNG_images/"
        self.checkpoint_dir = self.base_dir + 'checkpoint/'
        if not os.path.isdir(self.base_dir):
            print("Creating directory for project {}".format(self.base_dir))
            os.mkdir(self.base_dir)
        if not os.path.isdir(self.png_dir):
            print("Creating directory for PNG Images or project {}".format(self.base_dir))
            os.mkdir(self.png_dir)
        if not os.path.isdir(self.checkpoint_dir):
            print("Creating directory for checkpoints project {}".format(self.base_dir))
            os.mkdir(self.checkpoint_dir)

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

    def local_png_save(self):
        """
        Call this method to retrieve all the images annotated

        input :

        output :  - all png pictures uploaded to token/png_images/*.png

        """
        pbar = ProgressBar()
        images_infos = self.dict_annotations["images"]

        print("Downloading PNG images to your machine ...")
        cnt = 0

        for info in pbar(images_infos):
            pic_name = os.path.join(self.png_dir, info['external_picture_url'].split('.')[0] + '.png')
            if not os.path.isfile(pic_name):
                img_data = requests.get(info["signed_url"]).content
                with open(pic_name, 'wb') as handler:
                    handler.write(img_data)
                cnt += 1

        print("{} files were already on your machine".format(len(images_infos) - cnt))
        print(" {} PNG images have been downloaded to your machine".format(cnt))

    def generate_labelmap(self):

        """
                Call this method to generate the labelmap needed to initialize a training in tf

                output :  save in token/label_map.pbtxt

        """
        print("Generatin labelmap ...")
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

    def tf_vars_generator(self, label_map):
        """

        Generator for variable needed to instantiate a tf example needed for training.

        input : label_map
        :return: (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                   encoded_jpg, image_format, classes_text, classes, masks)
        """
        for idx in range(len(self.dict_annotations['images'])):
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            masks = []

            external_picture_url = self.dict_annotations['images'][idx]["external_picture_url"]
            internal_picture_id = self.dict_annotations['images'][idx]["internal_picture_id"]

            with open(os.path.join(self.png_dir, external_picture_url), 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size
            filename = self.dict_annotations['images'][idx]["external_picture_url"].encode('utf8')
            image_format = '{}'.format(self.dict_annotations['images'][idx]["external_picture_url"].split('.')[-1])
            image_format = bytes(image_format.encode('utf8'))
            for a in self.dict_annotations["annotations"]:
                if internal_picture_id == a["internal_picture_id"]:
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

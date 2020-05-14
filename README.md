Get the Python Package, this is not based on any deep learning framework, but it's built to integrate smoothly in your deep learning process.
pip install picsellia
General Class
Client
The Picsell.ia Client contains info necessary for connecting to the Picsell.ia Platform. It provides top-level methods to :
format data for training
download annotations & images
send training logs
send examples
save weights and SavedModel to Picsell.ia server and push it to the mode Hub.
Objects Methods
def __init__(self, token=None, png_dir=None, host="https://app.picsellia.com/sdk/"):
Creates and initializes a Picsell.ia Client. 
      Args: 
                token (str): TOKEN key, given on the platform. 
                host (str): URL of the Picsell.ia server to connect to. 
                png_dir (str): path/to/your/images 
                If you are working locally and don't want to dl images from our server 
      Raises: 
               InvalidQueryError: If no token provided as an argument. 
               AuthenticationError: If token does not match the provided token on the platform. 
               NetworkError: If Picsell.ia server not responding or host is incorrect.
               ResourceNotFoundError: If the png_dir provided doesn't point to images 
def init_model(self, model_name, custom=False, init="base", selected_model="mask_rcnn/"):
Create or initialise the NeuralNet instance on Picsell.ia server
If the model name exists on the server for this project, we will create a new version of your training.
Create all the repositories for your training with this architecture :
    your_code.py
project_id/
images/
network_id/
training_version/
logs/
checkpoints/
records/
config/
results/
exported_model/
     Args:
               model_name (str): Name you want to give to your NeuralNet 
               custom (bool) : False if known architecture, True if not 
               init (str) : checkpoints or base
               selected_model (str) : if using tf 1.X object detection API, point to model directory downloadable @ tensorflow github
    Raises: 
              AuthenticationError: If token does not match the provided token on the platform.
              NetworkError: If Picsell.ia server not responding or host is incorrect.
def dl_annotations(self, option="train"):
Pull all the annotations made on Picsell.ia Platform for your project.
     Args: 
             option (str): Export option (accepted or all)
             option = "accepted" will download only the reviewed and accepted annotations.
     Raises:
              AuthenticationError
Your annotations will be available @ client.dict_annotations
def _train_valid_split_obj_detection(self, prop=0.8):
Perform optimized train test split for object detection/segmentation training. 
Find optimal images repartitions between training and validation set to have the desired proportion of objects by sets.
      Args:
              prop (float) : Desired proportion for training set
      Raises:
               ResourceNotFoundError If no annotations in the client object @ client.dict_annotations
def _get_and_send_labels_repartition_obj_detection(self):
Perform train test split for Object Detection and send the repartition to Picsell.ia Backend
       Returns:
              cate (array(str)) : Array of the class name
              cnt_train (array(int)) : Array of the number of objects per class for the training set
              cnt_eval same for eval set
       Raises:
              ResourceNotFoundError If no annotations in the client object @ client.dict_annotations
def local_pic_save(self, prop=0.8):
Download all annotated images present in client.dict_annotations to your machine and save it to      /project_id/images/  && Perfom a train test split to perform training .
        Args:
               prop (float) : Desired proportion for training set
        Raises:
               ResourceNotFoundError If no annotations in the client object @ client.dict_annotations 
               ProcessingError If the train test split can't be performed.
def generate_labelmap(self):
Generate the labelmap.pbtxt file needed for Tensorflow training 
@ project_id/network_id/training_id/label_map.pbtxt
       Raises:
              ResourceNotFoundError  If no annotation in the client 
                                                                If no directories have been created
def send_labelmap(self):
Send label/id correspondance to Picsell.ia Backend
        Raises:
               ValueError If no label map created
               NetworkError If impossible to connect to Picsell.ia Backend
def send_logs(self, logs):
Send training logs to Picsell.ia Backend
         Args:
             logs (dict) dict of logs extracted from tf.events (find how to in the how to section)
         Raises:
             NetworkError If connection to Picsell.ia Backend failed
             ResourceNotFoundError If no network_id in @ client.network_id 
def send_examples(self,id=None):
Send inference examples to Picsell.ia Backend
        Args:
            id (int) id of the training, if not provided, will send example for experiment client.training_id 
        Raises:
            NetworkError If impossible to connect to Picsell.ia Backend
            FileNotFoundEror If no inference examples have been generated 
            ResourceNotFoundError If inference have not been saved to client.results_dir 

def send_weights(self,file_path=None):
Send frozen graph of model for inference into Picsell.ia platform 
       Args:
             file_path (str) path/to/saved_model.pb, 
                   if None will look for file @ client.network_id/client.trainind_id/saved_model.pb 
       Raises:
            NetworkError If impossible to connect to Picsell.ia Backend
            FileNotFoundEror If no frozen inference graph have been generated 
            ResourceNotFoundError If no .pb file saved @   file_path 
def send_checkpoints(self,index_path=None,data_path=None):
Send the index file and the .ckpt to retrieve your training stage or allow user to perform transfer learning from your experiments.
        Args:
              index_path (str) path/to/index_file, 
                    If None, will look for file @ client.checkpoint_dir/model.ckpt-{}.index
              data_path (str) path/to/data_file, 
                    If None, will look for file @ client.checkpoint_dir/model.ckpt-{}.data
        Raises:
              FileNotFoundError No such file at client.checkpoint_dir/model.*.*
              ValueError If you provided a data_path but no index_path
def tf_vars_generator(self, label_map, ensemble='train', annotation_type="polygon"):
Generator for tf_records creation
       Args:
              label_map (dict) label_map dictionnary from generate_labelmap()
              ensemble (str) train or test 
              annotation_type (str) "polygon" , "rectangle", "classification"
       Yields:
              width (int) image width
              height (int) image height
              xmins (array) top-left box x coordinates
              xmaxs (array) top-right box x coordinates
              ymins (int) top-left box y coordinates
              ymaxs (int) top-right box y coordinates
              filename ('utf-8') utf-8 encoded filename
              encoded_jpg (bytes) bytes encoded image
              image_format (bytes) bytes encoded image extension
              classes_text ('utf-8') utf-8 encoded classes name
              classes (array) classes in images
   (if annotation_type=="polygon")
              masks encoded image masks for segmentation

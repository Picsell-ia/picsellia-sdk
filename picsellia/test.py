import unittest
from picsellia.Client import Client
from picsellia.exceptions import *
import os
import shutil
class TestInit(unittest.TestCase):
    def test_normal_use_init(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")

        self.assertEqual(client.auth, {"Authorization": "Bearer " + "57c60ade109be36ef1a1c89f56247109fa448741"})
        self.assertTrue(isinstance(client.project_name_list, list))
        self.assertEqual(client.username, "tibl23")

        self.assertEqual(client.project_token, client.project_id)
        self.assertTrue(isinstance(client.project_name, str))
        self.assertTrue(isinstance(client.project_type, str))
        self.assertTrue(isinstance(client.network_names, list))

    def test_wrong_token_init(self):
        with self.assertRaises(AuthenticationError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879bs0ec")

    def test_normal_init_png_dir(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec", png_dir="test_project/images")

    def test_wrong_init_png_dir(self):
        with self.assertRaises(ResourceNotFoundError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec", png_dir="test_project")

    def test_create_network_normal_use(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.create_network(network_name="test_creation_network_0")
        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.training_id,0)

        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))
        shutil.rmtree('test_project/test_creation_network_0')
        ### Test re create same network"
        with self.assertRaises(InvalidQueryError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
            client.create_network(network_name="test_creation_network_0")


    def test_checkout_network_vanilla_normal(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.checkout_network(network_name="ssd_inceptionV2_COCO")
        index_0 = client.checkpoint_index
        data_0 = client.checkpoint_data
        config_0 = client.config_file
        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.network_name, "ssd_inceptionV2_COCO")
        self.assertEqual(type(client.model_selected),str)
        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))
        with self.assertRaises(ResourceNotFoundError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
            client.checkout_network(network_name="ssd_inceptionV2_COCO_fake")

        # assert transfer object_name ok

        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="76f6c66a-71c0-415e-9715-12a7c971e899")
        client.checkout_network(network_name="ssd_inceptionV2_COCO_2")
        index_1 = client.checkpoint_index
        data_1 = client.checkpoint_data
        config_1 = client.config_file

        self.assertEqual(index_0, index_1)
        self.assertEqual(data_0, data_1)
        self.assertEqual(config_0, config_1)


        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.network_name, "ssd_inceptionV2_COCO_2")
        self.assertEqual(type(client.model_selected),str)
        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))
        with self.assertRaises(ResourceNotFoundError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
            client.checkout_network(network_name="ssd_inceptionV2_COCO_fake")

    def test_checkout_network_missing_file(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.checkout_network(network_name="ssd_inceptionV2_COCO")
    def test_checkout_network_with_training(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.checkout_network(network_name="ssd_inceptionV2_COCO")
        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.network_name, "ssd_inceptionV2_COCO")
        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))

    def test_reset_network(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.reset_network(network_name="ssd_inceptionV2_COCO")
        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.network_name, "ssd_inceptionV2_COCO")
        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))

    def test_create_network_normal_different_version(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.create_network(network_name="test_creation_network_0")
        self.assertTrue(isinstance(client.training_id,int))
        self.assertEqual(client.training_id,0)

        self.assertTrue(isinstance(client.network_names,list))
        self.assertTrue(isinstance(client.network_name,str))
        self.assertTrue(isinstance(client.dict_annotations,dict))
        self.assertEqual(client.base_dir, os.path.join(client.project_name, client.network_name, str(client.training_id)))
        self.assertEqual(client.metrics_dir, os.path.join(client.base_dir, 'metrics'))
        self.assertEqual(client.checkpoint_dir, os.path.join(client.base_dir, 'checkpoint'))
        self.assertEqual(client.record_dir, os.path.join(client.base_dir, 'records'))
        self.assertEqual(client.config_dir, os.path.join(client.base_dir, 'config'))
        self.assertEqual(client.results_dir, os.path.join(client.base_dir, 'results'))
        self.assertEqual(client.exported_model_dir, os.path.join(client.base_dir, 'exported_model'))

        self.assertTrue(os.path.isdir(client.base_dir))
        self.assertTrue(os.path.isdir(client.metrics_dir))
        self.assertTrue(os.path.isdir(client.checkpoint_dir))
        self.assertTrue(os.path.isdir(client.record_dir))
        self.assertTrue(os.path.isdir(client.config_dir))
        self.assertTrue(os.path.isdir(client.results_dir))
        self.assertTrue(os.path.isdir(client.exported_model_dir))


import mock

class TestDownload(unittest.TestCase):
    def test_dl_annotations(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.dl_annotations()
        self.assertTrue(len(client.dict_annotations.keys())!=0)

    def test_dl_latest_saved_model_path(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.checkout_network(network_name="ssd_inceptionV2_COCO")
        self.assertEqual(client.network_id,"ff3fb57d-6f95-46f6-a6dc-aeb0985dd6a4")
        client.dl_latest_saved_model(path_to_save="test_saved/ici")
        self.assertTrue(os.path.isdir("test_saved/ici"))
        self.assertTrue(os.path.isfile(os.path.join("test_saved/ici","saved_model.pb")))
        shutil.rmtree("test_saved/ici")

    def test_dl_latest_saved_model_no_path(self):
        with self.assertRaises(InvalidQueryError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
            client.checkout_network(network_name="ssd_inceptionV2_COCO")
            client.dl_latest_saved_model()

    def test_dl_checkpoints_wrong_input(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.checkout_network(network_name="ssd_inceptionV2_COCO")


class TestProcessor(unittest.TestCase):
    def test_train_test_split(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client.checkout_project(project_token="4b003477-3b31-4f74-8952-8a9dc879b0ec")
        client.create_network(network_name="test_creation_network_0")
        client.dl_annotations()
        client.train_test_split(prop=0.7)

        self.assertEqual(len(client.index_url), len(client.dict_annotations["images"]))
        self.assertEqual(len(client.train_list), len(client.train_list_id))
        self.assertEqual(len(client.eval_list), len(client.eval_list_id))

    def test_get_dataset_list(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        ds_list = client.get_dataset_list()
        self.assertTrue(isinstance(ds_list,list))

    def test_create_dataset(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        ds_id = client.create_dataset(dataset_name="test_dataset_0")
        self.assertTrue(isinstance(ds_id,str))

        with self.assertRaises(ValueError):
            client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
            ds_id = client.create_dataset(dataset_name="test_dataset_0")

    def test_upload_and_create_dataset(self):
        client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
        client= client.create_and_upload_dataset(dataset_name="test_dataset_1", path_to_images="test_images/")



if __name__ == '__main__':
    #unittest.main()
    client = Client(api_token="57c60ade109be36ef1a1c89f56247109fa448741")
    client.checkout_project(project_token = "d8e65668-9e18-421e-966c-7daa8d7c7497")
    model_name = "ssd_base"
    client.checkout_network(model_name)
    client.dl_annotations()
    client.dl_pictures()
    client.train_test_split()
    client.generate_labelmap()
    a = client.tf_vars_generator(client.label_map, annotation_type="rectangle")
    x = next(a)
    print(x[:4])
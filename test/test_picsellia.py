from picsellia.Client import Client
import unittest


class TestClient(unittest.TestCase):
    def test_init_local(self):
        client = Client(token="8054234f-408e-4aee-ad4f-5346ff572d46",
                        host="https://backstage.picsellia.com/sdk/",
                        png_dir='/home/batman/Documents/PicsellAL/PennFudanPed/PNGImages')
        self.assertEqual(client.token, "8054234f-408e-4aee-ad4f-5346ff572d46")
        self.assertTrue(isinstance(client.project_id, str))
        self.assertTrue(isinstance(client.png_dir, str))

    def test_init_cloud(self):
        client = Client(token="8054234f-408e-4aee-ad4f-5346ff572d46",
                             host="https://backstage.picsellia.com/sdk/",
                             png_dir=None)
        self.assertEqual(client.token, "8054234f-408e-4aee-ad4f-5346ff572d46")
        self.assertTrue(isinstance(client.project_id, str))
        self.assertTrue(isinstance(client.png_dir, str))

    def test_init_model(self):
        client = Client(token="8054234f-408e-4aee-ad4f-5346ff572d46",
                        host="https://backstage.picsellia.com/sdk/",
                        png_dir=None)

        client.init_model('test')
        self.assertEqual(client.token, "8054234f-408e-4aee-ad4f-5346ff572d46")
        self.assertTrue(isinstance(client.project_id, str))
        self.assertTrue(isinstance(client.png_dir, str))
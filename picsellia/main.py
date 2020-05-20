from Client import *


print("Trying to create a new model ")
client = Client(api_token="bdcfce35e0ff716d76a1bd549cc0804c19cb0f23", host="http://localhost:8000/sdk/")
client.checkout_project(project_token="bf6e4395-01ba-40a3-81a6-3e5666490128")
client.upload_and_create()
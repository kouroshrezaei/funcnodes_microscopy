import funcnodes as fn
import unittest
import numpy as np
from funcnodes_microscopy.SEM import upload_sem_image
import os
from funcnodes_files import FileUpload
import base64


class TestSEM(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "1908248.tif"), "rb") as f:
            self.tiffbytes = f.read()

    async def test_upload_sem_image(self):
        base64tiff = base64.b64encode(self.tiffbytes).decode("utf-8")

        data = FileUpload(filename="1908248.tif", content=base64tiff, path="")

        load_sem: fn.Node = upload_sem_image()
        load_sem.inputs["input"].value = data
        self.assertIsInstance(load_sem, fn.Node)
        await load_sem
        image = load_sem.outputs["image"].value
        metadata = load_sem.outputs["metadata"].value
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["Pixel Size (nm)"], 7.344)

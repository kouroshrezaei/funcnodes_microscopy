import funcnodes as fn
import unittest
from funcnodes_microscopy.SEM import upload_sem_image
from funcnodes_microscopy.segmentation import (
    classical_segmentation,
    # ThresholdTypes,
    # RetrievalModes,
    # ContourApproximationModes,
)
import os
from funcnodes_files import FileUpload
import base64

fn.config.IN_NODE_TEST = True


class TestSegmentation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "1908248.tif"), "rb") as f:
            self.tiffbytes = f.read()

    async def test_classical_segmentation(self):
        base64tiff = base64.b64encode(self.tiffbytes).decode("utf-8")
        data = FileUpload(filename="1908248.tif", content=base64tiff, path="")
        load_sem: fn.Node = upload_sem_image()
        load_sem.inputs["input"].value = data
        self.assertIsInstance(load_sem, fn.Node)
        seg: fn.Node = classical_segmentation()
        seg.inputs["image"].connect(load_sem.outputs["image"])
        seg.inputs["threshold"].value = 0.2
        seg.inputs["iter"].value = 3
        seg.inputs["pixel_size"].value = 7.344
        seg.inputs["min_diameter"].value = 10

        # print()
        self.assertIsInstance(seg, fn.Node)
        await fn.run_until_complete(seg, load_sem)
        contour_dict = seg.outputs["out"].value
        self.assertIsInstance(contour_dict, dict)

import funcnodes as fn
from typing import Dict, Any
import cv2
import numpy as np
from skimage import filters, exposure
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def eccentricity_from_ellipse(contour):
    """Calculates the eccentricity fitting an ellipse from a contour"""

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a**2 - b**2) / a
    return ecc


def minimum_enclosed_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return x, y, radius


def maximum_inclosed_circle(contour):
    M = cv2.moments(contour)
    Y = M["m10"] / M["m00"]
    X = M["m01"] / M["m00"]
    c = np.array((Y, X), dtype=float)
    dist = []
    for j in range(len(contour)):
        b = contour[j][0]
        dist.append(np.linalg.norm(c - b))
    return Y, X, np.min(dist)


class ThresholdTypes(fn.DataEnum):
    """
    Threshold types.

    Attributes:
        BINARY: cv2.THRESH_BINARY: 0 or maxval (if x > thresh)
        BINARY_INV: cv2.THRESH_BINARY_INV: maxval or 0 (if x > thresh)
        TRUNC: cv2.THRESH_TRUNC: thresh or x (if x > thresh)
        TOZERO: cv2.THRESH_TOZERO: x or 0 (if x > thresh)
        TOZERO_INV: cv2.THRESH_TOZERO_INV 0 or x (if x > thresh)
    """

    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
    TOZERO_INV = cv2.THRESH_TOZERO_INV


class RetrievalModes(fn.DataEnum):
    """
    Mode of the contour retrieval algorithm.

    Attributes:
        EXTERNAL: cv2.RETR_EXTERNAL: retrieves only the extreme outer contours.
        It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
        LIST: cv2.RETR_LIST: retrieves all of the contours without establishing any hierarchical relationships.
        CCOMP: cv2.RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
        TREE: cv2.RETR_TREE: retrieves all of the contours and reconstructs a full hierarchy of nested contours.
        FLOODFILL: cv2.RETR_FLOODFILL
    """

    EXTERNAL = cv2.RETR_EXTERNAL
    LIST = cv2.RETR_LIST
    CCOMP = cv2.RETR_CCOMP
    TREE = cv2.RETR_TREE
    FLOODFILL = cv2.RETR_FLOODFILL


class ContourApproximationModes(fn.DataEnum):
    """
    Approximation modes for the contour retrieval algorithm.

    Attributes:
        NONE: cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
        SIMPLE: cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
        TC89_L1: cv2.CHAIN_APPROX_TC89_L1: applies one of the flavors of the Teh-Chin chain approximation algorithm
        TC89_KCOS: cv2.CHAIN_APPROX_TC89_KCOS: applies one of the flavors of the Teh-Chin chain approximation algorithm
    """

    NONE = cv2.CHAIN_APPROX_NONE
    SIMPLE = cv2.CHAIN_APPROX_SIMPLE
    TC89_L1 = cv2.CHAIN_APPROX_TC89_L1
    TC89_KCOS = cv2.CHAIN_APPROX_TC89_KCOS


@fn.NodeDecorator(
    node_id="fn.microscopy.segmentation.classical",
    name="Classical Particle Segmentation",
    # outputs=[
    #     {"name": "out", "type": OpenCVImageFormat},
    # ],
    default_io_options={
        "iter": {"value_options": {"min": 0, "max": 10}},
        "thresh": {"value_options": {"min": 0, "max": 1}},
        "max_eccentricity": {"value_options": {"min": 0, "max": 1}},
    },
    # default_render_options={"data": {"src": "out"}},
)
def classical_segmentation(
    image: np.ndarray,
    threshold: float,
    iter: int,
    pixel_size: float,
    min_diameter: float,
    max_eccentricity: float = 0.7,
    thresh_type: ThresholdTypes = ThresholdTypes.BINARY,
    contour_retr_type: RetrievalModes = RetrievalModes.EXTERNAL,
    contour_approx_type: ContourApproximationModes = ContourApproximationModes.SIMPLE,
) -> Dict[str, Any]:
    # Check if num_iterations is a positive integer
    if not isinstance(iter, int) or iter <= 0:
        raise ValueError("Number of iterations must be a positive integer")
    thresh_type = ThresholdTypes.v(thresh_type)
    contour_retr_type = RetrievalModes.v(contour_retr_type)
    contour_approx_type = ContourApproximationModes.v(contour_approx_type)
    final_countors = []
    final_countors_centers_X = []
    final_countors_centers_Y = []

    min_en_diameters = []
    max_in_diameters = []
    mean_diameters = []

    min_en_circle = []
    max_in_circle = []

    threshold_global_otsu = filters.threshold_otsu(image)
    global_otsu = image >= threshold_global_otsu
    # end of new thresh
    back_removed = np.zeros([image.shape[0], image.shape[1]])
    for i in range(global_otsu.shape[0]):
        for j in range(global_otsu.shape[1]):
            if global_otsu[i, j]:
                # print(global_otsu[i,j], image[i,j])
                back_removed[i, j] = image[i, j]

    back_removed_threshold_global_otsu = filters.threshold_mean(
        np.ma.masked_equal(back_removed, 0).astype("uint8")
    )

    peaks_removed = np.zeros([image.shape[0], image.shape[1]])
    peaks = np.zeros([image.shape[0], image.shape[1]])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (
                back_removed[i, j] > 0
                and back_removed[i, j] <= back_removed_threshold_global_otsu
            ):
                peaks_removed[i, j] = back_removed[i, j]
            else:
                peaks[i, j] = back_removed[i, j]

    img_norm = (image - np.amin(image)) / (np.amax(image) - np.amin(image))

    # img_hist = local_histogram_equalization(img_norm, radius=5)
    img_hist = exposure.equalize_adapthist(img_norm, clip_limit=0.1)

    initial_thresh = [threshold] * iter
    for iter, init_thresh in enumerate(initial_thresh):
        ret, thresh = cv2.threshold(img_hist, init_thresh, 255, thresh_type)
        thresh = thresh.astype(np.uint8)

        # Compute Euclidean distance from every binary pixel
        # to the nearest zero pixel then find peaks
        distance_map = ndimage.distance_transform_edt(thresh)
        local_max = peak_local_max(distance_map, min_distance=3, labels=thresh)

        # Perform connected component analysis then apply Watershed NEW version

        current_mask = np.zeros(distance_map.shape, dtype=bool)
        current_mask[tuple(local_max.T)] = True
        markers, _ = ndimage.label(current_mask)

        labels = watershed(-distance_map, markers, mask=thresh)

        # Iterate through unique labels
        for label in np.unique(labels):
            if label == 0:
                continue

            # Create a mask
            mask = np.zeros(img_hist.shape, dtype="uint8")
            mask[labels == label] = 255

            # BACKGROUND REMOVAL TODO!

            # END OF BACKGROUND REMOVAL TODO!

            # Find contours and determine contour area
            cnts = cv2.findContours(mask.copy(), contour_retr_type, contour_approx_type)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for i in range(len(cnts)):
                if len(cnts[i]) >= 5:
                    if (
                        eccentricity_from_ellipse(cnts[i]) < max_eccentricity
                        and 2 * minimum_enclosed_circle(cnts[i])[-1] * pixel_size
                        > min_diameter
                    ):
                        M = cv2.moments(cnts[i])
                        Y = int(M["m10"] / M["m00"])
                        X = int(M["m01"] / M["m00"])
                        # print(X,Y)
                        if global_otsu[X, Y]:
                            final_countors_centers_X.append(X)
                            final_countors_centers_Y.append(Y)
                            min_en_circle.append(minimum_enclosed_circle(cnts[i]))
                            max_in_circle.append(maximum_inclosed_circle(cnts[i]))
                            min_en_diameters.append(
                                2 * minimum_enclosed_circle(cnts[i])[-1] * pixel_size
                            )
                            max_in_diameters.append(
                                2 * maximum_inclosed_circle(cnts[i])[-1] * pixel_size
                            )
                            mean_diameters.append(
                                (
                                    2
                                    * minimum_enclosed_circle(cnts[i])[-1]
                                    * pixel_size
                                    + 2
                                    * maximum_inclosed_circle(cnts[i])[-1]
                                    * pixel_size
                                )
                                / 2
                            )
                            final_countors.append(cnts[i])

        return {
            "num_contours": len(final_countors),
            "contours": final_countors,
            "contours_centers_X": final_countors_centers_X,
            "contours_centers_Y": final_countors_centers_Y,
            "min_enclosed_circle": min_en_circle,
            "max_inclosed_circle": max_in_circle,
            "min_enclosed_diameters": min_en_diameters,
            "max_inclosed_diameters": max_in_diameters,
            "mean_diameters": mean_diameters,
        }


SEGMENTATION_NODE_SHELF = fn.Shelf(
    nodes=[classical_segmentation],
    subshelves=[],
    name="Particle segmentation",
    description="Here you can segment an image with different algorithms",
)

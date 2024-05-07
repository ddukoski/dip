import cv2
import numpy as np


class CompassEdgeDetector:
    """
    This class, upon initialization makes out edges (both horizontal and vertical),
    with a DEFAULT threshold of 65 (class attribute below) using compass operators,
    this can be modified to needs, but 65 works well for the image provided
    """

    THRESHOLD = ...

    def __init__(self, path, threshold=65, show_intermediate=False):
        self.image = cv2.imread(path, 0)
        self.THRESHOLD = threshold
        self.show_intermediate = show_intermediate
        self.edges = self.__compass_edges()

    def __compass_edges(self):
        """
        :return: NumPy array[uint8] -> the edges as gotten by applying the compass operator

        We are going to use 2 operators, due to clarity
        """

        if self.show_intermediate:
            cv2.imshow("Original", self.image)
            cv2.waitKey(0)

        gauss_img = cv2.GaussianBlur(self.image, (5, 5), 0)

        if self.show_intermediate:
            cv2.imshow("Gaussian Blur", gauss_img)
            cv2.waitKey(0)

        # define gradients, important to apply sobel on blurred image
        g_x = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=3)
        g_y = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=3)

        # vertical emphasis on edges (south-west)
        compass_ops_v = [np.array([[-1, -1, 2],
                                   [-1, 2, -1],
                                   [2, -1, -1]])]

        # horizontal emphasis on edges (north)
        compass_ops_h = [np.array([[-1, -1, -1],
                                   [2, 2, 2],
                                   [-1, -1, -1]])]

        edge_init = list()

        for kernel in compass_ops_v:
            insert = cv2.filter2D(g_x, cv2.CV_64F, kernel) + cv2.filter2D(g_y, cv2.CV_64F, kernel)
            edge_init.append(insert)

        for kernel in compass_ops_h:
            insert = cv2.filter2D(g_x, cv2.CV_64F, kernel) + cv2.filter2D(g_y, cv2.CV_64F, kernel)
            edge_init.append(insert)

        # keep local maximums
        keep_local_max = np.max(edge_init, axis=0)

        print(edge_init)

        if self.show_intermediate:
            cv2.imshow('Convolution and Compass', np.uint8(edge_init[0]))
            cv2.waitKey(0)

        # using 60 as a threshold for retaining edges
        edges = keep_local_max > self.THRESHOLD

        # since the edges array is just booleans, we clip them to fit grayscale maximum of 255 per pixel
        return np.uint8(edges * 255)


if __name__ == '__main__':

    image_path = '../pics/lake-edge-treeline-at-sunset.jpg'
    transformation_obj_TH60 = CompassEdgeDetector(image_path, 60, True)
    transformation_obj_TH70 = CompassEdgeDetector(image_path, 70)
    transformation_obj_TH80 = CompassEdgeDetector(image_path, 80)
    transformation_obj_TH90 = CompassEdgeDetector(image_path, 90)

    cv2.imshow("Final edges - Threshold of 60", transformation_obj_TH60.edges)
    cv2.imshow("Final edges - Threshold of 70", transformation_obj_TH70.edges)
    cv2.imshow("Final edges - Threshold of 80", transformation_obj_TH80.edges)
    cv2.imshow("Final edges - Threshold of 90", transformation_obj_TH90.edges)

    cv2.waitKey(0)

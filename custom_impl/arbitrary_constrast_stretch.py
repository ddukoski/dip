import cv2
import numpy as np


class Stretcher:

    def __init__(self, points, image):
        self.points = points
        self.image = image

        # we are making an identical 3d array (as the image) and filling it with zeroes, so we can access the pixels
        self.stretched_image = np.zeros_like(image)
        self.__manual_vectorize()

    def __manual_vectorize(self):
        """
        Performs manual vectorization directly on the pixels (in order to access the pixel as a tuple of channels)
        :return: void
        """

        img_h = self.image.shape[0]
        img_w = self.image.shape[1]

        print(img_h)
        print(img_w)

        for i in range(img_h):
            for j in range(img_w):
                self.stretched_image[i, j] = self.__stretch(self.image[i, j])

    def __stretch_channel(self, channel):
        """
        :param channel: NumPy uint8 -> The channel to be modified by the pixel (R - pixel[0], G - pixel[1] or B - pixel[2])
        :return channel_modify: NumPy uint8 -> The modified channel
        """
        channel_modify = None

        for i in range(self.points.size - 4, 2):

            x1 = self.points[i]
            y1 = self.points[i+1]
            x2 = self.points[i+2]
            y2 = self.points[i+3]

            if x1 < channel <= x2:
                channel_modify = (y2 - y1) / (x2 - x1) * (channel - x1) + y1
                break

        if channel_modify is None:  # perform truncation for last 4
            x1 = self.points[-4]
            y1 = self.points[-3]
            x2 = self.points[-2]
            y2 = self.points[-1]
            channel_modify = (y2 - y1) / (x2 - x1) * (channel - x1) + y1

        return channel_modify

    def __stretch(self, pixel):
        """
        :param pixel: NumPy array[int] -> a 3-tuple (logically) representing the intensity of the RGB channels accordingly
        :return pixel_modify: NumPy array[int] -> the modified pixel
        """

        pixel_modify = np.array([None, None, None])

        for channel in range(3):
            pixel_modify[channel] = self.__stretch_channel(pixel[channel])

        return pixel_modify


if __name__ == '__main__':

    """
    input list format: every 2 indices are represent (x, y) points
    """

    # concrete example of an arbitrary number of points (note that first two elements are made to avoid checking for
    # (0, 0) in the class Stretcher, and these are not good points, just arbitrary ones)
    arb_points = np.array([0, 0, 10, 15, 60, 70, 140, 160, 190, 220, 240, 255])

    img = cv2.imread("../pics/ohrid.jpg", 1)
    print(img)

    # Stretcher constructor acting as a function that automatically makes the stretched image (stretched_image is the
    # output)
    stretcher = Stretcher(arb_points, img)

    cv2.imshow('Contrast Stretched', stretcher.stretched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


from matplotlib import pyplot as plt
from skimage.transform import resize
import cv2


def resize(image):
    carved = resize(image, (100, 100))
    return carved



def main():
    img = cv2.imread('seamcarving.jpeg')
    plt.figure()
    plt.title('experimenting with seam carving')
    plt.imshow(resize(img))



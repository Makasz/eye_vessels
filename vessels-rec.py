import cv2
import numpy as np

def msqer(img, img2):
    err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
    err /= float(img.shape[0] * img.shape[1])
    return err


def prepare(image):
    # cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('test', 600, 600)

    (b, g, r) = cv2.split(image)
    histogram = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_g = histogram.apply(g)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # fin = cv2.GaussianBlur(contrast_g, (11,11), 0)
    fin = cv2.bilateralFilter(contrast_g, 9, 75, 75)
    fin = cv2.adaptiveThreshold(fin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 2)
    n_fin = cv2.erode(fin, kernel3, iterations=1)
    fundus_eroded = cv2.bitwise_not(n_fin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    n_fin = cv2.morphologyEx(n_fin, cv2.MORPH_OPEN,
                          kernel3, iterations=1)
    highThresh, thresh_img = cv2.threshold(n_fin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(highThresh)
    lowThresh = 0.3 * highThresh
    edges = cv2.Canny(n_fin, 10, 20)

    x1, xcontours, xhierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(xcontours):
        if 3000 >= cv2.contourArea(cnt) >= 100:
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    (im2, contours, hierarchy) = cv2.findContours(blood_vessels.copy(),
                                                  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = len(blood_vessels), len(blood_vessels[0])
    max, min = 0, 100000
    max_i, min_i = 0,0
    for i, cnt in enumerate(contours):
        x, _, _, _ = cv2.boundingRect(cnt)
        if x > max and cv2.contourArea(cnt) > 22000:
            max_i = i
            max = x
        if x < min and cv2.contourArea(cnt) > 8000:
            min_i = i
            min = x
        if cv2.contourArea(cnt) <= 800:
            cv2.drawContours(blood_vessels, contours, i, 0, cv2.FILLED)
    cv2.drawContours(blood_vessels, contours, min_i, 0, cv2.FILLED)
    cv2.drawContours(blood_vessels, contours, max_i, 0, cv2.FILLED)
    # cv2.imshow('test', n_fin)
    return blood_vessels


def main():
    eye_image = cv2.imread('original.jpg')
    expected_result = cv2.cvtColor(cv2.imread('expected_result.tif'), cv2.COLOR_BGR2GRAY)

    bloodvessel = prepare(eye_image)

    # cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('expected', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('original', 600, 600)
    cv2.resizeWindow('result', 1200, 1200)
    cv2.resizeWindow('expected', 600, 600)

    # cv2.imshow('original', eye_image)
    # cv2.imshow('result', bloodvessel)
    # cv2.imshow('expected', expected_result)
    print(msqer(bloodvessel, expected_result))
    cv2.imwrite('result.png', bloodvessel)
    cv2.waitKey(0)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()

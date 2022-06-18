import pytesseract
import cv2
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# exit(0)
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

img_1 = cv2.imread('298/298-1.jpg')
img_2 = cv2.imread('298/298-2.jpg')
img_3 = cv2.imread('298/298-3.jpg')
ogs = [img_1,img_2,img_3]
img1 = canny(get_grayscale(cv2.imread('298/298-1.jpg')))
img2 = canny(get_grayscale(cv2.imread('298/298-2.jpg')))
img3 = canny(get_grayscale(cv2.imread('298/298-3.jpg')))

custom_config = r'--oem 3 --psm 6 -l nld'
custom_config2 = r'--oem 3 --psm 4 -l nld'
c = 0
for imag in [img1,img2,img3]:
    c+=1

    d = pytesseract.image_to_data(imag, config=custom_config, output_type=Output.DICT)
    ymax,xmax = imag.shape
    imag = cv2.rectangle(imag, (0, 0), (imag.shape), (0, 0, 0), -1)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(float(d['conf'][i]) + 0.5) > 30:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            imag = cv2.rectangle(imag, (x, y), (x + w, y + h), (255, 255, 0), -1)

    # tt = cv2.resize(imag, (int(ymax/2), int(xmax/2))) 
    imag = cv2.rectangle(imag, (0,0), (xmax, int(ymax/3)), (0,0,0), -1)
    imag = cv2.rectangle(imag, (0, int(ymax* 2/3)), (xmax, int(ymax)), (0,0,0), -1)
    imag = cv2.rectangle(imag, (int(xmax*4/5),int(ymax/3)), (int(xmax), int(2 * ymax/3)), (0,0,0), -1)

    print("Dialating...")
    newim =  dilate(imag)
    newim2=  dilate(newim)
    newim3=  dilate(newim2)
    newim4=  dilate(newim3)
    print("Dialating finished")
    print("Contouring...")
    thresh_img = cv2.threshold(newim4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cnts, hir = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_boy = 0
    big_boy_data = []
    for cnt in cnts:
        approx = cv2.contourArea(cnt)
        if big_boy < approx:
            big_boy = approx
            big_boy_data = cnt
    

    thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    thresh_img = cv2.drawContours(thresh_img, [big_boy_data], -1, (0,0,0), -1)
    thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(thresh_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts, hir = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ymax,xmax = thresh_img.shape
    thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    thresh_img = cv2.drawContours(thresh_img, cnts, -1, (255,255,255), -1)
    print("Contouring finished")
    print("Masking...")
    part_1 = thresh_img.copy()
    part_2 = thresh_img.copy()
    
    cv2.rectangle(thresh_img, (0,0), (int(xmax/2), ymax), (0,255,0), 3)
    cv2.rectangle(thresh_img, (int(xmax/2),0), (xmax, ymax), (0,255,0), 3)

    part_1 = cv2.rectangle(part_1, (0,0), (int(xmax/2), ymax), (0,0,0), -1)
    part_2 = cv2.rectangle(part_2, (int(xmax/2),0), (xmax, ymax), (0,0,0), -1)
    
    result = cv2.bitwise_and(ogs[c - 1], part_1)
    result[thresh_img==0] = 255
    result2 = cv2.bitwise_and(ogs[c - 1], part_2)
    result2[thresh_img==0] = 255
    print("Masking finished")
    print("Reading string: ")
    print(pytesseract.image_to_string(result, config=custom_config2))
    print(pytesseract.image_to_string(result2, config=custom_config2))
    og = cv2.resize(ogs[c - 1], (int(ymax/2), int(xmax/2))) 
    thresh_img = cv2.resize(thresh_img, (int(ymax/2), int(xmax/2))) 
    result = cv2.resize(result, (int(ymax/2), int(xmax/2))) 
    result2 = cv2.resize(result2, (int(ymax/2), int(xmax/2))) 
    cv2.imshow("Original", og)
    cv2.imshow("Mask", thresh_img)
    cv2.imshow("New", result)
    cv2.imshow("New2", result2)
    cv2.waitKey(0)

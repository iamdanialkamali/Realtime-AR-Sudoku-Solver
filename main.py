import traceback
#########3
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

#####
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import cv2
import numpy as np
import time


import arabic_reshaper

from bidi.algorithm import get_display

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

input_img=cv2.imread("sudoko.jpg",3)

from keras.models import load_model
model = load_model('number.h5')
###############################################################

def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)  # floored quotient should be used here.
            for x in range(secTopX, secTopX + 3):
                for y in range(secTopY, secTopY + 3):
                    if grid[x][y] == e:
                        return False
            return True
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

######################################################3

def process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    return dilated

def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # print(len(contours))
    if len(contours) > 0 :
        largest_contour = np.squeeze(contours[0])
        sums = [sum(i) for i in largest_contour]
        differences = [i[0] - i[1] for i in largest_contour]

        top_left = np.argmin(sums)
        top_right = np.argmax(differences)
        bottom_right = np.argmax(sums)
        bottom_left = np.argmin(differences)

        corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
                   largest_contour[bottom_right]]
        return corners
    raise ValueError("NO CONTURE")

def get_block_num(block):

    img = np.copy(block)

    (thresh, img) = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY|cv2.THRESH_OTSU )
    img = np.abs(img - np.array([1]))
    img = img.astype(np.float32)
    # print(list(img))
    # print(np.sum(img))
    # print(img.shape)
    if (np.sum(img) > 10):
        img = cv2.resize(img, (32, 32))
        cv2.imshow("",img)

        # img = img[4:32,4:32]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        eroded = cv2.erode(img, kernel, iterations=1)
        #
        # cv2.imshow("",eroded)
        # print(eroded)
        img1 = np.array(eroded).reshape((1, 32, 32, 1))
        x = model.predict(img1)
        # print(x)

        return np.argmax(x[0])
    else:
        # print("WWWWWWWWWW")
        return -1

def inverse_perspective(img, dst_img, pts):
    dst_img = dst_img.copy()
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1],
                           [img.shape[1] - 1, img.shape[0] - 1]], dtype='float32')

    cv2.fillConvexPoly(dst_img, np.ceil(np.array([pts[0], pts[2], pts[3], pts[1]])).astype(int), 0, 16)
    M = cv2.getPerspectiveTransform(pts_source, pts)
    dst = cv2.warpPerspective(img, M, (dst_img.shape[1], dst_img.shape[0]))
    return dst_img + dst



    # Replace the URL with your own IPwebcam shot.jpg IP:port
url = 'http://192.168.43.189:8080/shot.jpg'
import requests
import numpy as np
import cv2


# video_capture = cv2.VideoCapture(0)

while True:
    # ret,input_img =  video_capture.read()
    try:
        img_res = requests.get(url)
        img_arr = np.array(bytearray(img_res.content), dtype=np.uint8)
        input_img = cv2.imdecode(img_arr, -1)
        input_img = cv2.resize(input_img,(1200,950))
        res = process(input_img)
        points = get_corners(res)
        points = np.float32(points)
        tl, tr, bl, br = points
        maxWidth = 900
        maxHeight = 900
        pts = np.array([[0, 0],[maxHeight - 1, 0],[0, maxWidth - 1],[ maxHeight - 1,maxWidth - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(points, pts)
        dst = cv2.warpPerspective(input_img, M, (maxHeight,maxWidth))

        block_x = maxWidth/9
        block_y = maxHeight/9
        block_dx = 20
        block_dy = 20
        matrix = np.zeros((9,9))
        unsolved_blocks = []
        gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        i, j = 4,4
        num_img = gray_dst[int(i * block_y) + block_dy:int((i + 1) * block_y) - block_dy,
                  int(j * block_x) + block_dx:int((j + 1) * block_x) - block_dx]
        num = get_block_num(num_img)
        print(num)

        # for i in range(9):
        #     for j in range(9):
        #         # i, j = 4,4
        #         num_img = gray_dst[int(i * block_y) + block_dy:int((i + 1) * block_y) - block_dy,
        #                   int(j * block_x) + block_dx:int((j + 1) * block_x) - block_dx]
        #         num = get_block_num(num_img)
        #         matrix[i][j] = num
        # print(matrix)
        # print(num)
        # # for i in range(9):
        #     for j in range(9):
        #         num_img = gray_dst[int(i * block_y) + block_dy:int((i + 1) * block_y) - block_dy,int(j * block_x) + block_dx:int((j + 1) * block_x) - block_dx]
        #         num = get_block_num(num_img)
        #         matrix[i][j] = num
        #         if num == -1 :
        #             unsolved_blocks.append([i,j])
        # num_img = gray_dst[int(i * block_y) + block_dy:int((i + 1) * block_y) - block_dy,int(j * block_x) + block_dx:int((j + 1) * block_x) - block_dx]
        # num = get_block_num(num_img)

        # solved_matrix = np.copy(matrix)
        # solveSudoku(solved_matrix)
        # print(matrix)
        # print(solved_matrix)


    except:
        print(traceback.format_exc())



    # try:
    #
    #
    #     matrix = np.zeros((9,9))
    #     unsolved_blocks = []
    #     gray_dst =cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #     for i in range(9):
    #         for j in range(9):
    #             num_img = gray_dst[j*block_x+block_dx:min((j+1)*block_x-block_dx,maxWidth),int(i*block_y+block_dy):min((i+1)*block_y-block_dy,maxHeight)]
    #             num = get_block_num(num_img)
    #             matrix[j][i] = num
    #             if num == 0 :
    #                 unsolved_blocks.append([j,i])
    #     i,j = 8,8
    #     # cv2.imshow('Video', gray_dst[j*block_x+block_dx:min((j+1)*block_x-block_dx,maxWidth),int(i*block_y+block_dy):min((i+1)*block_y-block_dy,maxHeight)])
    #
    #     solved_matrix = np.copy(matrix)
    #     solveSudoku(solved_matrix)
    #
    #     en2fa = {
    #         0: '۰',
    #         1: '۱',
    #         2: '۲',
    #         3: '۳',
    #         4: '۴',
    #         5: '۵',
    #         6: '۶',
    #         7: '۷',
    #         8: '۸',
    #         9: '۹'
    #     }
    #
    #     fontFile = "b_nazanin.ttf"
    #     img_pil = Image.fromarray(dst)
    #     font = ImageFont.truetype(fontFile, 60)
    #     draw = ImageDraw.Draw(img_pil)
    #     # if np.sum(matrix)< 15:
    #     #     continue
    #     print("SOLVING")
    #     for block in unsolved_blocks:
    #         num = solved_matrix[block[0], block[1]]
    #         x = block_x * block[0] + block_x * 0.3
    #         y = block_y * block[1] + block_y * 0.3
    #         b, g, r, a = 0, 0, 0, 10
    #         draw.text((y, x), en2fa[int(num)], font=font, fill=(b, g, r, a))
    #     print("SOLVED")
    #     print(solved_matrix)
    #     print(matrix)
    #
    #
    #
    #     final = inverse_perspective(np.array(img_pil), input_img, np.array((tl, tr, bl, br)))
    #     # print(final)
    #     cv2.imshow('Video', np.array(final))
    #     # break
    #
    #
    # except Exception:
    #     # print(Exception.__traceback__.__str__())
    #     print(traceback.format_exc())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('9.png',num_img)

        # break

# When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()

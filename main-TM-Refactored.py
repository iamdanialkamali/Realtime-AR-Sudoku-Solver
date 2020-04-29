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
import requests
import numpy as np
import cv2

from numpy import unravel_index

from keras.models import load_model
model = load_model('m_kheyli_hast.h5')
solving_start_time = time.time()
SOLVING_THRESHOLD = 1.5

templates = []
for i in range(1,10):
    templates.append(cv2.imread('./{}.png'.format(i),0))
video_capture = cv2.VideoCapture(0)

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
    if time.time() - solving_start_time > SOLVING_THRESHOLD:
        return True

    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the curren t cell for backtracking
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
    if len(contours) > 0:
        largest_contour = np.squeeze(contours[0])
        sums = [sum(i) for i in largest_contour]
        differences = [i[0] - i[1] for i in largest_contour]

        top_left = np.argmin(sums)
        top_right = np.argmax(differences)
        bottom_right = np.argmax(sums)
        bottom_left = np.argmin(differences)

        corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
                   largest_contour[bottom_right]]
        corners = np.float32(corners)
        return corners

    raise ValueError("NO CONTURE")

def get_block_num_TM(block):
    i = 1
    values = []
    img = np.copy(block)
    for template in templates:
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        indx = unravel_index(res.argmax(), res.shape)
        values.append(res[indx])
        if res[indx] > 0.9:
            return i
        i+=1
    if max(values) < 0.6:
        return 0

    return np.argmax(np.array(values)) + 1


def get_block_num(block):

    img = np.copy(block)

    # img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 2)

    img = cv2.bitwise_not(img, 0)


    img = img.astype(np.float32)
    # if (np.sum(img) > 5):
    img = cv2.resize(img, (32, 32))

    img1 = np.array(img).reshape((1, 32, 32, 1))
    x = model.predict(img1)

    return np.argmax(x[0])
    # else:
    #     return 0

def inverse_perspective(img, dst_img, pts):
    dst_img = dst_img.copy()
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1],
                           [img.shape[1] - 1, img.shape[0] - 1]], dtype='float32')

    cv2.fillConvexPoly(dst_img, np.ceil(np.array([pts[0], pts[2], pts[3], pts[1]])).astype(int), 0, 16)
    M = cv2.getPerspectiveTransform(pts_source, pts)
    dst = cv2.warpPerspective(img, M, (dst_img.shape[1], dst_img.shape[0]))
    return dst_img + dst



    # Replace the URL with your own IPwebcam shot.jpg IP:port

def get_ip_camera():
    url = 'http://192.168.43.189:8080/shot.jpg'
    img_res = requests.get(url)
    img_arr = np.array(bytearray(img_res.content), dtype=np.uint8)
    input_img = cv2.imdecode(img_arr, -1)
    return input_img
def get_webcam():
    ret,input_img = video_capture.read()
    return input_img
maxWidth = 900
maxHeight = 900

history = []
sudoku_results = []
wraped_history = np.zeros((4,2))

answers = []

en2fa = {
    0: '۰',
    1: '۱',
    2: '۲',
    3: '۳',
    4: '۴',
    5: '۵',
    6: '۶',
    7: '۷',
    8: '۸',
    9: '۹'
}

fontFile = "b_nazanin.ttf"
font = ImageFont.truetype(fontFile, 60)

def prespective_transform(points,input_img):
    pts = np.array([[0, 0], [maxHeight - 1, 0], [0, maxWidth - 1], [maxHeight - 1, maxWidth - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(points, pts)
    dst = cv2.warpPerspective(input_img, M, (maxHeight, maxWidth))
    return dst



def matrix_not_in_history(matrix):
    if len(history) == 0:
        return True
    x = np.abs(np.array([np.sum(matrix- past) for past in history]))
    if(np.min(x)>50):
        return True
    return False


def get_matrix_index(matrix):
    for i,past in enumerate(history):
        if np.sum(np.array(past)- np.array(matrix))<50:
            return i
while True:
    try:
        input_img = get_ip_camera()
        # input_img = get_webcam()
        time1 = time.time()
        preproccessed = process(input_img)
        points = get_corners(preproccessed)
        tl, tr, bl, br = points

        dst = prespective_transform(points,input_img)


        gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        print(np.sum(points - wraped_history))
        if np.abs(np.sum(points - wraped_history))>400:
            wraped_history = points
            block_x = maxWidth/9
            block_y = maxHeight/9
            block_dx = 0
            block_dy = 0

            matrix = np.zeros((9,9))
            unsolved_blocks = []
            # print("preproccess time is {}".format(time.time()-time1))

            time1 = time.time()
            for i in range(9):
                for j in range(9):
                    num_img = gray_dst[int(i * block_y) + block_dy:int((i + 1) * block_y) - block_dy,int(j * block_x) + block_dx:int((j + 1) * block_x) - block_dx]
                    num = get_block_num_TM(num_img)
                    matrix[i][j] = num
                    if num == 0 :
                        unsolved_blocks.append([i,j])

        # print("matrix fill time is {}".format(time.time()-time1))
        if np.sum(matrix)>100:
            if matrix_not_in_history(matrix):
                # print("SOLVING")
                solving_start_time = time.time()
                solved_matrix = np.copy(matrix)
                solveSudoku(solved_matrix)
                # print("SOLVED time is {} ".format(time.time()-solving_start_time))
                # print(solved_matrix)
                time1 = time.time()
                img_pil = Image.fromarray(dst)
                draw = ImageDraw.Draw(img_pil)
                for block in unsolved_blocks:
                    num = solved_matrix[block[0], block[1]]
                    x = block_x * block[0] + block_x * 0.3
                    y = block_y * block[1] + block_y * 0.3
                    b, g, r, a = 0, 0, 0, 10
                    draw.text((y, x), en2fa[int(num)], font=font, fill=(b, g, r, a))
                # print("DRAW_TIME is {}".format(time.time()-time1))
                sudoku_results.append(img_pil)
                time1 = time.time()
                final = inverse_perspective(np.array(img_pil), input_img, np.array((tl, tr, bl, br)))
                # print("inverse Wrap time is {}".format(time.time()-time1))
                answers.append(solved_matrix[:])
                cv2.imshow('Video', np.array(final))
                cv2.imwrite('Video.png', np.array(final))
                cv2.waitKey(1)

                history.append(matrix[:])
            else :
                solved_matrix = answers[get_matrix_index(matrix)]
                img_pil = sudoku_results[get_matrix_index(matrix)]
                final = inverse_perspective(np.array(img_pil), input_img, np.array((tl, tr, bl, br)))
                cv2.imshow('Video', np.array(final))
                cv2.waitKey(1)


        else:
            cv2.imshow('Video', input_img)
            cv2.waitKey(1)


    except:
        pass
        # print(traceback.format_exc())



    

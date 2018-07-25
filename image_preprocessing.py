# -*- coding: utf-8 -*-
import numpy as np
import random
import re
import cv2
from matplotlib import pyplot as plt
import os
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('CNN.conf')

def make_one_hot(data1,index):
    return (np.arange(index)==data1[:,None]).astype(np.integer)
    #return (np.arange(index)==data1[:,None]).astype(np.float)

def array_to_subdir_and_image(DIR_NAME,
                              IMAGE_WIDTH,
                              IMAGE_HEIGHT,
                              IMAGE_DEPTH,
                              image_array,
                              kinds_array
                              ):
    for i in range(0, len(kinds_array)):
        #print(tY[i])
        #print(tX[i])
        image = image_array[i] * 255
        image = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
        fn = DIR_NAME + '/' + str(kinds_array[i]) + '/light_'+ str(i) + '.png'
        pn = DIR_NAME + '/' + str(kinds_array[i])
        folder = os.path.exists(pn)
        if not folder:
            os.makedirs(pn)
        cv2.imwrite(fn, image)


######将DIR_NAME目录下的文件夹为类别，其中的图片文件读取后转换为一维数组
def subdir_and_image_to_array(DIR_NAME,
                              VALIDATION_PERCENTAGE,
                              TESEING_PERCENTAGE,
                              IMAGE_WIDTH,
                              IMAGE_HEIGHT,
                              IMAGE_DEPTH,
                              SHUFFLE_DATA
                              ):
    fn_list      = []
    training_X   = []
    training_Y   = []
    validation_X = []
    validation_Y = []
    testing_X    = []
    testing_Y    = []

    ########将DIR_NAME目录下文件夹指定为类型，将其中的图片文件文件名、类型名、类型编号存入fn_list列表，分割符｜
    for root1,dirs1,files1 in os.walk(DIR_NAME):
        for index in range(len(dirs1)):
            dir1 = dirs1[index]
            #print(dir1)
            for root2,dirs2,files2 in os.walk(DIR_NAME + '/' + dir1):
                for file2 in files2:
                    ######fn由三部分组成，图片文件的路径与文件名、类型名、类型编号，分割符为｜
                    fn = DIR_NAME + '/' + dir1 + '/' + file2 + '|' + dir1 + '|' + str(index)
                    fn_list.append(fn)
        category_number = index + 1

    ########将列表的顺序打乱，便于后边神经网络训练
    if SHUFFLE_DATA == 1:
        random.shuffle(fn_list)

    ######## 按照列表读取文件并生成训练、验证、测试数据
    for fn in fn_list:
        fn_split          = re.split('\|',fn)
        image_filename    = fn_split[0]
        image_category    = fn_split[1]
        image_category_id = int(fn_split[2])
        if IMAGE_DEPTH == 1:
            template = cv2.imread(image_filename,0)
        if IMAGE_DEPTH == 3:
            template = cv2.imread(image_filename,1)
        #print(template)
        #print(template.shape)
        #print(fn)
        try:
            template = cv2.resize(template,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation = cv2.INTER_CUBIC)
            # 随机划分数据
            chance = float(np.random.randint(10000))/100
            if chance < VALIDATION_PERCENTAGE:
                validation_X.append(template)
                validation_Y.append(image_category_id)
            elif chance < (TESEING_PERCENTAGE + VALIDATION_PERCENTAGE):
                testing_X.append(template)
                testing_Y.append(image_category_id)
            else:
                training_X.append(template)
                training_Y.append(image_category_id)
        except:
            print('find error picture file:' + image_filename)

    training_X_array     = np.array(training_X, dtype                             = float)
    training_Y_array     = np.array(training_Y, dtype                             = float)
    training_X_array     = training_X_array / 255
    training_X_array     = training_X_array.reshape(-1,IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH)
    training_Y_one_hot   = make_one_hot(training_Y_array,category_number)

    validation_X_array   = np.array(validation_X, dtype                           = float)
    validation_Y_array   = np.array(validation_Y, dtype                           = float)
    validation_X_array   = validation_X_array / 255
    validation_X_array   = validation_X_array.reshape(-1,IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH)
    validation_Y_one_hot = make_one_hot(validation_Y_array,category_number)

    testing_X_array      = np.array(testing_X, dtype                              = float)
    testing_Y_array      = np.array(testing_Y, dtype                              = float)
    testing_X_array      = testing_X_array / 255
    testing_X_array      = testing_X_array.reshape(-1,IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH)
    testing_Y_one_hot    = make_one_hot(testing_Y_array,category_number)
    return(category_number, len(training_X), len(validation_X), len(testing_X), training_X_array, training_Y_one_hot, validation_X_array, validation_Y_one_hot, testing_X_array, testing_Y_one_hot)



def image_increase(dirname, filename):
    image = cv2.imread(dirname + '/' + filename)

    ########在原图上水平与垂直变换###############################################
    # Flipped Horizontally 水平翻转
    h_filename = dirname + '/' + 'h_' + filename
    h_image     = cv2.flip(image, 1)
    cv2.imwrite(h_filename, h_image)

    # Flipped Vertically 垂直翻转
    v_filename = dirname + '/' + 'v_' + filename
    v_image     = cv2.flip(image, 0)
    cv2.imwrite(v_filename, v_image)

    # Flipped Horizontally & Vertically 水平垂直翻转
    hv_filename = dirname + '/' + 'hv_' + filename
    hv_image     = cv2.flip(image, -1)
    cv2.imwrite(hv_filename, hv_image)

    ######## numpy rotate 90 逆时针旋转90度，然后水平与垂直变换##################
    r_filename = dirname + '/' + 'r_' + filename
    r_image = np.rot90(image)
    cv2.imwrite(r_filename, r_image)

    # Flipped Horizontally 水平翻转
    rh_filename = dirname + '/' + 'rh_' + filename
    rh_image     = cv2.flip(r_image, 1)
    cv2.imwrite(rh_filename, rh_image)

    # Flipped Vertically 垂直翻转
    rv_filename = dirname + '/' + 'rv_' + filename
    rv_image     = cv2.flip(r_image, 0)
    cv2.imwrite(rv_filename, rv_image)

    # Flipped Horizontally & Vertically 水平垂直翻转
    rhv_filename = dirname + '/' + 'rhv_' + filename
    rhv_image     = cv2.flip(r_image, -1)
    cv2.imwrite(rhv_filename, rhv_image)


def subdir_and_image_increase(DIR_NAME):
    for root1,dirs1,files1 in os.walk(DIR_NAME):
        for index in range(len(dirs1)):
            dir1 = dirs1[index]
            #print(dir1)
            for root2,dirs2,files2 in os.walk(DIR_NAME + '/' + dir1):
                for file2 in files2:
                    image_increase(DIR_NAME+dir1,file2)

                    #fn = DIR_NAME + dir1 + '/' + file2
                    #print(fn)
                    #template = cv2.imread(fn,0)



if __name__ == '__main__':
    PIC_DIR = cf.get('CNN_train', 'PIC_DIR')

    #subdir_and_image_increase(PIC_DIR)
    #image_increase('data/output/a','a.jpg')


    X,Y,a,b,c,d = subdir_and_image_to_array(PIC_DIR,10,10)
    print(X)
    print(Y)

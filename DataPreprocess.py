import glob

import cv2
from PIL import Image
import numpy as np
import os
import threading

def video_to_png(video_path, png_save_path):
    # 读取对应的脸部坐标文件
    txt_path = video_path.replace('.avi','.txt')
    # 读取文件名，创建保存图片的文件夹
    folder_name = video_path.split('\\')[-1]
    folder_name = folder_name.replace('.avi','')
    folder_path = os.path.join(png_save_path, folder_name)
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)

    with open(txt_path, 'r') as txt:
        cap = cv2.VideoCapture(video_path)  # 获取视频对象
        isOpened = cap.isOpened             # 判断是否打开
        imageNum = 0

        frame_num = 0
        while (isOpened):
            (frameState, frame) = cap.read()  # 记录每帧及获取状态
            frame_num += 1

            if frameState == True and frame_num % 15 == 0: #每秒抽2帧
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # # 读取该帧的脸部坐标
                # coord = txt.readline()
                # coord = coord.split(',')
                # # box = (int(coord[2]), int(coord[0]), int(coord[3]), int(coord[1]))
                # # 裁切图片
                # frame = frame.crop(box)
                # 存储
                fileName = os.path.join(folder_path,str(imageNum) + '.png')
                # fileName = os.path.join(png_save_path,fileName)
                frame.save(fileName)
                imageNum += 1
            elif frameState == False:
                break

        print('finish!')
        cap.release()


# 读取所有.avi文件
videos = glob.glob('Data_Origin/*.avi')

for video in videos:
    video_to_png(video,'Data_Processed_Train')
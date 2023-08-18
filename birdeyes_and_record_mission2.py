import time
import cv2
import config as cfg

import os
import sys
import signal
import csv

import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import serial
import json
import message

CWD_PATH = os.getcwd()

ser = serial.Serial(
        port='COM3',
        baudrate=9600
    )

'''
def recording():
    if cfg.recording:
        cfg.recording = False
        cfg.f.close()
    else:
        cfg.recording = True
        if cfg.currentDir == '':
            cfg.currentDir = time.strftime('%Y-%m-%d')
            os.mkdir(cfg.outputDir+cfg.currentDir)
            cfg.f=open(cfg.outputDir+cfg.currentDir+'/data.csv','w')
        else:
            cfg.f=open(cfg.outputDir+cfg.currentDir+'/data.csv','a')
        cfg.fwriter = csv.writer(cfg.f)
'''

def saveimage(birdView):
    if cfg.recording:
        myfile = 'img_'+time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+str(cfg.cnt)+'.jpg'
        print(myfile, cfg.wheel)

        if(cfg.wheel == 1):
            output_folder = "./captured_frames_1"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)
        
        elif(cfg.wheel == 2):
            output_folder = "./captured_frames_2"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)
        
        elif(cfg.wheel == 3):
            output_folder = "./captured_frames_3"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)

        elif(cfg.wheel == 4):
            output_folder = "./captured_frames_4"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)

        elif(cfg.wheel == 5):
            output_folder = "./captured_frames_5"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)
        
        elif(cfg.wheel == 6):
            output_folder = "./captured_frames_6"
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, birdView)
        

        #cfg.fwriter.writerow((myfile, cfg.wheel))
        cv2.imshow('TEST_birdview', birdView)
        # cv2.imwrite(cfg.outputDir+cfg.currentDir+'/'+ myfile, birdView)

        cfg.cnt += 1

def saveimage_TFlight(cropped_image):
    if cfg.recording:
        myfile = 'TFimg_'+time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+str(cfg.cnt)+'.jpg'
        print(myfile)
        print(stop)

        if(fw <= 90 and stop > 90):
            output_folder = "./TF_captured_frames_1" #정지정지(빨간불)
            os.makedirs(output_folder, exist_ok=True)
            cfg.currentDir = output_folder
            cv2.imwrite(output_folder + '/'+ myfile, cropped_image)
        

        #cfg.fwriter.writerow((myfile, cfg.wheel))
        cv2.imshow('TEST_TFview', cropped_image)
        # cv2.imwrite(cfg.outputDir+cfg.currentDir+'/'+ myfile, birdView)

        cfg.cnt += 1

def perspectiveWarp(inpImage):

    # Get image size
    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Perspective points to be warped
    lu_x = 230-30-60+40+10+40+40-60-40-20+60-40+60-40-40-20+60
    lu_y = 200+60-40-60-40-40+120+60-60-60+60+60

    ru_x = 410+30+60-40-10-40-40+60+40+20-60+40-60+40+40+20-60
    ru_y = 200+60-40-60-40-40+120+60-60-60+60+60

    ld_x = 0+60-40+60-20-40-30+40
    ld_y = 480-100+40+60

    rd_x = 640-60+40-60+20+40+30-40
    rd_y = 480-100+40+60

    src = np.float32([[lu_x, lu_y],
                      [ru_x, ru_y],
                      [ld_x, ld_y],
                      [rd_x, rd_y]])

    dst = np.float32([[100-100, 0],
                      [540+120, 0],
                      [100-100, 500+140],
                      [540+120, 500+140]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)
    # print(img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    birdseyeLeft  = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    

    return birdseye, birdseyeLeft, birdseyeRight, minv


'''
Image = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while(True):
    ret, frame = Image.read()
    cv2.imshow('video', frame)


    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
    
    if ser.readable():
        dict_json = message.receive(ser) #엔코더의 값(Real_angle), 초음파센서 8개 값(D1~D8)
        if (dict_json != 0):
            print(dict_json)
    
    

Image.release()
cv2.destroyAllWindows()
'''

vid = cv2.VideoCapture(1,cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FPS, 60)
enc_val=0
steer_LR=0
fw = 0
stop = 0
start_flag = False
count = 0


def capture_frames_from_webcam(output_folder, num_frames=100, capture_interval=0.1):
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # 웹캠에서 프레임 읽어오기
    for i in range(num_frames):
        ret, frame = vid.read()

        if not ret:
            print("웹캠에서 프레임을 읽어오는데 문제가 발생했습니다.")
            break
        birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
        # 프레임 이미지를 파일로 저장
        file_name = f"{output_folder}/frame_{i:03d}.png"
        cv2.imwrite(file_name, birdView)

        # 프레임을 화면에 출력 (선택사항)
        cv2.imshow("Frame", birdView)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 초당 5개 프레임으로 제한
        time.sleep(capture_interval)

    # 웹캠 캡처 객체 해제
    cap.release()
    cv2.destroyAllWindows()

'''
if __name__ == "__main__":
    # 프레임을 저장할 폴더 생성
    output_folder = "captured_frames"
    os.makedirs(output_folder, exist_ok=True)

    # 초당 5개의 프레임을 받아와서 사진으로 저장 (총 100개의 프레임)
    capture_frames_from_webcam(output_folder, num_frames=100, capture_interval=0.2)
    '''



while(True):
      
    # Capture the video frame
    # by frame
    #-------------------
    
    if ser.readable():
        #print("serial OK!!!")
        dict_json = message.receive(ser) #엔코더의 값(Real_angle), 초음파센서 8개 값(D1~D8)
        if (dict_json != 0):
            enc_val = dict_json["enc_val"]
            steer_LR = dict_json["steer_LR"]
            fw = dict_json["fw"]
            stop = dict_json["stop"]
            # print(stop)
    # print(fw)
    
    ret, frame = vid.read()
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
  
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    cv2.imshow('birdView', birdView)

    cropped_image = frame[0:80, 20:300]
    #cv2.imshow('cropped_image', cropped_image)
    

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    #----------------------------------------
    
    if fw >= 20: #115:'s'
        if start_flag == False: 
            start_flag = True
        else:
            start_flag = False
        #print('start flag:',start_flag)
        
    
    elif fw < 10 and fw > 0:
        if start_flag == True: 
            start_flag = False
            cfg.recording = False

    
    if stop >= 90:
        cfg.recording = True
        if fw < 90:
            saveimage_TFlight(birdView)
            

    
    

    if start_flag:
        # Left arrow: 81, Right arrow: 83, Up arrow: 82, Down arrow: 84
        if steer_LR < 5 and steer_LR > -5: 
            cfg.recording = True
            #print('Straight')
            cfg.wheel = 3
            #recording()
            if cfg.recording:
                #start_flag = True
                saveimage(birdView)
                print(3)
        
        #print('cfg.recording:',cfg.recording)
        elif steer_LR > -50 and steer_LR <= -5: 
            cfg.recording = True
            cfg.wheel = 2
            #recording()
            if cfg.recording:
                #start_flag = True
                saveimage(birdView)
                print(2)
        elif steer_LR <= -51: 
            cfg.recording = True
            cfg.wheel = 1
            #recording()
            if cfg.recording:
                #start_flag = True
                saveimage(birdView)
                print(1)
        elif steer_LR >= 5 and steer_LR < 50: 
            cfg.recording = True
            cfg.wheel = 4
            #recording()
            if cfg.recording:
                #start_flag = True
                saveimage(birdView)
                print(4)
        elif steer_LR >= 51: 
            cfg.recording = True
            cfg.wheel = 5
            #recording()
            if cfg.recording:
                #start_flag = True
                saveimage(birdView)
                print(5)
        

            

        else:
           start_flag = False
           cfg.cnt = 0

    
    
    
  
vid.release()
cv2.destroyAllWindows()
            
    
    


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
import serial
import message

'''
ser = serial.Serial(
        port='COM8',
        baudrate=9600
    )
'''
def print_enter_to_start():
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("██╗   ██╗███╗   ██╗██╗ ██████╗ ██████╗ ███╗   ██╗")
    print("██║   ██║████╗  ██║██║██╔════╝██╔═══██╗████╗  ██║")
    print("██║   ██║██╔██╗ ██║██║██║     ██║   ██║██╔██╗ ██║")
    print("██║   ██║██║╚██╗██║██║██║     ██║   ██║██║╚██╗██║")
    print("╚██████╔╝██║ ╚████║██║╚██████╗╚██████╔╝██║ ╚████║")
    print(" ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝")
    print("")
    print("███████ ███    ██ ████████ ███████ ██████      ████████  ██████      ███████ ████████  █████  ██████  ████████   ██")
    print("██      ████   ██    ██    ██      ██   ██        ██    ██    ██     ██         ██    ██   ██ ██   ██    ██      ██")
    print("█████   ██ ██  ██    ██    █████   ██████         ██    ██    ██     ███████    ██    ███████ ██████     ██      ██")
    print("██      ██  ██ ██    ██    ██      ██   ██        ██    ██    ██          ██    ██    ██   ██ ██   ██    ██        ")
    print("███████ ██   ████    ██    ███████ ██   ██        ██     ██████      ███████    ██    ██   ██ ██   ██    ██      ██")
    print("")

def perspectiveWarp(inpImage):

    # Get image size
    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Perspective points to be warped
    lu_x = 190
    lu_y = 260

    ru_x = 450
    ru_y = 260


    ld_x = 30
    ld_y = 480

    rd_x = 610
    rd_y = 480

    src = np.float32([[lu_x, lu_y],
                      [ru_x, ru_y],
                      [ld_x, ld_y],
                      [rd_x, rd_y]])

    dst = np.float32([[0, 0],
                      [660, 0],
                      [0, 640],
                      [660, 640]])

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Inverse matrix to unwarp the image for final window
    # minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)
    # print(img_size)

    # Get the birdseye window dimensions
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    #birdseyeLeft  = birdseye[0:height, 0:width // 2]
    #birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    

    return birdseye


# 딥러닝 모델 정의 (예시로 간단한 CNN 모델을 사용)
class LaneClassifier(nn.Module):
    '''
    def __init__(self):
        super(LaneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    '''
    def __init__(self):
        super(LaneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 학습된 모델 불러오기
model = LaneClassifier()
model.load_state_dict(torch.load("lane_classifier_model_class6_epoch20_0813.pth"))
# lane_classifier_model_class6_epoch20_0815
model.eval()

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# 이미지 분류 함수
def classify_image(image_path, model):
    image = preprocess_image(image_path)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    # print(predicted)
    predicted = 5
    #return predicted.item()
    return predicted

def classify_video(video_path, model): #비디오 영상 넣을때 테스트용
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 비디오 정보 가져오기
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 결과를 저장할 리스트
    results = []
    x = input("")

    # 프레임별로 이미지 분류 수행
    while x=="":
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, dsize=(640, 480))
        # cv2.imshow('frame', frame)
        birdView = perspectiveWarp(frame)
        # print(fps)
        

        if not ret:
            break

        # 프레임 이미지를 임시 파일로 저장
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path,birdView)
        #birdView = birdView.copy()
        #temp_image = birdView
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 이미지 분류 결과를 저장
        result = classify_image(temp_image_path, model)
        #print(result)
        if result == 0:
            res_dir = "LEFT_MAX"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (120, 440), (255,255,255), 2)
        elif result == 1:
            res_dir = "LEFT_MID"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (300, 350), (100, 200), (255,255,255), 2)
        elif result == 2:
            res_dir = "STRAIGHT"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)
        elif result == 3:
            res_dir = "RIGHT_MID"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (300, 350), (100, 200), (255,255,255), 2)
        elif result == 4:
            res_dir = "RIGHT_MAX"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)
        elif result == 5:
            res_dir = "STOP"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)

        cv2.imshow('birdview', birdView)
        results.append(result)
        # print(result)
        temp_image_path = None

        # 임시 파일 삭제
        #os.remove(temp_image_path)
        # temp_image = None


    # 비디오 캡처 객체 해제
    cap.release()

    return results


def classify_realtime_video(model): #실전 리얼타임 영상 테스트용
    # 비디오 캡처 객체 생성
    ser = serial.Serial(
        port='COM3',
        baudrate=9600
    )
    ser.flush()
        
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)
    


    # 결과를 저장할 리스트
    results = []
    x = input("")
    # 프레임별로 이미지 분류 수행
    while x=="":
        
        # print(fps)
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(640, 480))
        birdView = perspectiveWarp(frame)

        '''
        if ser.readable():
            #print("serial OK!!!")
            dict_json = message.receive(ser) #엔코더의 값(Real_angle), 초음파센서 8개 값(D1~D8)
            #ser.flush()
            
            if (dict_json != 0):
                enc_val = dict_json["enc_val"]
                print("enc_val:  ", enc_val)
                #@steer_LR = dict_json["steer_LR"]
                #fw = dict_json["fw"]
                #stop = dict_json["stop"]
            
            
            if (dict_json != 0):
                enc_val = dict_json["enc_val"]
                print("enc_val:  ", enc_val)
                #@steer_LR = dict_json["steer_LR"]
                #fw = dict_json["fw"]
                #stop = dict_json["stop"]
            '''
        

        if not ret:
            break

        # 프레임 이미지를 임시 파일로 저장
        temp_image_path = "temp_frame.jpg"
        # birdView_pre = preprocess_image(birdView)
        cv2.imwrite(temp_image_path,birdView)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 이미지 분류 결과를 저장
        result = classify_image(temp_image_path, model)
        # message.send(ser, n)
        if result == 0:
            res_dir = "LEFT_MAX" #12345 그것
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (120, 440), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 1)
        elif result == 1:
            res_dir = "LEFT_MID"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (300, 350), (100, 200), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 2)
        elif result == 2:
            res_dir = "STRAIGHT"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 3)
        elif result == 3:
            res_dir = "RIGHT_MID"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (300, 350), (100, 200), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 4)
        elif result == 4:
            res_dir = "RIGHT_MAX"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 5)
        elif result == 5:
            res_dir = "STOP"
            cv2.putText(birdView, res_dir, (270, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
            #cv2.arrowedLine(birdView, (270, 440), (270, 340), (255,255,255), 2)
            if ser.readable():
                message.send(ser, 6)
        cv2.imshow('birdview', birdView)
        # cv2.imshow('frame', frame)
        results.append(result)

        # 임시 파일 삭제
        # os.remove(temp_image_path)
        temp_image_path = None


    # 비디오 캡처 객체 해제
    cap.release()

    return results

# 테스트 or 굴리기
# image_path = "test_image10_res1.jpg"  # 테스트할 이미지 파일 경로
video_path = "test_video3.mp4"
print_enter_to_start()
# results = classify_realtime_video(model)
result = classify_video(video_path, model)

#if x == "":
 #   result = classify_video(video_path, model)
 #   results = classify_realtime_video(model)

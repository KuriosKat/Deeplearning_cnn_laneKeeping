'''
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomBrightnessContrast(p=0.8)
    A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0))
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("test_image1_res1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
cv2.imwrite("augmented_image_class1/transformed_image.png", transformed_image)
'''
import os
import cv2
import albumentations as A
import albumentations as B
import albumentations as C
import albumentations as D
import albumentations as E
# from albumentations.pytorch import ToTensor

# 이미지가 있는 폴더 경로
input_folder = 'Augment_in_busan/captured_frames_5'
output_folder_GaussNoise = 'Augment_in_busan/augmented_frames_5_GaussNoise'
output_folder_CoarseDropout = 'Augment_in_busan/augmented_frames_5_CoarseDropout'
output_folder_HueSaturationValue = 'Augment_in_busan/augmented_frames_5_HueSaturationValue'
output_folder_ColorJitter = 'Augment_in_busan/augmented_frames_5_ColorJitter'
output_folder_CoarseDropout2 = 'Augment_in_busan/augmented_frames_5_CoarseDropout2'

# 폴더가 없다면 생성
if not os.path.exists(output_folder_GaussNoise):
    os.makedirs(output_folder_GaussNoise)
if not os.path.exists(output_folder_CoarseDropout):
    os.makedirs(output_folder_CoarseDropout)
if not os.path.exists(output_folder_HueSaturationValue):
    os.makedirs(output_folder_HueSaturationValue)
if not os.path.exists(output_folder_ColorJitter):
    os.makedirs(output_folder_ColorJitter)
if not os.path.exists(output_folder_CoarseDropout2):
    os.makedirs(output_folder_CoarseDropout2)

# Albumentations 변환 함수 생성
transformA = A.Compose([
    A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 80.0))
    # A.CoarseDropout(always_apply=False, p=1.0, max_holes=24, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8)
    # A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20))
])
transformB = B.Compose([
    #A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0))
    # B.CoarseDropout(always_apply=False, p=1.0, max_holes=18, max_height=48, max_width=48, min_holes=12, min_height=24, min_width=24)
    B.GaussNoise(always_apply=False, p=1.0, var_limit=(5.0, 40.0))
])
transformC = C.Compose([
    # A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0))
    #C.CoarseDropout(always_apply=False, p=1.0, max_holes=24, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8)
    C.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-5, 5), val_shift_limit=(-5, 5))
])
transformD = D.Compose([
    # A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0))
    #C.CoarseDropout(always_apply=False, p=1.0, max_holes=24, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8)
    D.ColorJitter(brightness=0.8)
])
transformE = E.Compose([
    #A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0))
    E.CoarseDropout(always_apply=False, p=1.0, max_holes=8, max_height=32, max_width=32, min_holes=8, min_height=24, min_width=24)
    # B.Cutout(num_holes=2, max_h_size=50, max_w_size=50, fill_value=[255, 255, 255], always_apply=True)
])
cnt = 0
# 폴더 내 이미지들을 불러와서 변환 및 저장
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        augmented = transformA(image=image)
        augmented_image = augmented['image']
        # 이미지 저장
        output_path = os.path.join(output_folder_GaussNoise, f'augmentedA_{filename}')
        cv2.imwrite(output_path, augmented_image)
        print("completed A!!!", cnt)
        cnt = cnt+1
cnt = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        augmented = transformB(image=image)
        augmented_image = augmented['image']
        # 이미지 저장
        output_path = os.path.join(output_folder_CoarseDropout, f'augmentedB_{filename}')
        cv2.imwrite(output_path, augmented_image)
        print("completed B!!!", cnt)
        cnt = cnt+1
cnt = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        augmented = transformC(image=image)
        augmented_image = augmented['image']
        # 이미지 저장
        output_path = os.path.join(output_folder_HueSaturationValue, f'augmentedC_{filename}')
        cv2.imwrite(output_path, augmented_image)
        print("completed C!!!", cnt)
        cnt = cnt + 1
cnt = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        augmented = transformD(image=image)
        augmented_image = augmented['image']
        # 이미지 저장
        output_path = os.path.join(output_folder_ColorJitter, f'augmentedD_{filename}')
        cv2.imwrite(output_path, augmented_image)
        print("completed D!!!", cnt)
        cnt = cnt + 1
cnt = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        augmented = transformE(image=image)
        augmented_image = augmented['image']
        # 이미지 저장
        output_path = os.path.join(output_folder_CoarseDropout2, f'augmentedE_{filename}')
        cv2.imwrite(output_path, augmented_image)
        print("completed E!!!", cnt)
        cnt = cnt + 1
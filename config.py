# -*- coding: utf-8 -*-


# Configuration
width = 640  # Video width requested from camera
height = 480  # Video height requested from camera

wheel = 0  #0:straight, 1:left, 2:right

recording = False

cnt = 0
outputDir = './lesson4/data/'
currentDir = 'training'
file = ""
f = ''
fwriter = ''

Voicecontrol = False

AIcontrol = False
modelheight = -160 ###-130 ###-150 #-115 #-130 #-150 #-250 #-200

# training speed setting
maxturn_speed = 60
minturn_speed = 5  ###20  ###15
normal_speed_left = 20
normal_speed_right = 20
wheel_alignment_left = 0
wheel_alignment_right = 0


# testing speed setting(
ai_maxturn_speed = 100
ai_minturn_speed = 20
ai_normal_speed_left = 100
ai_normal_speed_right = 100





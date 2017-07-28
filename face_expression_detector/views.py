# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import time
from cv2 import *
from classifier import *

from django.shortcuts import render


# Create your views here.

def index(request):
    return render(request, 'face_expression_detector/index.html', {})


def detect(request):
    time.sleep(0.05)
    cam = VideoCapture(0)  # 0 -> index of camera
    s, img = cam.read()
    if s:  # frame captured without any errors
        # namedWindow("cam-test")
        # imshow("cam-test", img)
        # destroyWindow("cam-test")
        imwrite("face_expression_detector/static/filename.jpg", img)
        cam.release
        expression = runClassifier("face_expression_detector/static/filename.jpg")
        print "************", expression
    context = {"expression": expression}
    return render(request, 'face_expression_detector/detect.html', context)



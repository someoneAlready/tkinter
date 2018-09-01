#!/usr/bin/env python


import numpy as np

from PIL import Image
from PIL import ImageTk
import Tkinter as tk
import mxnet as mx
import cv2
import os
import time
import gluoncv
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import mxnet as mx
from gluoncv import model_zoo, data, utils
cap = cv2.VideoCapture(0)
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

seg_net = gluoncv.model_zoo.get_model('fcn_resnet50_voc', pretrained=True)
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


def load_test(img, short, max_size=1024, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    img = mx.nd.array(img)
    img = mx.image.resize_short(img, short)
    if isinstance(max_size, int) and max(img.shape) > max_size:
        img = timage.resize_long(img, max_size)
    orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    tensors = img.expand_dims(0)
    return tensors, orig_img



def plot_bbox(img, origin, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True):

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))


    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    bboxes[:, (0, 2)] *= img.shape[0] / float(origin.shape[0])
    bboxes[:, (1, 3)] *= img.shape[1] / float(origin.shape[1])

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(55,255,155),3)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                    (xmin + 3, ymin + 18),
                    font, 0.6, (105, 105, 255), 1)
    return img

def show_frame():
    global operators_val
    global display
    global window

    ret, frame = cap.read()
#    frame = cv2.imread('ade20k_example.jpg')
    frame = cv2.resize(frame, (0,0), fx=0.9, fy=0.9)

    frame = cv2.flip(frame, 1)
    if int(operators_val.get()) == 0:
        pass
    elif int(operators_val.get()) == 1:
        x, img = load_test(frame, short=200)
        output = seg_net.demo(x)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        print(predict.shape)

        mask = get_color_pallete(predict, 'pascal_voc')
        img = np.asarray(mask)
        print(img.shape)

        mask = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        frame = frame.astype(np.float32) + mask.astype(np.float32)
        frame[frame > 255] = 255
        frame = frame.astype(np.uint8)
    elif int(operators_val.get()) == 2:
        x, img = load_test(frame, short=200)
        class_IDs, scores, bounding_boxs = net(x)
        frame = plot_bbox(frame, img, bounding_boxs[0], scores[0],
                        class_IDs[0], class_names=net.classes)
    elif int(operators_val.get()) == 3:
        frame = 255 - frame
    elif int(operators_val.get()) == 4:
        
        model = gluoncv.model_zoo.get_model('psp_resnet50_ade', pretrained=True)
        img, _ = load_test(frame, 150)
        output = model.demo(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        mask = get_color_pallete(predict, 'ade20k')
        img = np.asarray(mask.convert('RGB'))
        mask = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        frame = frame.astype(np.float32) + mask.astype(np.float32)
        frame[frame > 255] = 255
        frame = frame.astype(np.uint8)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    display.imgtk = imgtk 
    display.configure(image=imgtk)
    window.after(10, show_frame) 

def empty():    
    pass

def main():
    global color
    global detect
    global seg
    global display
    global window

    window = tk.Tk()  #Makes main window
    window.wm_title("Digital Microscope")
    window.config(background="#FFFFFF")

    display = tk.Label(window)
    display.pack()

    f = tk.Frame(window).pack()
    

    operators = ['origin', 'segment', 'detect', 'color', 'scene']
    global operators_val 
    operators_val = tk.IntVar()

    for i, operator in enumerate(operators):
        c = tk.Radiobutton(f, text=operator, value=i, 
                variable=operators_val, command=empty)
        c.pack(side='left', padx=10, pady=10)


    show_frame() 
    window.mainloop() 

if __name__ == '__main__':
    main()
    

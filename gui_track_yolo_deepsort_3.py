from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

import threading
import time
import itertools
import PySimpleGUI as sg
import imutils
from imutils.video import FPS
# from imutils.video import WebcamVideoStream
from imutils.video import VideoStream
# import os.path
import numpy as np
import dlib

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def load_info(config_deepsort, device, imgsz, out, weights):
            
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, 
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    device = select_device(device)

    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    return t0, half, device, model, names, deepsort


def save_images(frame_idx, outputs, save_imgs_path, image):
    # Save results (image with detections)
    if len(outputs) != 0:  
        cv2.imwrite(save_imgs_path+'/image'+str(frame_idx)+'.png', image)

def save_videos(vid_path, save_path, vid_writer, vid_cap, save_vids_path, fourcc, image):
    if vid_path != save_path:  # new video
        vid_path = save_path
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # release previous video writer

        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        vid_writer = cv2.VideoWriter(save_vids_path+'/stream.avi', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    vid_writer.write(image)
    return vid_path, vid_writer
    
def write_tracks(frame_idx, save_txt, outputs, txt_path):
    # Write MOT compliant results to file
    if save_txt and len(outputs) != 0:  
        for j, output in enumerate(outputs):
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_w = output[2]
            bbox_h = output[3]
            identity = output[-1]
            with open(txt_path+'/results.txt', 'a') as f:
                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

def getData(source, imgsz):
    webcam = source == '0' or source == '2' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    
    # Set Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        
    data = enumerate(dataset)
    return webcam, dataset, data

# def detect(agnostic_nms, augment, classes, conf_thres, device, 
#            iou_thres, half, model, webcam, names, deepsort, path, img, im0s, save_path, 
#            vid_path, vid_writer, vid_cap, save_vids_path, fourcc, save_vid):
def detect(agnostic_nms, augment, classes, conf_thres, device, 
       iou_thres, half, model, webcam, names, deepsort, path, img, im0s):
        
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()
    
    outputs = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s
            
        im_org = im0.copy()
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            bbox_xywh = []
            confs = []

            # Adapt detections to deep sort input format
            for *xyxy, conf, cls in det:
                img_h, img_w, _ = im0.shape
                x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf.item()])

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass detections to deepsort
            outputs = deepsort.update(xywhs, confss, im0)
            
            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                draw_boxes(im0, bbox_xyxy, identities)
                    
               
    return im0, outputs

username = 'operator'
password = 'lavs206u'

# Camera Information
host = '158.110.145.'
cam_id = '187'
stream = ':8888/axis-cgi/mjpg/video.cgi?.mjpg'

# For now will only show the name of the file that was chosen
image_viewer_column = [[sg.Image(filename='', size=(40,20), key='-IM1-'), sg.Image(filename='', size=(40,20), key='-IM2-')],
                       [sg.Image(filename='', size=(40,20), key='-IM3-'), sg.Image(filename='', size=(40,20), key='-IM4-')]]

sg.theme('LightGrey6')
# ----- Full layout -----
# layout = [[sg.Column(image_viewer_column)]]                #VSeparator = Vertical Separator on the window

layout = [[sg.Radio("None", "Radio", True, size=(10,1)), 
           sg.Radio("Track", "Radio", default=False, size=(10,1), key='-TRACK-'),
           sg.Checkbox('Print Tracks', default=False, key="-PRINT-"),
           sg.Checkbox('Save Images', default=False, key="-SAVEIMG-"),
           sg.Checkbox('Record', default=False, key="-SAVEVID-")],
        [sg.Image(filename='', key='-IM1-'), sg.VSeparator(),
           sg.Image(filename='', key='-IM2-')],
          [sg.HSeparator()],
          [sg.Image(filename='', key='-IM3-'), sg.VSeparator(),
           sg.Image(filename='', key='-IM4-')]]

window = sg.Window("Live Camera Feed", layout, grab_anywhere=True)

agnostic_nms=False
augment=False
classes=[0]
conf_thres=0.4
config_deepsort='deep_sort/configs/deep_sort.yaml'
device=''
fourcc='MJPG'
img_size=640
iou_thres=0.5
output='inference/output'
save_txt=False
save_vid=False
weights='yolov5/weights/yolov5x.pt'

source1 = '0'

t, half, device, model, names, deepsort = load_info(config_deepsort, device, 
                                                             img_size, output, weights)

save_path = str(Path(output))

camList = ["1", "2", "3", "4"] 


out_imgs = save_path + '/images/'
out_vids = save_path + '/videos/'
out_txt = save_path + '/tracks/'


vid_path1, vid_writer1 = None, None

webcam1, dataset1, data1 = getData(source1, img_size)

fps = FPS().start()
while True:
    event, values = window.read(timeout=20)
        
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    
    dataiter1 = next(data1)
    frame_idx1, path1, img1, im0s1, vs1 = dataiter1[0], dataset1.sources, dataiter1[1][1], dataset1.imgs, dataset1.cap
    frame1 = im0s1[0]
          
        
    if values["-TRACK-"]:
        frame1, outputs1 = detect(agnostic_nms, augment, classes, conf_thres, device, 
           iou_thres, half, model, webcam1, names, deepsort, path1, img1, im0s1)

        if values["-PRINT-"]:
            for cam in camList:
                txt_path = out_txt + cam + str(t)
                if not Path(txt_path).exists():
                    Path(txt_path).mkdir()
                    
            save_txt = True
            write_tracks(frame_idx1, save_txt, outputs1, txt_path)
            
        if values["-SAVEIMG-"]:
            for cam in camList:
                save_imgs_path = out_imgs + cam + str(t)
                if not Path(save_imgs_path).exists():
                    Path(save_imgs_path).mkdir()
            save_images(frame_idx1, outputs1, save_imgs_path, frame1)
            
    if values["-SAVEVID-"]:
        for cam in camList:
            save_vids_path = out_vids + cam + str(t)
            if not Path(save_vids_path).exists():
                Path(save_vids_path).mkdir()
        vid_path1, vid_writer1 = save_videos(vid_path1, save_path, vid_writer1, vs1, save_vids_path, fourcc, frame1)
    
    frame1 = imutils.resize(frame1, width=img_size)
    
    win_text = 'Camera: '+cam_id
    imgbytes1 = cv2.imencode(".png", frame1)[1].tobytes()
    
    window["-IM1-"].update(data=imgbytes1)
    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    
cv2.destroyAllWindows()
vs1.release()




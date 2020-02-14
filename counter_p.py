#!/usr/bin/pytnon3
# base ob deep_sort body path tracking project
from __future__ import division
import torch
import numpy as np
import cv2
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
from deep_sort import nn_matching
from deep_sort import preprocessing
import generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from shapely.geometry import LineString, Point
from utils.models import Darknet
from utils.parse_config import *
from utils.utils import non_max_suppression
import math
import warnings
warnings.filterwarnings('ignore')

model_name = "yolov3" 
# path to input video
video_name = "video/67.avi"
# border line
border_line = [(0, 450), (1600, 450)]
# save or don't the output video to the disk
writeVideo_flag = False
root_dir = os.getcwd()
# detector section
model_def = "Model_data/yolov3.cfg"
weights_path = "Model_data/yolov3.weights"
# wget -c https://pjreddie.com/media/files/yolov3.weigh
conf_thres = 0.5
nms_thres = 0.4
img_size = 416
body_res = (256, 128)
# end detector
# embedding pose
threshold = 0.5 # 07
nms_max_overlap = 0.9 # 1
max_cosine_distance = 0.2 # 03
# end emedding
def drawBorderLine(a, b):
    length = 40
    vX0 = b[0] - a[0]; vY0 = b[1] - a[1]
    mag = math.sqrt(vX0*vX0 + vY0*vY0)
    vX = vX0 / mag; vY = vY0 / mag
    temp = vX; vX = -vY; vY = temp
    z0 = (int(a[0]+vX0/2), int(a[1]+vY0/2))
    z1 = (int(a[0]+vX0/2 - vX * length), int(a[1] +vY0/2- vY * length))
    cv2.line(frame, a, b, (255, 255, 0), 2)
    cv2.arrowedLine(frame, z0, z1, (0, 255, 0), 2)
    cv2.putText(frame, "Out", z1, 0, 1, (0, 255, 0), 1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = Darknet(model_def, img_size=img_size).to(device)
    detector.load_darknet_weights(weights_path)
    tr_img = True
    nn_budget = None
    path_track = 20 # how many frames in path are saves
    cnt_people_in = {}
    cnt_people_out = {}
    skip_counter = 10
    counter = 0
    
    # human ids model
    model_filename = 'Model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    cap = cv2.VideoCapture(video_name)
    show_fh_fw = (800, 600)
    max_hum_w = show_fh_fw[0]/2
    ratio_h_w = (show_fh_fw[0]/img_size, show_fh_fw[1]/img_size)
    border_line = [(int(border_line[0][0]/ratio_h_w[0]),int(border_line[0][1]/ratio_h_w[1])),(int(border_line[1][0]/ratio_h_w[0]),int(border_line[1][1]/ratio_h_w[1]))]
    border_line_str = LineString(border_line)
    border_line_a = (border_line[1][0] - border_line[0][0])
    border_line_b = (border_line[1][1] - border_line[0][1])
    # ratio_h_w = (1,1)
    out = None
    if writeVideo_flag:
        outFile = root_dir+'/video/' + model_name + '_' + video_name
        print("Save out video to file " + outFile)
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)
    while True:
        r, frame = cap.read()
        if (not r):
            print("skip frame ", skip_counter)
            skip_counter -= 1
            if (skip_counter > 0): continue
            else: break
        start = time.time()
        counter += 1
        frame = cv2.resize(frame, show_fh_fw)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = frame.shape[:2]       
        input_imgs = cv2.resize(frame, (img_size, img_size))
        input_imgs = (np.array([input_imgs])/255.0).astype(np.float16).transpose(0, 3, 1, 2)
        input_imgs = torch.from_numpy(input_imgs).float().to(device)
        with torch.no_grad():
            obj_detec = detector(input_imgs)
            # print(obj_detec.shape)
            obj_detec = non_max_suppression(obj_detec, conf_thres, nms_thres)
            # print(obj_detec)
        # print(boxes)
        boxs = []
        confs = []
        for item in obj_detec:
            if item is not None:
                i = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in item: #classes[int(cls_pred)]
                    if((cls_pred == 0) and (y2-y1 < max_hum_w)):
                            boxs.append([int(y1*ratio_h_w[1]), int(x1*ratio_h_w[0]), int(y2*ratio_h_w[1]), int(x2*ratio_h_w[0])])
                            # print(conf, cls_conf)
                            # confs.append(float(conf))
                            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            # person_photo = frame[y1:y2, x1:x2]
        # print(confs)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs, confs, features)]
        # print(detections)
        boxes = np.array([d.tlwh for d in detections]) # w and h replace by x2 y2
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print(detections)
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if(not track.is_confirmed() or track.time_since_update > 1):
                # if(track.time_since_update > life_frame_limit): track.state = 3 # if missed to long than delete id
                continue 
            bbox = track.to_tlwh()
            x1y1 = (int(bbox[1]+(bbox[3] - bbox[1])/2), int(bbox[0]+(bbox[2] - bbox[0])/2))
            clr = (255, 255, 0) # default color
            track_name = str(track.track_id) # default name
            if(hasattr(track, 'xy')):
                # detect direction
                track_line = LineString([track.xy[0], x1y1])
                if(track_line.intersection(border_line_str)):
                    # print("intersection!!", track_name)
                    if(not hasattr(track, 'calculated')):
                        if(border_line_a * (x1y1[1] - border_line[0][1]) -  border_line_b * (x1y1[0] - border_line[0][0])) > 0:
                            cnt_people_in[track.track_id] = 0
                            track.calculated = "in_" + str(len(cnt_people_in)) + "_"
                            track.color = (52, 235, 240)
                        else: # 
                            cnt_people_out[track.track_id] = 0
                            track.calculated = "out_" + str(len(cnt_people_out)) + "_"
                            track.color = (0, 255, 0)
                        track.cross_cnt = path_track
                    clr = track.color
                # else:
                    
                if(hasattr(track, 'calculated')):
                    clr = track.color
                    track_name = track.calculated  + track_name
                    track.cross_cnt -= 1
                    if(track.cross_cnt < 1): track.state = 3 # delete from track list
                track.xy = np.append(track.xy, [x1y1], axis=0)
                track.xy = track.xy[-path_track:]
                # cv2.arrowedLine(frame,(track.x1[0], track.y1[0]),(x1, y1),(0,255,0),4)
                cv2.polylines(frame, [track.xy], False, clr, 3)
            else: track.xy = np.array([x1y1])
            cv2.circle(frame, x1y1, 9, clr, -1)
            cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 2)
            # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame, track_name, x1y1, 0, 5e-3 * 200, clr, 1)
        drawBorderLine(border_line[0], border_line[1])
        cv2.rectangle(frame, (0,10), (800, 100), (20, 20, 20), -1)
        cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame:"+str(counter), (10, 40), 0, 1, (255, 255, 0), 1)
        cv2.putText(frame, "People in: "+str(len(cnt_people_in)), (10, 80), 0, 1, (52, 235, 240), 1)
        cv2.putText(frame, " out: "+str(len(cnt_people_out)), (240, 80), 0, 1, (0, 255, 0), 1)
        # print("end frame")
        cv2.imshow("preview", frame)
        if writeVideo_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
    cap.release()
    if writeVideo_flag: out.release()
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))

# 
# based on deep sort body path tracking project
# TODO
# add intersection rule - follow to prev direction
# add rule for forbidden too fast change locations - update ids
# add tracking several videos
# check if calculated a right direction

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
from utils.utils import non_max_suppression
from utils.osnet import *
import math
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import torch.nn.functional as F

model_name = "yolov3" 
# path to input video
video_name = "39.avi"
video_path = "video/"
# border line
border_line = [(0, 400), (1200, 400)]
# save or don't the output video to the disk
writeVideo_flag = False
root_dir = os.getcwd()
# detector section
model_def = "utils/yolov3.cfg"
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
    # detector.eval()  # Set in evaluation mode
    # classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    tr_img = True
    nn_budget = None
    path_track = 20 # how many frames in path are saves
    cnt_people_in = {}
    cnt_people_out = {}
    skip_counter = 10
    counter = 0
    conf_thres = torch.tensor(conf_thres).to(device)
    nms_thres = torch.tensor(nms_thres).to(device)
    t_zero = torch.tensor(0).to(device)
    # print(conf_thres)
    # human ids model

    # reid_model_def = "osnet_x0_25"
    # reid_weights_path = "Model_data/osnet_x0_25_imagenet.pth"
    reid_size = (256,128)
    # encoder = osnet_x0_25().to(device)
    encoder = osnet_x0_25(100).to(device)
    
    # # wget https://github.com/Qidian213/deep_sort_yolov3/raw/master/model_data/mars-small128.pb
    # encoder = gdet.create_box_encoder('Model_data/mars-small128.pb', batch_size=64)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    cap = cv2.VideoCapture(video_path+video_name)
    # show_fh_fw = (800, 600)
    img_size_start = (1600,1200)
    show_fh_fw = (img_size, img_size)
    max_hum_w = show_fh_fw[0]/2
    ratio_h_w = (show_fh_fw[0]/img_size, show_fh_fw[1]/img_size)
    max_hum_w = torch.tensor(max_hum_w).to(device)
    start_ratio_h_w = (img_size_start[0]/show_fh_fw[0],img_size_start[1]/(show_fh_fw[1]-104))
    border_line = [(int(border_line[0][0]/start_ratio_h_w[0]),int(border_line[0][1]/start_ratio_h_w[1])),(int(border_line[1][0]/start_ratio_h_w[0]),int(border_line[1][1]/start_ratio_h_w[1]))]
    
    border_line_str = LineString(border_line)
    border_line_a = (border_line[1][0] - border_line[0][0])
    border_line_b = (border_line[1][1] - border_line[0][1])
    # ratio_h_w = (1,1)
    out = None
    if writeVideo_flag:
        outFile = root_dir+'/video/' + model_name + '_' + video_name
        print("Save out video to file " + outFile)
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'XVID'), 10, show_fh_fw)
        # out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'MP4V'), 10, show_fh_fw)
    while True:
        r, frame = cap.read()
        if (not r):
            print("skip frame ", skip_counter)
            skip_counter -= 1
            if (skip_counter > 0): continue
            else: break
        start = time.time()
        counter += 1
        #======
        frame = cv2.copyMakeBorder( frame, 0, 400, 0, 0, cv2.BORDER_CONSTANT)
        frame = cv2.resize(frame,(img_size, img_size))
        # frame_cuda = transforms.ToTensor()(frame).to(device)
        # input_imgs = frame_cuda.unsqueeze(0)
        frame_cuda = transforms.ToTensor()(frame).to(device).unsqueeze(0)
        with torch.no_grad():
            obj_detec = detector(frame_cuda)
            # print(obj_detec.shape)
            obj_detec = non_max_suppression(obj_detec, conf_thres, nms_thres)
            # print(obj_detec)
        # print(boxes)
        # boxs = torch.tensor([]).to(device)
        boxs = []
        persons = []
        for item in obj_detec:
            if item is not None:
                i = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in item: #classes[int(cls_pred)]
                    if((cls_pred == t_zero) and (y2-y1 < max_hum_w)):
                        c_loc = [int(y1), int(x1), int(y2), int(x2)]
                        boxs.append(c_loc)
                        # p_loc.append(c_loc)
                        person = frame_cuda[:, :, c_loc[0]:c_loc[2], c_loc[1]:c_loc[3]]
                        person = torch.nn.functional.interpolate(person, body_res)
                        # cv2.imwrite( "video/"+str(i)+"_ge.jpg", (person*255).squeeze(0).permute(1, 2, 0).numpy())
                        # cv2.imshow("preview2", (person*255).squeeze(0).permute(1, 2, 0).numpy())
                        persons.append(person)
        t_boxs_imgs = torch.cat(persons, 0)
        # print(t_boxs_imgs.shape)
        with torch.no_grad():
            outputs = encoder(t_boxs_imgs, True)
            # print("d1=",outputs.shape)
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        # if outputs.is_cuda: outputs = outputs.cpu()        
        # print("d1=", outputs.shape) # d1= torch.Size([8, 128])
        features = outputs.numpy()
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections]) # w and h replace by x2 y2
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
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
            cv2.circle(frame, x1y1, 5, clr, -1)
            cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 1)
            cv2.putText(frame, track_name, x1y1, 0, 0.4, clr, 1)
        # frame = input_imgs.squeeze(0).detach().numpy().transpose(1, 2, 0)
        drawBorderLine(border_line[0], border_line[1])
        # cv2.rectangle(frame, (0,10), (800, 100), (20, 20, 20), -1)
        cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame: "+str(counter), (10, 340), 0, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "People in: "+str(len(cnt_people_in)), (10, 360), 0, 0.4, (52, 235, 240), 1)
        cv2.putText(frame, " out: "+str(len(cnt_people_out)), (43, 376), 0, 0.4, (0, 255, 0), 1)
        # print("end frame")
        cv2.imshow("preview", frame)
        if writeVideo_flag:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
    cap.release()
    if writeVideo_flag: out.release()
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))

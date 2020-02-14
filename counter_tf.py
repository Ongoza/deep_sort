#!/usr/bin/pytnon3
# base ob deep_sort body path tracking project
# Tensorflow Object Detection Detector
import numpy as np
import tensorflow as tf
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

import warnings
warnings.filterwarnings('ignore')
# models for download  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# path to model local folder "Projects/Model_data"
# имя модели (папки) из папки Model_data
model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09' # 
# model_name ='faster_rcnn_resnet101_coco_2018_01_28'
# model_name = "rfcn_resnet101_coco_2018_01_28" 
# path to input video
video_name = "video/39.avi"
# border line
border_line = [(0, 400), (1000, 400)]
# save or don't the output video to the disk
writeVideo_flag = False


root_dir = os.getcwd()

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
            # with tf.compat.v1.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                        int(boxes[0, i, 1] * im_width),
                        int(boxes[0, i, 2] * im_height),
                        int(boxes[0, i, 3] * im_width))
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])
    
    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    # model_path = 'data/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    model_path = 'Model_data/'+model_name+'/frozen_inference_graph.pb'
    detector = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.3 # 07
    nms_max_overlap = 0.9 # 1
    max_cosine_distance = 0.3 # 03
    tr_img = True
    nn_budget = None
    # life_frame_limit = 20 # how many frames can skip id for tracking
    path_track = 20 # how many frames in path are saves
    cnt_people_in = {}
    cnt_people_out = {}
    skip_counter = 10
    counter = 0
    # border_line = [(580, 100), (340, 420)]
    # border_line = [(0, 400), (1000, 400)]
    border_line_str = LineString(border_line)
    model_filename = 'Model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    cap = cv2.VideoCapture(video_name)
    show_fh_fw = (800, 600)
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
        # frame = cv2.resize(frame, (fh_show, fw_show))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fh, fw = frame.shape[:2]
        # print("frame_size=", fh, fw)
        max_hum_w = fw/2
        all_boxs, scores, classes, num = detector.processFrame(frame)
        boxs = []
        for i in range(len(all_boxs)): # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                cur_hum_w = all_boxs[i][3] - all_boxs[i][1]
                if(max_hum_w > cur_hum_w): boxs.append(all_boxs[i])
                # cv2.rectangle(frame,(all_boxs[i][1],all_boxs[i][0]),(all_boxs[i][3],all_boxs[i][2]),(255,255,0),2)
        # People tracking
        
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections]) # !!! w and h replace by x2 y2
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
                deltaY = track.xy[0][1]-x1y1[1]
                track_line = LineString([track.xy[0], x1y1])
                if(track_line.intersection(border_line_str)):
                    if(not hasattr(track, 'calculated')):
                        if(deltaY > 0):
                            cnt_people_in[track.track_id] = 0
                            track.calculated = "in_" + str(len(cnt_people_in)) + "_"
                            track.color = (52, 235, 240)
                        else:
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
                cv2.polylines(frame, [track.xy], False, clr, 4)
            else: track.xy = np.array([x1y1])
            cv2.circle(frame, x1y1, 9, clr, -1)
            cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), clr, 2)
            # cv2.putText(frame, str(track.track_id),(int(bbox[1]), int(bbox[0])),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame, track_name, x1y1, 0, 5e-3 * 200, clr, 2)
        cv2.line(frame, border_line[0], border_line[1], (255, 0, 0), 10)
        # if(tr_img):
        #     cv2.imwrite( "video/Image.jpg", frame )
        #     tr_img = False
        cv2.rectangle(frame, (0,10), (1200, 100), (20, 20, 20), -1)
        cv2.putText(frame, "FPS: "+str(round(1./(time.time()-start), 2))+" frame:"+str(counter)+" Model:"+model_name, (10, 40), 0, 1, (255, 255, 0), 2)
        cv2.putText(frame, "People in: "+str(len(cnt_people_in)), (10, 80), 0, 1, (52, 235, 240), 2)
        cv2.putText(frame, " out: "+str(len(cnt_people_out)), (240, 80), 0, 1, (0, 255, 0), 2)
        frame = cv2.resize(frame, show_fh_fw)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("preview", frame)
        if writeVideo_flag: out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
    cap.release()
    if writeVideo_flag: out.release()
    cv2.destroyAllWindows()
    print("Model: "+model_name+". People in: "+str(len(cnt_people_in))+", out: "+str(len(cnt_people_out)))

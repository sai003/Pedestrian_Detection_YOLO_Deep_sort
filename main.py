import warnings
from collections import deque
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from timeit import time
from utils import create_box_encoder
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import natsort
from os import listdir
import cv2
from PIL import Image
import imutils
from deep_sort import preprocessing

INPUT_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/"
NMS_THRESHOLD=0.2
MIN_CONFIDENCE=0.2

np.random.seed(43)
MAX_COLORS = 200
color_list = np.random.randint(0, 255, size=(MAX_COLORS, 3),dtype="uint8")

paths_list = [deque(maxlen=30) for _ in range(9999)]

def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes

	# return the list of results
    return boxes, confidences, centroids

def start_detecting():
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.2
    counter = []
    find_objects = ['person']
    
    DEEP_SORT_MODEL_FILENAME = 'models/market1501.pb'
    encoder = create_box_encoder(DEEP_SORT_MODEL_FILENAME,batch_size=64)
    
    cosine_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(cosine_metric)
    
    all_jpg_files = listdir(INPUT_IMAGES_FOLDER)
    all_jpg_files = natsort.natsorted(all_jpg_files)
    
    labelsPath = "class.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    
    weights_path = "./models/yolov4.weights"
    config_path = "./models/yolov4.cfg"
    
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    '''
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    '''
    
    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    
    for jpg_file_name in all_jpg_files:
        image = cv2.imread(INPUT_IMAGES_FOLDER+jpg_file_name)
        
        image = imutils.resize(image, width=1200)
        boxs, confidences, centroid = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))
        
        features = encoder(image,boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        
        tracker.predict()
        tracker.update(detections)
        
        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in color_list[indexIDs[i] % len(color_list)]]
            #print(frame_index)
            #list_file.write(str(frame_index)+',')
            #list_file.write(str(track.track_id)+',')
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 2)
            #b0 = str(bbox[0])#.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            #b1 = str(bbox[1])#.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            #b2 = str(bbox[2]-bbox[0])#.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            #b3 = str(bbox[3]-bbox[1])

            #list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            #print(str(track.track_id))
            #list_file.write('\n')
            #list_file.write(str(track.track_id)+',')
            cv2.putText(image,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),1)
            
            cv2.putText(image, 'Person',(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),1)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]

            paths_list[track.track_id].append(center)

            thickness = 1
            #center point
            cv2.circle(image,  (center), 1, color, thickness)
            
            for j in range(1, len(paths_list[track.track_id])):
                if paths_list[track.track_id][j - 1] is None or paths_list[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image,(paths_list[track.track_id][j-1]), (paths_list[track.track_id][j]),(color),thickness)
        
        '''
        for i in range(len(boxs)):
            cv2.rectangle(image, (boxs[i][0],boxs[i][1]), (boxs[i][0]+boxs[i][2],boxs[i][1]+boxs[i][3]), (0, 255, 0), 2)
    
        
        #print(centroids)
        
        cv2.putText(image, F"Count: {len(boxs)}", (530, 30), font, 1, color, 2) 
        '''
        
        count = len(set(counter))
        cv2.putText(image, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(image, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        
        
        cv2.imshow("Detection",image)

        key = cv2.waitKey(30)
        if key == 27:
            break
        

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    
    start_detecting()
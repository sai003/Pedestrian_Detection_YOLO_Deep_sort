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

#INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-02/" # Perf not good
#MASK_IMAGES_FOLDER = "./step_images/0002/"
#SHOW_PERFORMANCE = 1

#INPUT_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/"
#MASK_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/" # just adding this variable so that code doesn't give error
#SHOW_PERFORMANCE = 0

INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-09/"
MASK_IMAGES_FOLDER = "./step_images/0009/"
SHOW_PERFORMANCE = 1

OUTPUT_IMAGES_FOLDER = "./output/"

NMS_THRESHOLD=0.2
MIN_CONFIDENCE=0.2

np.random.seed(43)
MAX_COLORS = 200
color_list = np.random.randint(0, 255, size=(MAX_COLORS, 3),dtype="uint8")

paths_list = [deque(maxlen=30) for _ in range(9999)]

FRAMES_TO_HIGHLIGHT = 50

pause = False
drawing = False
ix = 0
iy = 0
ox = 0
oy = 0
bbox_list_per_frame = {}


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
    
    individuals = set()
    new_individuals = dict()
    person_frame_tracker = dict()
    last_jpg_file = ''
    
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
    
    if SHOW_PERFORMANCE == 1:
        mask_person_ids = list()
        sum_calculated_frame_count = 0.0
        sum_actual_frame_count = 0.0
    
    for jpg_file_name in all_jpg_files:
        last_jpg_file = jpg_file_name
        
        image = cv2.imread(INPUT_IMAGES_FOLDER+jpg_file_name)
        
        if SHOW_PERFORMANCE == 1:
            mask_file = jpg_file_name.split('.')[0]+'.png'
            mask_image = cv2.imread(MASK_IMAGES_FOLDER+mask_file) 
            
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


        print("Processing file:", jpg_file_name)
        if SHOW_PERFORMANCE == 1: 
            print("\nMask file:", mask_file)
        
        bbox_list_per_frame[jpg_file_name] = list()
        for det in detections:
            bbox = det.to_tlbr()
            #cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in color_list[indexIDs[i] % len(color_list)]]
            bbox_list_per_frame[jpg_file_name].append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
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
            cv2.putText(image, 'Person:',(int(bbox[0]), int(bbox[1] -15)),0, 5e-3 * 150, (color),1)
            cv2.putText(image,str(track.track_id),(int(bbox[0]+90), int(bbox[1] -15)),0, 5e-3 * 150, (color),1)
            
            """For Task 3.3"""
            if track.track_id not in individuals:
                individuals.add(track.track_id)
                new_individuals[track.track_id] = FRAMES_TO_HIGHLIGHT
                    
            if track.track_id not in person_frame_tracker:
                person_frame_tracker[track.track_id] = list()
            person_frame_tracker[track.track_id].append((jpg_file_name, (int(bbox[0]),int(bbox[1]))))
                    
            if track.track_id in new_individuals and new_individuals[track.track_id] > 0:
                cv2.arrowedLine(image, (int(bbox[0]-15), int(bbox[1]-15)), (int(bbox[0]-2), int(bbox[1]-2)), (255, 0, 0), 3, 8, 0, 0.35)
                new_individuals[track.track_id] -= 1
            # cv.arrowedLine(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, tipLength=0.1)
            """Task 3.3"""
            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
    
            paths_list[track.track_id].append(center)
    
            thickness = 1
            #center point
            cv2.circle(image, (center), 1, color, thickness)
                
            for j in range(1, len(paths_list[track.track_id])):
                if paths_list[track.track_id][j - 1] is None or paths_list[track.track_id][j] is None:
                   continue
                   thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image,(paths_list[track.track_id][j-1]), (paths_list[track.track_id][j]),(color),thickness)
               
            '''
            if drawing == True:
                cv2.rectangle(image,pt1=(ix,iy),pt2=(ox,oy),color=(255,255,255),thickness=2)
                if ix < bbox[0] < ox and ix < bbox[2] < ox and iy < bbox[1] < oy and iy < bbox[3] < oy :
                    inbox_count += 1
            '''
                
        
        '''
        for i in range(len(boxs)):
            cv2.rectangle(image, (boxs[i][0],boxs[i][1]), (boxs[i][0]+boxs[i][2],boxs[i][1]+boxs[i][3]), (0, 255, 0), 2)
        
            
        #print(centroids)
        
        cv2.putText(image, F"Count: {len(boxs)}", (530, 30), font, 1, color, 2) 
        '''
            
        count = len(set(counter))
        cv2.putText(image, "Unique Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(image, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        '''
        if drawing:
            cv2.putText(image, "Pedestrians in Given area: "+str(inbox_count),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),2)
        '''
        if SHOW_PERFORMANCE == 1:
            print("Calculated frame count:", i)
            actual_frame_count = len(np.unique(mask_image[:, :, 0])) - 1 # subtracting 1 due to background label ID
            print("Actual frame count:", actual_frame_count)        
            
            sum_calculated_frame_count += i
            sum_actual_frame_count += actual_frame_count
            
            print("Calculated total count:", count)
            mask_person_ids.extend(np.unique(mask_image[:, :, 0]))
            mask_person_ids = list(set(mask_person_ids))
            print("Actual total count:",len(mask_person_ids)-1) # subtracting 1 due to background label ID
            
        cv2.imwrite(OUTPUT_IMAGES_FOLDER+jpg_file_name, image)
        
    if SHOW_PERFORMANCE == 1:
        print()
        print("Final Performance Metrics:")
        print("Per frame pedestrian detection efficiency = {:.2f}%".format(100*sum_calculated_frame_count/sum_actual_frame_count))
        print("Calculated total count:", count)
        print("Actual total count:",len(mask_person_ids)-1) # subtracting 1 due to background label ID
        print("Calculated to Actual Total Count Ratio =", float(count)/float(len(mask_person_ids)-1))
        
        
    all_jpg_files = listdir(OUTPUT_IMAGES_FOLDER)
    all_jpg_files = natsort.natsorted(all_jpg_files)
    to_delete = list()
    
    for p_ids in person_frame_tracker:
        if(len(person_frame_tracker[p_ids]) <= FRAMES_TO_HIGHLIGHT):
            to_delete.append(p_ids)
        else:
            person_frame_tracker[p_ids] = person_frame_tracker[p_ids][-FRAMES_TO_HIGHLIGHT:]
            
    for p_ids in to_delete:
        del person_frame_tracker[p_ids]
        
    
    to_delete = list()
    for p_ids in person_frame_tracker:
        if person_frame_tracker[p_ids][-1][0] == last_jpg_file: to_delete.append(p_ids)
    for p_ids in to_delete:
        del person_frame_tracker[p_ids]
    
    #print("Final length:", len(person_frame_tracker))
    ################ CHECK ONLY CONTINUOUS LAST FRAMES ################
    
    frame_coordinates = dict()
    for p_ids in person_frame_tracker:
        for file, coord in person_frame_tracker[p_ids]:
            if file not in frame_coordinates:
                frame_coordinates[file] = list()
            frame_coordinates[file].append(coord)
            
    for jpg_file_name in all_jpg_files:
        image = cv2.imread(OUTPUT_IMAGES_FOLDER+jpg_file_name)
        
        if jpg_file_name in frame_coordinates:
            for coord in frame_coordinates[jpg_file_name]:
                cv2.arrowedLine(image, (coord[0]-15, coord[1]-15), (coord[0]-2, coord[1]-2), (0, 0, 255), 3, 8, 0, 0.35)
                
        inbox_count = 0
                
        def draw(event,x,y,flags,params):
            global drawing,ix,iy,ox,oy,pause
            if pause == True:
                if(event==1):
                    ix = x
                    iy = y
                '''
                if(event==0):
                    if(drawing==True):
                        cv2.rectangle(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=1)
                '''
                if(event==4):
                    drawing = True
                    ox, oy = x,y
                    cv2.rectangle(image,pt1=(ix,iy),pt2=(ox,oy),color=(255,255,255),thickness=2)
                
        cv2.namedWindow("Detection")
        cv2.setMouseCallback("Detection",draw)
        
        
        key = cv2.waitKey(30)
        
        if key & 0xFF == ord('p'):
            global pause,drawing
            pause = True
            drawing = False
        
        if drawing == True:
            cv2.rectangle(image,pt1=(ix,iy),pt2=(ox,oy),color=(255,255,255),thickness=2)
            for bbox in bbox_list_per_frame[jpg_file_name]:
                if ix < bbox[0] < ox and ix < bbox[2] < ox and iy < bbox[1] < oy and iy < bbox[3] < oy :
                    inbox_count += 1
            cv2.putText(image, "Pedestrians in Given area: "+str(inbox_count),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),2)
        
        cv2.imshow("Detection",image)
        while(pause):
            cv2.imshow("Detection",image)
            key = cv2.waitKey(30)
            if key & 0xFF == ord('p'):
                pause = not pause
            
        if key & 0xFF == ord('q'):
                break
                

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    
    start_detecting()
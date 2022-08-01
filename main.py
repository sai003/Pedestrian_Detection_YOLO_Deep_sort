import cv2
import imutils
import warnings
import natsort
import numpy as np
from os import listdir

from collections import deque
from utils import create_box_encoder
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection


#INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-02/" # Perf not good
#MASK_IMAGES_FOLDER = "./step_images/0002/"
#SHOW_PERFORMANCE = 1

INPUT_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/"
MASK_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/" # just adding this variable so that code doesn't give error
SHOW_PERFORMANCE = 0

#INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-09/"
#MASK_IMAGES_FOLDER = "./step_images/0009/"
#SHOW_PERFORMANCE = 1

OUTPUT_IMAGES_FOLDER = "./output/"

NMS_THRESHOLD=0.2
MIN_CONFIDENCE=0.2
COLOR = {
	1: (0, 255, 0),
	2: (255, 0, 0),
	3: (0, 255, 255),
	4: (255, 255, 0),
	5: (0, 0, 255),
	6: (255, 0, 255),
	7: (150, 150, 0),
	8: (150, 0, 150),
}

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
    
    return boxes, confidences, centroids


#calculate neared person
def nearToPerson(person1, person2):
	yRatio = 1.5	#y value ratio
	distThres = 0.6 #threshold of distance ratio
	
	distanceX = abs((person1[1][0] + person1[1][2]) / 2 - (person2[1][0] + person2[1][2]) / 2)
	distanceY = abs((person1[1][1] + person1[1][3]) / 2 - (person2[1][1] + person2[1][3]) / 2)
	distance = np.sqrt(distanceX*distanceX + distanceY*distanceY*yRatio)
	height = min(person1[1][3]-person1[1][1], person2[1][3]-person2[1][1])
	
	if distance / height < distThres:
		return True

	return False

def classify_groups(results):
	processed_numbers = []
	group_array = []
	n_person_cnt = len(results)

	nearMatrix = np.zeros(n_person_cnt*n_person_cnt)
	nearMatrix = nearMatrix.reshape(n_person_cnt, n_person_cnt)

	for idx1 in range(n_person_cnt):
		res1 = results[idx1]
		nearMatrix[idx1, idx1] = 1
		for idx2 in range(idx1+1, n_person_cnt):
			res2 = results[idx2]
			if nearToPerson(res1, res2):
				nearMatrix[idx1, idx2] = 1
				nearMatrix[idx2, idx1] = 1

				for idx in range(n_person_cnt):
					if nearMatrix[idx1, idx] == 1:
						nearMatrix[idx2, idx] = 1
						nearMatrix[idx, idx2] = 1

	for idx in range(n_person_cnt):
		if idx not in processed_numbers:
			processed_numbers.append(idx)
			new_group = []
			new_group.append(idx)

			for idx2 in range(n_person_cnt):
				if nearMatrix[idx, idx2]:
					if idx2 not in processed_numbers:
						processed_numbers.append(idx2)
						new_group.append(idx2)
			
			group_array.append(new_group)

	return nearMatrix


def start_detecting():
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.2
    counter = []
    
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
    
    weights_path = "./models/yolov4-tiny.weights"
    config_path = "./models/yolov4-tiny.cfg"
    #weights_path = "./models/yolov4.weights"
    #config_path = "./models/yolov4.cfg"
    
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    '''
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    '''
    
    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 120)
    color = (0, 0, 255)
    
    if SHOW_PERFORMANCE == 1:
        mask_person_ids = list()
        sum_calculated_frame_count = 0.0
        sum_actual_frame_count = 0.0
    
    for jpg_file_name in all_jpg_files:
        print("Processing file:", jpg_file_name)
        last_jpg_file = jpg_file_name
        
        image = cv2.imread(INPUT_IMAGES_FOLDER+jpg_file_name)
        
        if SHOW_PERFORMANCE == 1:
            mask_file = jpg_file_name.split('.')[0]+'.png'
            mask_image = cv2.imread(MASK_IMAGES_FOLDER+mask_file) 
            
        image = imutils.resize(image, width=1200)
        boxs, confidences, centroid = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
            
        features = encoder(image,boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
    
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        indices.sort()
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)
            
        i = int(0)
        indexIDs = []
        boxes = []
        results = []

        
        if SHOW_PERFORMANCE == 1: 
            print("\nMask file:", mask_file)
        
        bbox_list_per_frame[jpg_file_name] = list()
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in color_list[indexIDs[i] % len(color_list)]]
            bbox_list_per_frame[jpg_file_name].append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            
            results.append((1.0, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) ))
            
            """For Task 3.3"""
            if track.track_id not in individuals:
                individuals.add(track.track_id)
                new_individuals[track.track_id] = FRAMES_TO_HIGHLIGHT
                    
            if track.track_id not in person_frame_tracker:
                person_frame_tracker[track.track_id] = list()
            person_frame_tracker[track.track_id].append((jpg_file_name, (int(bbox[0]),int(bbox[1]))))
                    
            if track.track_id in new_individuals and new_individuals[track.track_id] > 0:
                cv2.arrowedLine(image, (int(bbox[0]-15), int(bbox[1]-15)), (int(bbox[0]-2), int(bbox[1]-2)), (255, 255, 255), 3, 8, 0, 0.35)
                new_individuals[track.track_id] -= 1
            """Task 3.3"""
            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
    
            paths_list[track.track_id].append(center)
    
            thickness = 2
            #center point
            cv2.circle(image, (center), 1, color, thickness)
                
            for j in range(1, len(paths_list[track.track_id])):
                if paths_list[track.track_id][j - 1] is None or paths_list[track.track_id][j] is None:
                   continue
                   thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image,(paths_list[track.track_id][j-1]), (paths_list[track.track_id][j]),(color),thickness)
               
        # getting group counts
        nearMatrix = classify_groups(results)
        alone_person = 0
        for idx in range(len(results)):
            res = results[idx]
            # if np.sum(nearMatrix[idx, :]) > 1:
            category = np.sum(nearMatrix[idx, :])
            cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), COLOR[int(category)], 2)
            if category > 1:
                cv2.putText(image, F"Group: {int(category)}", (res[1][0],res[1][1]-5), font, 0.5, COLOR[int(category)], 2)
            else:
                cv2.putText(image, "Alone", (res[1][0],res[1][1]-5), font, 0.5, COLOR[int(category)], 2)
                alone_person += 1
        
        count = len(set(counter))  
        cv2.putText(image, F"Current Count: {len(results)}", (530, 30), font, 0.7, text_color, 2)
        cv2.putText(image, F"Total Count: {count}", (530, 50), font, 0.7, text_color, 2)
        cv2.putText(image, F"Alone: {alone_person}", (530, 70), font, 0.7, text_color, 2)
        cv2.putText(image, F"In Groups: {len(results)-alone_person}", (530, 90), font, 0.7, text_color, 2)
    
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
    
    frame_coordinates = dict()
    for p_ids in person_frame_tracker:
        for file, coord in person_frame_tracker[p_ids]:
            if file not in frame_coordinates:
                frame_coordinates[file] = list()
            frame_coordinates[file].append(coord)
            
    while True:
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
                cv2.putText(image, "Count in Area: "+str(inbox_count), (530, 110), font, 0.7, text_color, 2)
            
            cv2.imshow("Detection",image)
            while(pause):
                cv2.imshow("Detection",image)
                key = cv2.waitKey(30)
                if key & 0xFF == ord('p'):
                    pause = not pause
                
            if key & 0xFF == ord('q'):
                    break
                
        key = cv2.waitKey(3000)
                    

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    start_detecting()
    
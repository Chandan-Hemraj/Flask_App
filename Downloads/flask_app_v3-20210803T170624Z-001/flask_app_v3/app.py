# *******************************************       IMPORTING FILES         ************************************************************
import time
import warnings

import bcrypt
import cv2
import keras
import numpy as np
import six.moves.urllib as urllib
from flask import Flask, render_template, url_for, request, session, redirect, Response, stream_with_context, jsonify
from flask_pymongo import PyMongo
from tensorflow.keras import models


# *******************************************************************************************************************


app = Flask(__name__)


#  ************************  Connecting to MongoDB  *****************************************************
app.config['MONGO_DBNAME'] = 'mydatabase'
app.config['MONGO_URI'] = 'mongodb+srv://Nimo014:' + urllib.parse.quote_plus(
    "Nirav@1412") + '@cluster0.a4pqx.mongodb.net/mydatabase?retryWrites=true&w=majority'

# mongodb+srv://Nimo014:<password>@cluster0.a4pqx.mongodb.net/myFirstDatabase?retryWrites=true&w=majority

mongo = PyMongo(app)

#     ******************************************  START MODE 2 BLOCK ****************************************************************8
'''
leaf_classes = {0: 'HEALTHY', 1 : 'RUST', 2: 'SCAB'}
upper_h = 109
upper_s = 252
upper_v = 186
lower_h = 32
lower_s = 0
lower_v = 0
import warnings
warnings.filterwarnings('ignore')
print(cv2.__version__) # -> should be 4.5.1
import numpy as np
# matplotlib.use('TkAgg')
import time
from filterpy.kalman import KalmanFilter


# ----------------------------------------------
# leaf

def leaf_detection(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v))
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)
    dialation = mask.copy()
    canny_output = cv2.Canny(dialation, 100, 200)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def leaf_classifier(roi):
    test_image = cv2.resize(roi, (64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    result = leaf_model.predict(test_image)
    result = np.argmax(result)
    clas = leaf_classes[result]
    return clas


# -------------------------------------
# apple
def activation(img):
    count = 0
    for i in img:
        for j in i:
            if j == 255:
                count += 1
    return count


def image_processing(image):
    # img = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 30, 65])
    upper_red = np.array([179, 255, 255])

    lower_orange = np.array([0, 30, 0])
    upper_orange = np.array([17, 255, 255])

    lower_yellow = np.array([18, 30, 0])
    upper_yellow = np.array([32, 255, 255])

    lower_green = np.array([33, 30, 100])
    upper_green = np.array([50, 200, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_orange, upper_orange)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_mask = cv2.bitwise_or(mask1, mask2)

    ry_mask = cv2.bitwise_or(red_mask, yellow_mask)

    final_mask = cv2.bitwise_or(ry_mask, green_mask)

    red_count = activation(red_mask)
    yellow_count = activation(yellow_mask)
    green_count = activation(green_mask)
    total_count = activation(final_mask)

    red_percent = round((red_count / total_count) * 100, 3)
    yellow_percent = round((yellow_count / total_count) * 100, 3)
    green_percent = round((green_count / total_count) * 100, 3)

    return red_percent, yellow_percent, green_percent


# ------------------------------------------
# drawing boxes
WHITE = (255, 255, 255)
YELLOW = (66, 244, 238)
GREEN = (80, 220, 60)
LIGHT_CYAN = (255, 255, 224)
DARK_BLUE = (139, 0, 0)
GRAY = (128, 128, 128)


def label_object(image, apple, textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin, box_color=DARK_BLUE,
                 text_color=YELLOW):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness)
    pos = (xmid - textsize[0] // 2, ymid + textsize[1] // 2)
    cv2.putText(image, apple, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, thickness, cv2.LINE_AA)


# ----------------------------------------------
# Tracking

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


# ----------------------------------------------------------

# apple size

def extract_apple(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 30, 65])
    upper_red = np.array([179, 255, 255])

    lower_orange = np.array([0, 30, 0])
    upper_orange = np.array([17, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_orange, upper_orange)

    red_mask = cv2.bitwise_or(mask1, mask2)

    return red_mask

model_name = 'Pipeline-task_2/modelA1.h5'
leaf_model = keras.models.load_model(model_name)
#model_name2 = '/content/drive/MyDrive/Pipeline-task_2/model_E0.h5' #model_E0.h5 link
#model2 = keras.models.load_model(model_name2)
modelConfiguration_apple= "Pipeline-task_2/yolov4 model files/yolo_apples/yolov4-obj.cfg" #config file for yolo
modelWeights_apple = "Pipeline-task_2/yolov4 model files/yolo_apples/backup/yolov4-obj_last.weights" #weights link - inside the backup folder
modelConfiguration_leaf= "Pipeline-task_2/yolov4 model files/yolo_leaf/yolov4-obj.cfg" #config file for yolo
modelWeights_leaf= "Pipeline-task_2/yolov4 model files/yolo_leaf/backup/yolov4-obj_last.weights" #weights link - inside the backup folder
net_apple = cv2.dnn.readNetFromDarknet(modelConfiguration_apple, modelWeights_apple)
net_leaf = cv2.dnn.readNetFromDarknet(modelConfiguration_leaf, modelWeights_leaf)
net_apple.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_apple.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
net_leaf.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_leaf.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#model = models.load_model('/content/drive/MyDrive/Pipeline-task_2/modeldisease.h5')
model = models.load_model('Pipeline-task_2/modeldisease_acc84_size48_layer128_1024_1024_1024.h5')
classes=['apple']
cap = cv2.VideoCapture("Pipeline-task_2/Apple farm 3D video of kashmir.mp4")
#cap = cv2.VideoCapture("/content/drive/MyDrive/Pipeline-task_2/tree.mp4")
# cap = cv2.VideoCapture("/content/drive/MyDrive/Pipeline-task_2/Apple farm 3D video of kashmir.mp4") # video being used for testing ( can use yours if you want )
# cap = cv2.VideoCapture("/content/drive/MyDrive/Pipeline-task_2/tree.mp4") # video being used for testing ( can use yours if you want )

def detect(choice):
    re, frame = cap.read()
    height, width, layers = frame.shape
    framecount = 0
    leaf_count = {}
    leaf_c = 1
    apple_count = {}
    apple_c = 1
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    thickness = 3
    writer = cv2.VideoWriter("kashmir apple track1.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height))
    apple_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.2)
    leaf_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.2)
    fully_ripe = 0
    half_ripe = 0
    raw = 0
    half_raw = 0
    total_leaf = 0
    total_healthy = 0
    rust_count = 0
    scab_count = 0
    blotch = 0
    # normal=0
    rotten = 0
    scab = 0
    DPI = 96  # dpi of camera
    area_average = 0  # for overall average area of apples observed
    apple_count_area = 0
    # ---------------------------------------------------------------

    while True:
        re, img = cap.read()

        if not re:
            break
        # framecount+=1
        # print(framecount)
        if (framecount >= 0):
            start = time.time()
            framecount += 1
            if (framecount % 10 == 0):
                print("______________________________________________________")
                print("Frame ", framecount)
                print("______________________________________________________")
            start = time.time()
            # retr, img = cap.read()
            img1 = img.copy()

            # ---------------------------------------------------------------------------------------
            # #LEAF
            # total_leaf = 0
            # total_healthy = 0
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
            net_leaf.setInput(blob)
            output_layers_names = net_leaf.getUnconnectedOutLayersNames()
            layerOutputs = net_leaf.forward(output_layers_names)
            boxes = []
            confidences = []
            # font = cv2.FONT_HERSHEY_SIMPLE
            class_ids = []
            # print(time.time()-start)
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = abs(int(center_x - w / 2))
                        y = abs(int(center_y - h / 2))
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            if len(boxes) == 0:
                # cv2_imshow(img1)
                # cv2.waitKey(1)
                print("No leaves in frame :", framecount)
                # continue
            else:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indexes) != 0:
                    leaf_objects = []
                    for i in indexes.flatten():
                        x, y, w, h, score = [abs(a) for a in boxes[i]] + [confidences[i]]
                        leaf_objects.append([x, y, x + w, y + h, score])
                    # for i in indexes.flatten():
                    #   #print(i)
                    #   #print(boxes[i])
                    leaf_objects = np.array(leaf_objects)
                    leaf_bboxes = leaf_tracker.update(leaf_objects)

                    for b in leaf_bboxes:
                        x, y, w, h, id = b
                        xmin = int(abs(x))
                        ymin = int(abs(y))
                        xmax = int(abs(w))
                        ymax = int(abs(h))
                        xmid = int(round((xmin + xmax) / 2))

                        ymid = int(round((ymin + ymax) / 2))

                        if id not in leaf_count:
                            leaf_count[id] = leaf_c
                            leaf_c += 1
                            leaf = img[ymin:ymax, xmin:xmax]
                            total_leaf += 1
                            leaf_type = leaf_classifier(leaf)
                            if leaf_type.lower() == 'healthy':
                                total_healthy += 1
                            elif leaf_type.lower() == 'scab':
                                scab_count += 1
                            else:
                                rust_count += 1

                        textsize, _ = cv2.getTextSize(
                            str(int(leaf_count[id])), fontface, fontscale, thickness)
                        label_object(img1, str(int(leaf_count[id])), textsize, thickness, xmax, xmid, xmin, ymax, ymid,
                                     ymin, box_color=WHITE, text_color=GREEN)

                # cv2.rectangle(img1, (x, y), (x+w, y+h), color, 2)
                # cv2_imshow(img1)
                # writer.write(img1)
                # print("-----------------------------")
                # print("Leaf disease detection")
                # print("Number of total leaves detected: ",total_leaf)
                # print("Number of helthy leaves : ",total_healthy)
                # print("-----------------------------")
            if (framecount % 10 == 0):
                print("-----------------------------")
                print("Leaf disease detection")
                print("Number of total leaves detected: ", total_leaf)
                print("Number of helthy leaves : ", total_healthy)
                print("Number of scab (unhealthy) leaves : ", scab_count)
                print("Number of rust (unhealthy) leaves : ", rust_count)
            # cv2_imshow(img1)
            # writer.write(img1)
            # if framecount >= 10*20:
            #   break

            # #---------------------------------------------------------------------------------------
            #     #APPLE
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
            net_apple.setInput(blob)
            output_layers_names = net_apple.getUnconnectedOutLayersNames()
            layerOutputs = net_apple.forward(output_layers_names)
            boxes = []
            confidences = []
            class_ids = []
            # print(time.time()-start)
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = abs(int(center_x - w / 2))
                        y = abs(int(center_y - h / 2))
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # apple colour
            if len(boxes) == 0:
                # cv2_imshow(img1)
                # cv2.waitKey(1)
                print("No apples in frame :", framecount)
                continue
            else:
                red_perc_list = []
                # fully_ripe=0
                # half_ripe=0
                # half_raw=0
                # raw=0
                # blotch=0
                # #normal=0
                # rotten=0
                # scab=0
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                apple_objects = []
                for i in indexes.flatten():
                    x, y, w, h, score = [abs(a) for a in boxes[i]] + [confidences[i]]
                    apple_objects.append([x, y, x + w, y + h, score])
                # print("Time taken for detection: ",(time.time()-start))

                # start=time.time()
                apple_objects = np.array(apple_objects)
                apple_bboxes = apple_tracker.update(apple_objects)

                for b in apple_bboxes:
                    x, y, w, h, id = b
                    xmin = int(abs(x))
                    ymin = int(abs(y))
                    xmax = int(abs(w))
                    ymax = int(abs(h))
                    xmid = int(round((xmin + xmax) / 2))

                    ymid = int(round((ymin + ymax) / 2))

                    if id not in apple_count:
                        apple_count[id] = apple_c
                        apple_c += 1
                        apple = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                        # print(apple.shape)
                        apple1 = apple2 = apple.copy()
                        # apple disease detection classification
                        test_image = cv2.resize(apple, (48, 48))
                        # cv2_imshow(test_image)
                        test_image = np.expand_dims(test_image, axis=0)
                        prediction = np.argmax(model.predict([test_image]))
                        count = 0
                        # print(prediction[0])
                        # print(prediction)
                        # print(count)
                        if (prediction == 1):
                            # print("Blotch_Apple")
                            blotch += 1
                        elif (prediction == 2):
                            # print("Rot_Apple")
                            rotten += 1
                            # cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), WHITE, 4)
                            # cv2_imshow(img1)
                        elif (prediction == 3):
                            # print("Scab_Apple")
                            scab += 1
                        # -------------------------------------------------------------
                        # apple size calculation
                        apple3 = extract_apple(apple2)
                        # cv2_imshow(apple3)
                        apple_count_area += 1
                        area = (cv2.countNonZero(apple3) * 25.4) / (DPI * 10)
                        area_average += area
                        # print("apple area : ",area,"cm")
                        # -------------------------------------------------------------
                        # red percentage calculator
                        red_percent = image_processing(apple1)
                        red_perc_list.append(red_percent[0])
                        perc = red_percent[0]
                        if (perc > 90):
                            fully_ripe += 1
                            text = str(fully_ripe)

                        elif (perc > 70 and perc <= 90):
                            half_ripe += 1
                            text = str(half_ripe)

                        elif (perc > 50 and perc <= 70):
                            half_raw += 1
                            text = str(half_raw)

                        else:
                            raw += 1
                            text = str(raw)
                    # -------------------------------------------------------------
                    # print(red_percent[0])
                    # cv2_imshow(apple)

                    # cv2.rectangle(img1, (xmin, ymin), (xmax, ymax), DARK_BLUE, 2)
                    # cv2.putText(img1, str(count[id]), (xmin, ymin+20), cv2.FONT_HERSHEY_PLAIN, 1, YELLOW, 3)
                    textsize, _ = cv2.getTextSize(
                        str(int(apple_count[id])), fontface, fontscale, thickness)
                    label_object(img1, str(int(apple_count[id])), textsize, thickness, xmax, xmid, xmin, ymax, ymid, ymin)

                    color = (0, 255, 0)
                    cv2.putText(img1, f"full_ripe: {fully_ripe}", (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1,
                                cv2.LINE_AA)
                    cv2.putText(img1, f"half_ripe: {half_ripe}", (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1,
                                cv2.LINE_AA)
                    cv2.putText(img1, f"half_raw: {half_raw}", (20, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1,
                                cv2.LINE_AA)
                    cv2.putText(img1, f"raw: {raw}", (20, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1, cv2.LINE_AA)
                if (framecount % 10 == 0):
                    print("-----------------------------")
                    print("Ripeness of apples based on red colour")
                    print("-----------------------------")
                    # print("The redness of the apples in frame:",red_perc_list)
                    print("Number of fully ripe apples in given frame : ", fully_ripe)
                    print("Number of half ripe apples in given frame : ", half_ripe)
                    print("Number of half raw apples in given frame : ", half_raw)
                    print("Number of raw apples in given frame : ", raw)
                    print("-----------------------------")
                    print("Apple Disease classification Results:")
                    print("-----------------------------")
                    print("Number of blotch  apples in given frame : ", blotch)
                    print("Number of rotten apples in given frame : ", rotten)
                    print("Number of scab apples in given frame : ", scab)
                    print("-----------------------------")
                    print("Apple Area Results:")
                    print("-----------------------------")
                    print("Average area of ", apple_count_area, " apples observed is : ", (area_average / apple_count_area),
                          "cm^2")
                    print("-----------------------------")

                writer.write(img1)


                if framecount >= 10 * 30:
                    break

                result = {'framecount': framecount}

                total_apple = fully_ripe + half_ripe + half_raw + raw

                if choice == 1:
                    result['total_apple'] = total_apple
                    yield render_template('mode2_apple_count.html', result=result)

                if choice == 2:
                    result['fully_ripe'] = fully_ripe
                    result['half_ripe'] = half_ripe
                    result['half_raw'] = half_raw
                    result['raw'] = raw

                    yield render_template('mode2_yield.html', result=result)

                if choice == 3:
                    result['blotch'] = blotch
                    result['rotten'] = rotten
                    result['scab'] = scab

                    yield render_template('mode2_apple_disease.html', result=result)

                if choice == 4:
                    result['total_leaf'] = total_leaf
                    result['rust_count'] = rust_count
                    result['scab_count'] = scab_count
                    result['total_healthy'] = total_healthy

                    yield render_template('mode2_leaf_disease.html', result=result)
        else:
            continue
    cap.release()
    cv2.destroyAllWindows()
    writer.release()

@app.route('/Apple_Count')
def apple_count():
    return Response(stream_with_context(detect(1)))


@app.route('/Yield')
def Yield():
    return Response(stream_with_context(detect(2)))


@app.route('/apple_disease')
def apple_disease():
    return Response(stream_with_context(detect(3)))

@app.route('/leaf_disease')
def leaf_disease():
    return Response(stream_with_context(detect(4)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Windows for mode 2
@app.route('/mode2_apple_count')
def mode2_1():
    return render_template('mode2 window 1.html')


@app.route('/mode2_yield')
def mode2_2():
    return render_template('mode2 window 2.html')

@app.route('/mode2_apple_disease')
def mode2_3_1():
    return render_template('mode2 window 3_1.html')

@app.route('/mode2_leaf_disease')
def mode2_3_2():
    return render_template('mode2 window 3_2.html')

'''
#     ******************************************    END MODE 2 BLOCK    ****************************************************************


#     ******************************************  START MODE 1 BLOCK ****************************************************************

#     ******************************************  END MODE 1 BLOCK ****************************************************************


#      *********************************REGISTRATION and HOME PAGE FUNCTION ***********************************

@app.route('/chart')
def chart():
    return render_template('temp.html')
# Home Page
@app.route('/')
def home():
    if 'username' in session:
        user = session['username']
        return render_template('base.html', user=user)
    return render_template('iframe.html')


# Authenticate Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'name': request.form['username']})

        if login_user:
            if bcrypt.hashpw(request.form['password'].encode('utf-8'), login_user['password']) == login_user[
                'password']:
                session['username'] = request.form['username']

                return redirect(url_for('home'))

        error = 'Invalid username or password'
    return render_template('registration/login.html', error=error)


# Authenticate Register page
@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'name': request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password1'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name': request.form['username'],
                          'password': hashpass,
                          "email_id": request.form['email'],
                          "address": request.form['address'],
                          "license_key": request.form['license_key'],
                          })
            session['username'] = request.form['username']
            return redirect(url_for('home'))

        return 'That username already exists!'

    return render_template('registration/register.html')


#      *********************************END HOME PAGE BLOCK ***********************************



app.secret_key = 'mysecret'
app.run(debug=True)

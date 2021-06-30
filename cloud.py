from helpers import *
import cv2
import tensorflow as tf
import json
import sys
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Checking for incorrect usage
if len(sys.argv) != 2:
    print("Usage: python cloud.py path_to_image")
    exit(-1)
image_file = sys.argv[1]

# These are set to the default names from exported models, update as needed.
ENDPOINT = "Insert Here"
prediction_key = "Insert Here"
prediction_resource_id = "Insert Here"
project_id = "Insert Here"


prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

# Create tf graph and return list of labels
image = cv2.imread(image_file)
if image is None:
    print("Invalid Path")
    exit(-1)
# Create clone to visualize on
clone = image.copy()
pred = [[],[]]
# Extracting probabilities and box bounds
s_success, im_buf_arr = cv2.imencode(".jpg", image)
byte_im = im_buf_arr.tobytes()
results = predictor.detect_image(
    project_id, 'Insert iteration name here', byte_im)
temp = []
for prediction in results.predictions:
    pred[0].append([prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.left+prediction.bounding_box.width, prediction.bounding_box.top+prediction.bounding_box.height])
    pred[1].append(prediction.probability)
final = []
# Scaling percentage box bounds
h, w = image.shape[:2]
for i in pred[0]:
    for j in range(4):
        if j % 2 == 0:
            i[j] *= w
        else:
            i[j] *= h
        if i[j] < 0:
            i[j] = 0
coords = pred[0]
# Applying non-maxima suppression
ind = tf.image.non_max_suppression(pred[0], pred[1], len(coords), iou_threshold=0.5,
                                   score_threshold=float('-inf'), name=None)
# Only taking into account images with a high probability
for i in ind:
    j = pred[1][i]
    if j > 0.8:
        j = [j, pred[0][i], i]
        final.append(j)
coords_list = []
# Creating a final list of box bounds which only included those of selected boxes
for j in final:
    coords_list.append(list(coords[j[2]]))
# Order boxes according to their rows and columns
# Rows and columns of each box stored in loc
loc, coords_list = order_boxes(coords_list)
# Extracting all lettuce images from main image
images = []
for i in coords_list:
    images.append(image[int(i[1]):int(i[3]), int(i[0]):int(i[2])])
if len(images) == 0:
    print("No lettuces detected")
    exit(-1)
# Calculating green intensity of each leaf
green_intensity = []
for i in images:
    total = 0
    count = 0
    for j in i:
        for x in j:
            if int(x[1]) > 0.75*(int(x[0])+int(x[2])):
                total += x[1]
                count += 1
    if count > 0:
        avg = total/count
    else:
        avg = 0
    if avg > 220:
        avg = 220
    green_intensity.append(1-avg/220)
# Calculating the relative size of each lettuce
sizes = []
total = 0
for i in images:
    total += i.shape[0] * i.shape[1]
avg = total/len(images)
for i in images:
    area = i.shape[0] * i.shape[1]
    if area > 1.25*avg:
        sizes.append("Large")
    elif area < 0.75*avg:
        sizes.append("Small")
    else:
        sizes.append("Medium")
project_id = "Insert Here"
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
pred = []
for j in images:
    is_success, im_buf_arr = cv2.imencode(".jpg", j)
    byte_im = im_buf_arr.tobytes()
    results = predictor.classify_image(
        project_id, 'Insert Iteration Name Here', byte_im)
    predictions = [[], []]
    for prediction in results.predictions:
        if prediction.tag_name == 'Diseased' and prediction.probability < 0.65:
            predictions[0].append(prediction.probability)
            predictions[1].append('Mixed')
        else:
            predictions[0].append(prediction.probability)
            predictions[1].append(prediction.tag_name)
    # Take the most likely result into account
    prob = max(predictions[0])
    for i in range(len(predictions[0])):
        if predictions[0][i] == prob:
            ind = i
    label = predictions[1][ind]
    pred.append((label, prob))
# Resize to fit visualization on screen
if clone.shape[0] > 800 or clone.shape[1] > 800:
    scale = 800/max(clone.shape)
    clone = cv2.resize(clone, (int(clone.shape[1]*scale), int(clone.shape[0]*scale)))
scale = clone.shape[0]/image.shape[0]
for i in range(len(coords_list)):
    for j in range(len(coords_list[i])):
        coords_list[i][j] *= scale
output_info = []
# Output to json and visualize as rectangles
for i in range(len(coords_list)):
    output_info.append({
        "Coordinates": coords_list[i],
        "Object_Identification_Probability": float(final[i][0]),
        "Status": pred[i][0], "Classification_Probability": pred[i][1],
        "Row": loc[i][0],
        "Column": loc[i][1],
        "Green_Intensity": green_intensity[i],
        "Size": sizes[i]})
    if pred[i][0] == 'Healthy':
        color = (0, 255, 0)
    elif pred[i][0] == 'Mixed':
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    cv2.rectangle(clone, (int(coords_list[i][0]), int(coords_list[i][1])),
                  (int(coords_list[i][2]), int(coords_list[i][3])), color, 2)
    cv2.putText(clone, pred[i][0], (int(coords_list[i][0]), int(coords_list[i][1])+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(clone, str(loc[i][0]) + ":" + str(loc[i][1]),
                (int(coords_list[i][0]), int(coords_list[i][1]) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
with open("output_info.json", "w") as write_file:
    json.dump(output_info, write_file, indent=4)
cv2.imshow('Visual Representation', clone)
cv2.waitKey(0)
cv2.imwrite('Visual_Representation.png', clone)

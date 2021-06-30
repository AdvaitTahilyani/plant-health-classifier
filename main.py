from helpers import *
import cv2
import tensorflow as tf
import json
import sys

# Checking for incorrect usage
if len(sys.argv) != 2:
    print("Usage: python main.py path_to_image")
    exit(-1)
image_file = sys.argv[1]

# These are set to the default names from exported models, update as needed.
INPUT_TENSOR_NAME = 'image_tensor:0'
OUTPUT_TENSOR_NAMES = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']
filename = "Object_Identification_Model/model.pb"
labels_filename = "Object_Identification_Model/labels.txt"

# Create tf graph and return list of labels
labels = create_tf_graph(filename, labels_filename)
image = cv2.imread(image_file)
if image is None:
    print("Invalid Path")
    exit(-1)
image = resize_down_to_1600_max_dim(image)
# Create clone to visualize on
clone = image.copy()
# Necessary preprocessing
network_input_size = 320
augmented_image = cv2.resize(image, (network_input_size, network_input_size))
# Extracting probabilities and box bounds
pred = predict_from_graph(OUTPUT_TENSOR_NAMES, augmented_image, INPUT_TENSOR_NAME)
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
ind = tf.image.non_max_suppression(pred[0], pred[1], len(coords), iou_threshold=0.8,
                                   score_threshold=float('-inf'), name=None)
# Only taking into account images with a high probability
for i in ind:
    j = pred[1][i]
    if j > 0.6:
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
    avg = total/count
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
# These are set to the default names from exported models, update as needed.
output_layer = 'loss:0'
input_node = 'Placeholder:0'
filename = "Image_Classification_Model/model.pb"
labels_filename = "Image_Classification_Model/labels.txt"
# Load graph and extract labels
labels = create_tf_graph(filename, labels_filename)
pred = []
for j in images:
    # Necessary preprocessing
    img = resize_down_to_1600_max_dim(j)
    h, w = img.shape[:2]
    min_dim = min(w, h)
    max_square_image = crop_center(img, min_dim, min_dim)
    augmented_image = resize_to_256_square(max_square_image)
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]
    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
    predictions = predict_from_graph([output_layer], augmented_image, input_node)
    # Convert numpy array to list
    predictions = predictions.tolist()
    # Take the most likely result into account
    prob = max(predictions[0])
    for i in range(len(predictions[0])):
        if predictions[0][i] == prob:
            ind = i
    # If diseased probability is low, output as mixed
    if labels[ind] == "Diseased" and prob < 0.5:
        label = "Mixed"
    else:
        label = labels[ind]
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

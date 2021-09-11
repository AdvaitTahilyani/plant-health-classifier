import cv2
import tensorflow as tf


def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if h < 1600 and w < 1600:
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def resize_to_256_square(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def create_tf_graph(filename, labels_filename):
    labels = []
    # Load graph
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())
    return labels


def predict_from_graph(output_tensor_name, image, input_tensor_name):
    with tf.compat.v1.Session() as sess:
        try:
            # Check for multiple output tensor names
            if len(output_tensor_name) > 1:
                output_tensors = [sess.graph.get_tensor_by_name(n) for n in output_tensor_name]
            else:
                output_tensors = output_tensor_name[0]
            predictions = sess.run(output_tensors, {input_tensor_name: [image]})
        except KeyError:
            print("Couldn't find classification output layer.")
            exit(-1)
    return predictions


def order_boxes(coords_list, height_preference=-2, min_height_difference=0.2, max_x=0.6):
    # Initialize necessary variables
    sort_list = []
    loc = []
    z = -1
    row = 1
    col = 1
    for x in range(len(coords_list)):
        # Dummy values to be replaced
        j = [20000000000, 20000000000, 200000000000000, 200000000000]
        # Order boxes with preference given to height
        # Height preference controls how much priority is given to higher objects
        # Objects with x values greater than max_x get counted in different rows when compared to higher placed objects
        # Provided they have min_height difference
        # All these values are in percent form in respect to the average width/length of the two lettuce being compared
        # Adjust values to account for edge cases pertaining to application
        for i in coords_list:
            if ((i[0] <= j[0] and i[0] - j[0] < height_preference * (i[1] - j[1]) and (not (
                    j[1] < i[1] and (j[0] - i[0] > max_x * ((i[2] - i[0]) + (j[2] - j[0]))) and (
                    i[1] - j[1] > (min_height_difference * ((i[3] - i[1]) + (j[3] - j[1]))))))) or (i[1] <= j[1] and (
                    ((1 / height_preference) * (i[0] - j[0]) > i[1] - j[1]) or (
                    i[0] - j[0] > max_x * ((i[2] - i[0]) + (j[2] - j[0])) and (
                    j[1] - i[1] > (min_height_difference * ((j[3] - j[1]) + (i[3] - i[1])))))))) and i not in sort_list:
                j = i
        # Check if a new row has been formed
        if z >= 0 and ((sort_list[z][1] < j[1] and sort_list[z][0] > j[0]) or (j[1]-sort_list[z][1] > 1.25*((sort_list[z][3]-sort_list[z][1]) + (j[3]-j[1]))/2)):
            col = 1
            row += 1
        # Record the row and column of the lettuce
        loc.append((row, col))
        col += 1
        z += 1
        sort_list.append(j)
    return loc, sort_list

import tensorflow as tf
import pymysql.cursors
from tqdm import tqdm
import os

from object_detection.utils import dataset_util

label_map = ['loadedtrunk', 'emptytrunk', 'car', 'forklift', 'others']

height = 600  # Image height
width = 960  # Image width

image_directory = '/media/grayson/DATA/tianjin/tianjin_selected/JPEGImages'

flags = tf.app.flags
flags.DEFINE_string('output_path', '/media/grayson/DATA/tianjin/tianjin_tf_record/tianjin.record',
                    'Path to output TFRecord')
FLAGS = flags.FLAGS


def modify_illegal_coordinate(coordinate_value, coordinate_type):
    if coordinate_value < 0:
        return 0
    if coordinate_type == 'width':
        coordinate_value = width if coordinate_value > width else coordinate_value
    elif coordinate_type == 'height':
        coordinate_value = height if coordinate_value > height else coordinate_value
    return coordinate_value


def parse_bounding_boxes(bounding_boxes_str):
    bounding_boxes = {'xmins': [], 'xmaxs': [], 'ymins': [], 'ymaxs': [], 'classes_text': [], 'classes': []}
    if bounding_boxes_str == '':
        return bounding_boxes
    for bounding_box_str in bounding_boxes_str.split(','):
        if bounding_box_str.find('NaN') != -1:
            continue
        bounding_box_values = bounding_box_str.split('_')
        if len(bounding_box_values) != 5:
            continue
        xmin = modify_illegal_coordinate(float(bounding_box_values[1].partition('.')[0]), 'width')
        ymin = modify_illegal_coordinate(float(bounding_box_values[2].partition('.')[0]), 'height')
        xmax = modify_illegal_coordinate(float(bounding_box_values[3].partition('.')[0]), 'width')
        ymax = modify_illegal_coordinate(float(bounding_box_values[4].partition('.')[0]), 'height')
        if xmin >= xmax or ymin >= ymax:
            continue

        bounding_boxes['xmins'].append(xmin/width)
        bounding_boxes['xmaxs'].append(xmax/width)
        bounding_boxes['ymins'].append(ymin/height)
        bounding_boxes['ymaxs'].append(ymax/height)
        bounding_boxes['classes'].append(int(bounding_box_values[0]))
        bounding_boxes['classes_text'].append(label_map[int(bounding_box_values[0])])

    return bounding_boxes


def create_tf_record(name, bounding_boxes):
    # TODO: Populate the following variables from your example.
    filename = name + '.jpg'  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(os.path.join(image_directory, filename), 'rb') as fid:
        encoded_image_data = fid.read()  # Encoded image bytes
    image_format = b'jpg'  # b'jpeg' or b'png'

    xmins = bounding_boxes['xmins']  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = bounding_boxes['xmaxs']  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = bounding_boxes['ymins']  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = bounding_boxes['ymaxs']  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = bounding_boxes['classes_text']  # List of string class name of bounding box (1 per box)
    classes = bounding_boxes['classes']  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='detection',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM `t_image`"
            cursor.execute(sql)
            result = cursor.fetchall()

    finally:
        connection.close()

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO: Write code to read in your dataset to examples variable

    for row in tqdm(result):
        bounding_boxes = parse_bounding_boxes(str(row['i_lanes']))
        tf_example = create_tf_record(str(row['i_name']), bounding_boxes)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()

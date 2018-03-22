import pymysql
import numpy as np
import os
import cv2

import inference
import convert_image_to_tf_record

eval_min_ground_truth_scale = 0.1
min_score_thresh = 0.01
height = 600
width = 960
total_num = 0
found_num = 0

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='detection',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `t_image` WHERE `i_batch`='0026'"
        cursor.execute(sql)
        result = cursor.fetchall()

finally:
    connection.close()

sess, tensor_dict, image_tensor = inference.init_model()
sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims([[[0, 0, 0]]], 0)})


def calculate_iou(rect_a, rect_b):
    if rect_a['x'] > rect_b['x'] + rect_b['w']:
        return 0.
    if rect_a['y'] > rect_b['y'] + rect_b['h']:
        return 0.
    if rect_a['x'] + rect_a['w'] < rect_b['x']:
        return 0.
    if rect_a['y'] + rect_a['h'] < rect_b['y']:
        return 0.
    inter_col = min(rect_a['x'] + rect_a['w'], rect_b['x'] + rect_b['w']) - max(rect_a['x'], rect_b['x'])
    inter_row = min(rect_a['y'] + rect_a['h'], rect_b['y'] + rect_b['h']) - max(rect_a['y'], rect_b['y'])
    inter_area = inter_col * inter_row
    area_a = rect_a['w'] * rect_a['h']
    area_b = rect_b['w'] * rect_b['h']
    iou = float(inter_area) / float(area_a + area_b - inter_area)
    return iou


# get image from disk
image_dir = '/media/grayson/DATA/tianjin/tianjin_selected/JPEGImages'

for row in result:
    image_path = os.path.join(image_dir, str(row['i_name']) + '.jpg')
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output_dict = inference.run_inference_for_single_image(sess, tensor_dict, image_tensor, image_rgb)

    detection_boxes = output_dict['detection_boxes']
    detection_scores = output_dict['detection_scores']
    ground_truth_boxes = convert_image_to_tf_record.parse_bounding_boxes(str(row['i_lanes']))

    for i in range(len(ground_truth_boxes['xmaxs'])):
        g_y = ground_truth_boxes['ymins'][i]
        g_x = ground_truth_boxes['xmins'][i]
        g_h = ground_truth_boxes['ymaxs'][i] - g_y
        g_w = ground_truth_boxes['xmaxs'][i] - g_x
        if g_h >= eval_min_ground_truth_scale and g_w >= eval_min_ground_truth_scale:
            total_num += 1
            for j in range(detection_boxes.shape[0]):
                if detection_scores[i] >= min_score_thresh:
                    d_y = detection_boxes[j][0]
                    d_x = detection_boxes[j][1]
                    d_h = detection_boxes[j][2] - d_y
                    d_w = detection_boxes[j][3] - d_x
                    iou = calculate_iou({'x': g_x, 'y': g_y, 'h': g_h, 'w': g_h},
                                        {'x': d_x, 'y': d_y, 'h': d_h, 'w': d_w})
                    if iou >= 0.4:
                        found_num += 1
                        break

    inference.display_with_opencv(image_rgb, output_dict, min_score_thresh=min_score_thresh, wait_time=1000)

print('recall:', float(found_num) / float(total_num))

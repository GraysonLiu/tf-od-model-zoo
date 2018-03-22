import cv2
import numpy as np
import tensorflow as tf

import rospy
from sensor_msgs.msg import CompressedImage
import time


def run_inference_for_single_image(sess, tensor_dict, image_tensor, image):
    time1 = time.time()

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    time2 = time.time()
    # print('{}ms'.format((time2 - time1) * 1000))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def compressed_image_detection_callback(image_compressed):
    np_data = np.fromstring(image_compressed.data, np.uint8)
    image_bayer = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_BG2BGR)
    image_bgr = cv2.resize(image_bgr, (960, 600))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output_dict = run_inference_for_single_image(sess, tensor_dict, image_tensor, image_rgb)
    display_with_opencv(image_rgb, output_dict)


def display_with_opencv(image_rgb, output_dict, min_score_thresh=0.1, wait_time=1):
    height, width, _ = image_rgb.shape
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    img_bgr_with_bounding_box = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for i in range(boxes.shape[0]):
        if scores[i] >= min_score_thresh:
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            cv2.rectangle(img_bgr_with_bounding_box, (int(xmin * width), int(ymin * height)),
                          (int(xmax * width), int(ymax * height)), (0, 0, 255), 2)
    cv2.namedWindow('image')
    cv2.imshow('image', img_bgr_with_bounding_box)
    cv2.waitKey(wait_time)


def init_model():
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

    PATH_TO_CKPT = '/home/grayson/ws/tf-od-model-zoo/models/ssd_mobilenet_v1_coco_2017_11_17/inference/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    return sess, tensor_dict, image_tensor


if __name__ == '__main__':
    sess, tensor_dict, image_tensor = init_model()
    sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims([[[0, 0, 0]]], 0)})

    # get image from ros message
    rospy.init_node('save_image', anonymous=True)
    rospy.Subscriber('/pylon_camera_node/image_raw/compressed', CompressedImage, compressed_image_detection_callback)
    rospy.spin()

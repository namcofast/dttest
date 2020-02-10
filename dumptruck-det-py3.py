import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from math import ceil
from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#import MySQLdb
import json
import requests

#MySQL Configuration

MYSQL_HOST = 'localhost'
MYSQL_USERNAME = 'root' #Enter user name of SQL DB default is root
MYSQL_PASSWORD = '' #Enter the root password created during MySQL installation
MYSQL_DB = 'lyrics' # Name of DB to write VEHICLE_LOG table

#Project Configuration

CAMERA_ID = '1'
PROJECT_ID = '1'
OBJECT_ID = '2'
REMARK = 'Test'

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output_model'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'training', 'object-detection-dump.pbtxt')

NUM_CLASSES = 1
i=0
prev_cent=0
entry_cnt=0
exit_cnt=0
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def chk_exit(x, cent, counter):
    if counter<50:
        return 0
    Dir = x - cent
    if (not(300 <= cent <= 337) and (Dir>0) and (300 <= x <=337)):
        print (cent)
        print (x)
        return 1
    else:
        return 0
def chk_entry(x, cent, counter):
    Dir = x - cent
    if counter< 50:
        return 0
    if (not(300 <= cent <= 337) and (Dir<0) and (300 <= x <=337)):
        print (x)
        print (x)
        return 1
    else:
        return 0
def detect_objects(image_np, sess, detection_graph):
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    image_np, cx, cy=vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)
    #print (image_np)
    return image_np, cx, cy


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb,sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    #db = MySQLdb.connect(MYSQL_HOST, MYSQL_USERNAME, MYSQL_PASSWORD, MYSQL_DB) # Enter the credentials of DB here
    #cursor = db.cursor()

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=1920, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=1080, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()
    i=0
    prev_cent=300
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src='lyrics.mp4',
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()
    
    #video capture
    out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 23, (480,240))
    
    #sqls = "CREATE TABLE IF NOT EXISTS VEHICLE_LOG(Camera_id VARCHAR(32),\
    #Entry_Count INT, Exit_Count INT,\
    #Project_id VARCHAR(32), Object_id VARCHAR(32),\
    #Remark VARCHAR(32), timestmp VARCHAR(45))"
    #cursor.execute(sqls)
    #db.commit()
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        if frame is None:
            break
        frame=cv2.resize(frame, (480, 240))
        input_q.put(frame)

        t = time.time()
        tm=str(time.asctime( time.localtime(time.time())))
        #output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        output_rgb,cx,cy=output_q.get()
        output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        if (chk_exit(ceil(cx), ceil(prev_cent), i)):
            exit_cnt+=1
            i = 0
        if (chk_entry(ceil(cx), ceil(prev_cent), i)):
            entry_cnt+=1
            i = 0
        ft = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_rgb,"  Trucks Entered: "+str(entry_cnt),(235,20), ft, 0.4,(255,0,0),1,cv2.LINE_AA)
        #cv2.putText(output_rgb,"Trucks Exited: "+str(exit_cnt)+"  Trucks Entered: "+str(entry_cnt),(235,20), ft, 0.4,(255,0,0),1,cv2.LINE_AA)
        log = 'log.txt'
        logfile = open(log, "a")
        sttime=time.strftime("%c")
        logfile.write(str(exit_cnt)+" Trucks exited "+str(entry_cnt)+" Trucks entered at "+sttime+'\n')

        logfile.close()
        
        #try:
        #    cursor.execute("INSERT INTO VEHICLE_LOG VALUES ('%s','%s','%s','%s','%s','%s','%s')"\
        #    % (CAMERA_ID,entry_cnt ,exit_cnt, PROJECT_ID, OBJECT_ID, REMARK,tm))
        #    db.commit()
        #except:
        #    db.rollback()
    #     payload = [{
    #         "Camera_id":CAMERA_ID,
    #         "Entry_Count":entry_cnt,
    #         "Exit_Count":entry_cnt,
    #         "Project_id":PROJECT_ID,
    #         "Object_id":OBJECT_ID,
    #         "Remark":REMARK,
    #         "timestamp":tm
    # }]
    #     r = requests.post('https://mobile-app2.gammonconstruction.com/SiteDiary/Lavender/CV/api/cv/cv_upload', json=json.dumps(payload))
    #     if r.status_code == 200:
    #         print("API sucess")
    #     else:
    #         print("API Error")
        # cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Video', 500,500)
        cv2.imshow('Video', output_rgb)
        #cv2.imwrite("output/"+str(i)+".jpg",output_rgb)
        out.write(output_rgb)
        fps.update()
        #print (i)
        i+=1
        prev_cent = cx
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
    #db.close()
    print('Closed all assets............')


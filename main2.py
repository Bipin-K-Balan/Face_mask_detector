import numpy as np
import tensorflow as tf
import os
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
import matplotlib.pyplot as plt


class Classifier():
    def __init__(self):
        cwd_path = os.getcwd()
        self.PATH_TO_MODEL = os.path.join(cwd_path,'inference_graph/frozen_inference_graph.pb')
        self.PATH_TO_LABEL = os.path.join(cwd_path,'inference_graph/mask_labelmap.pbtxt')
        #self.imagepath = os.path.join(cwd_path,'img1.png')

        self.class_no = 2

        self.labelmap = label_map_util.load_labelmap(self.PATH_TO_LABEL)
        self.catagories = label_map_util.convert_label_map_to_categories(self.labelmap,self.class_no,use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.catagories)

        self.class_names_mapping = {1: "Mask", 2: "Without Mask"}

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self,img):
        self.sess = tf.Session(graph=self.detection_graph)
        #img = cv2.imread('img1.png')
        img_expanded = np.expand_dims(img, axis=0)

        (boxes, scores, classes, num) = self.sess.run([self.d_boxes, self.d_scores, self.d_classes, self.num_d], feed_dict={self.image_tensor: img_expanded})
        #print(boxes.shape)
        # print(scores.shape)
        # print(classes.shape)
        #print(np.array(boxes))
        #print(np.array(scores))
        #print((np.array(classes)))
        # print(num)
        # scores_f = scores.flatten()
        # #print(scores_f)
        # sc = []
        # conf =[]
        # for i in range(0,len(scores_f)):
        #     if scores_f[i] > 0.80:
        #         sc.append(i)
        #         conf.append(scores_f[i])
        #
        # apprx_conf = [round(h,3) for h in conf]
        # #print('approxed conf is',apprx_conf)
        # #print('sc is',sc)
        # classes_f = classes.flatten()
        # #print(classes_f)
        # class_list = [classes_f[j] for j in sc]
        # #print('class list is',class_list)
        # #top_scores = [e for l2 in scores for e in l2 if e > 0.30]
        #
        # new_boxes = boxes.reshape(100, 4)
        # #print('new boxes',new_boxes)
        # #max_boxes_to_draw = new_boxes.shape[0]
        # box_list = [new_boxes[k] for k in sc]
        # #print('drawing box cordinates',box_list)
        # #print('max boxes to draw',len(box_list))
        # ymin,xmin,ymax,xmax = box_list[0]
        # #draw_box = cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),thickness=10)
        # #img_out = cv2.imshow('Output',draw_box)
        # #cv2.waitKey(0)
        vis_util.visualize_boxes_and_labels_on_image_array(img,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),self.category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.40)

        #return box_list,apprx_conf,class_list,len(box_list), img_out
        return img


## use this commands if we want to detect masks in video/ webcam
#'videoplayback.mp4'
# input_image = cv2.VideoCapture(0)
# input_image.set(3,640)
# input_image.set(4,480)
# input_image.set(10,100)
# while True:
#     bool, frame = input_image.read()
#     ob = Classifier()
#     a = ob.get_classification(img=frame)
#     cv2.imshow('Mask Detection',a)
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         break
# cv2.destroyAllWindows()
# cv2.VideoCapture(0).release()
#print(a,b,c,d,e)
#print(a)


#### for detecting masks in images
# image = cv2.imread('img2.png')
# ob = Classifier()
# a = ob.get_classification(img=image)
# cv2.imshow('Mask Detection',a)
# cv2.waitKey(0)



# from PIL import Image
# image = Image.fromarray(a,'RGB')
# image.save('output.png')
# image.show()

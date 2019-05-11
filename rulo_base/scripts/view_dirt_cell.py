#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from collections import deque, OrderedDict

path_dir = '/home/mtb/Documents/data/train/train_2.csv'
filename=  [path_dir]

def convert_str_to_float(val):
    elem_list = deque()
    for elem in val:
        elem_list.append(float(elem))
    return elem_list     
    


def convert_to_np(val_list, index):
    data = OrderedDict()
    # data = val_list[3][7:-1].split(',')
    # print len(data)
    data['inputs'] = np.asarray(convert_str_to_float(val_list[index[0]][1:-1].split(',')), dtype=np.float) 
    data['outputs'] = np.asarray(convert_str_to_float(val_list[index[1]][7:-2].split(',')), dtype=np.float)
    return data
   
def decode_csv(value):

    index = []
    extract_data = value.split('"')
    for elem in extract_data :
        if len(elem)>1:
            index.append(extract_data.index(elem))
    [input_index, output_index] = index
    
    in_, out = convert_to_np(extract_data, index).values()
    return in_, out



filename_queue = tf.train.string_input_producer(filename)
line_reader = tf.TextLineReader(skip_header_lines=1)
key, value = line_reader.read(filename_queue)

if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
       
        input_list = []
        output_list =[]
        for elem in range(2):
                features, labels = sess.run([key,value])
                print features
                val = decode_csv(labels)
               
                output_list.append(val[1])
        print list(output_list[0])
     
        coord.request_stop()
        coord.join(threads)
               
       
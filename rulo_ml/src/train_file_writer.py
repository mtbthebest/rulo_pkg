#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict
from csvreader import csvread


path_dir = '/home/barry/hdd/data/train/train_values/'
filename=  []
for files in os.listdir(path_dir):
    if files.endswith('.csv'):
        if files.startswith('train'):
            filename.append(files)        
filenames = [path_dir+ elem for elem in filename]


length_file = path_dir + 'length_dirt_val.csv'

file_length =[]
data = csvread(length_file)

for file_name in filename:
        file_length.append(int(data[0][file_name]))




def convert_str_to_float(val):
    elem_list = deque()
    for elem in val:
        elem_list.append(float(elem))
    return elem_list     
    


def convert_to_np(val_list, index):
    data = OrderedDict()

    data['inputs'] = np.asarray(convert_str_to_float(val_list[index[0]][1:-1].split(',')), dtype=np.float) 
    data['outputs'] = np.asarray(convert_str_to_float(val_list[index[1]][1:-1].split(',')), dtype=np.float)
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



filename_queue = tf.train.string_input_producer(filenames)
line_reader = tf.TextLineReader(skip_header_lines=1)
key, value = line_reader.read(filename_queue)

if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
       
        input_list = []
        output_list =[]

        for elem in range(sum(file_length)):  
                features, labels = sess.run([key,value] )
                print features           
                val = decode_csv(labels)
                if elem==0:
                    train_data= np.array([val[0],val[1]])
                else:                
                    train_data = np.vstack([train_data, np.array([val[0],val[1]])])
        
        print 'success'
        np.savez(path_dir+'data_grid_val_IO',train_data)
        coord.request_stop()
        coord.join(threads)
               
# a = [np.asarray([1,56,3,4]),np.asarray([1,5,3,4])]
# b = [np.asarray([1,6,3,4]),np.asarray([1,54,3,4])]
# # y = np.array([a,b])

# c = [np.asarray([1,8,3,4]),np.asarray([1,5,3,5])]
# d = [np.asarray([1,56,3,4]),np.asarray([1,54,3,4])]
# y = np.array([a,b,c,d])


# print y[:3,0]
# print y[2:]

# x =  np.arange(1500000,1550000)
# print np.random.choice(x, size=1)[0]
# print int(1500000/10000)
# a = np.array([1,2,3,4], ndmin=2)
# print a
# print 500%1000
# c = np.asarray([1,2,3,4])
# y = np.stack([y,np.array([c,d])])
# print y.shape
# y = np.vstack([y,np.array([a,b])])
# y =np.delete(y,0,0)
# np.random.shuffle(y)
# print y


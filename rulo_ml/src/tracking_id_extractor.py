#!/usr/bin/env	python
import os
import sys
import numpy as np
import csv
import pandas as pd
from pandas import Series, DataFrame
from math import fabs
from collections import deque, OrderedDict
# from rulo_utils.csvreader import csvread
# from rulo_utils.csvwriter import csvwriter
from time import sleep


tracking_folder = '/home/barry/hdd/tracking/'
# day_folder = ['21','22', '23', '24', '25', '26', '28', '29', '30']
day_folder = ['12-4/', '12-5/', '12-6/', '12-6/', '12-7/', '12-8/', '12-8/']
# day_folder = ['12-/11', '12-12/', '12-12/', '12-13/', '12-13/', '12-14/', '12-14/']
folder_name = ['noon', 'noon', 'morning', 'noon', 'noon', 'morning', 'noon']
# folder_name = ['noon', 'morning', 'noon', 'morning', 'noon', 'morning', 'noon']
filename = ['/HumanPositions/Real-time Data.csv']
start_line_list = [310721, 9639, 16315, 181277, 4178, 744, 531]
# start_line_list = [5850, 2248, 404, 4555, 896, 2079, 2190]
id_start = [np.array(['14151100']), np.array(['12033900']), np.array(['21164400']), np.array(
    ['12000201']), np.array(['9140600']), np.array(['18270100', '18271300', '18285400']), np.array(['12145100'])]
# id_start = [np.array(['10185900']), np.array(['19174400']), np.array(['10552702']), np.array(
# ['22015600']), np.array(['10293500', '10294300']), np.array(['20221302', '20221304', '20221303', '20221400', '20221402']), np.array(['11054500', '11155600'])]


pose_start = [np.array([[4.285, 1.939]]), np.array([[1.746, 7.105]]), np.array([[3.903, 2.426]]), np.array([[4.233, 1.964]]), np.array(
    [[1.832, 7.163]]), np.array([[3.953, 1.806], [0.165, 6.538], [1.722, -1.702]]), np.array([[0.438, 5.007]])]

# pose_start = [np.array([[1.506, 6.404]]), np.array([[2.619, 4.879]]), np.array([[0.614, 5.095]]), np.array([[2.721, 4.895]]), np.array(
# [[0.212, 5.322], [-1.079, 5.257]]), np.array([[3.953, 1.806], [0.165, 6.538], [1.722, -1.702]]), np.array([[1.328, -3.250], [0.140, 5.323]])]
# save_path = [tracking_folder + 'dataframe/' + str(i)+ '.csv' for i in range(len(day_folder))]
save_path = [tracking_folder + 'id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.npy'
             for i in range(len(day_folder))]
save_csv = [tracking_folder + 'id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.csv'
            for i in range(len(day_folder))]


id_save_path = [tracking_folder + 'id-12-24/id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.npy'
                for i in range(len(day_folder))]
pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
entrance_corners_x_limit = [3.2, 4.5]
entrance_corners_y_limit = [1.3, 2.9]
# entrance_corners_x_limit = [2.4, 4.2]
# entrance_corners_y_limit = [1.0, 4.2]


class Extraction:

    def reset(self, value):
        try:
            value = np.delete(value, range(0, value.shape[0]), axis=0)
        except:
            pass
    def extract_id(self):
          
            # self.filename = [tracking_folder + day_folder[i] + folder_name[i] + filename[0] for i in range(len(day_folder))]
            self.filename = ['/home/mtb/Desktop/Real-time Data.csv']
            step_print = 1
            # print save_path
            # for k in range(0, len(self.filename)-3):
            for k in range(1):
                with open(self.filename[k], 'r') as csvfile:
                # with open(self.filename[0], 'r') as csvfile:
                    self.data = OrderedDict()
                    csvreader = csv.reader(csvfile)
                    time_list_unique = []
                    start_line = start_line_list[k]
                    self.human_id = id_start[k]
                    self.human_pose = pose_start[k]

                    start_line = start_line_list[3]
                    self.human_id = id_start[3]
                    self.human_pose = pose_start[3]

                    self.last_position = OrderedDict()
                    self.missing_match = OrderedDict()
                    self.front_door_buffer = OrderedDict()
                    self.back_door_buffer = OrderedDict()
                    self.front_door_buffer_last_pose = OrderedDict()
                    self.back_door_buffer_last_pose = OrderedDict()
                    self.missing_id_dict = OrderedDict()
                    self.missing_found_id_dict = OrderedDict()
                    self.buffer_dict = OrderedDict()
                    self.replacement = OrderedDict()

                    self.buffer_list = []

                    self.missing_flag = False
                    try:
                        for elem in [self.id, self.human_id, self.missing_id_dict,self.eject_human_id]:
                            self.reset(elem)
                    except:
                        pass
        
                    self.number_of_human = self.human_id.shape[0]                  
                    for i in range(self.number_of_human):
                        try:
                            self.id = np.concatenate((self.id, [self.human_id[i]]))
                        except:
                            self.id = np.array([self.human_id[i]])
                        self.last_position[self.human_id[i]] = self.human_pose[i]
                    self.step = 0
                    for row in csvreader:
                        self.step += 1
                        if self.step >= start_line:
                            if float(row[0]) not in time_list_unique:
                                self.data = OrderedDict()
                                del time_list_unique[:]
                                time_list_unique.append(float(row[0]))
                                id_list = []
                                coordinates = []
                          
                            self.data = OrderedDict()
                            if int(row[2]) > 0:    # Number of people
                                    for elem in row:
                                        if elem == '':
                                            row.remove(elem)
                                    start = 4
                                    step = 7
                                    num_tracked = int(row[2])
                                    for i in np.linspace(start, len(row), num_tracked + 1, dtype=np.int32):
                                        try:
                                            if int(row[i]) == 0:    # human detection
                                                x = (
                                                    float(row[i + 3]) / float(1000))
                                                y = - \
                                                    (float(
                                                        row[i + 2]) / float(1000))
                                              
                                                self.data[str(row[i + 1])] = [x,y]                             
                                        except:
                                            break                                
                                    self.process_data()                                   
                    
                        # if self.step >= start_line +1:
                        #     sys.exit(0)
                        
                        if step_print * 100000 < self.step:
                            print str(self.filename[k]) + ': ' + str(self.step)
                            step_print +=1
                            # print self.id
                        # print str(self.filename[k]) + ': ' + str(self.step)
                np.save(id_save_path[k], self.id)
                      

    def process_data(self):
        # First check if the present id are still in the room
        self.check_for_present_human_id()
        #Second check the new id
        self.check_for_new_human_id()  
        #Third check the buffers if it is a human
        self.verify_new_human_id()
        #Fourth reset if the missing one was found
       
        
    def check_for_present_human_id(self):
        id_quit_the_room=[]
        for present_ids in self.human_id:
            # Present in the room
            if present_ids in self.data.keys():
                #Update the last position
                self.update_last_position(present_ids, self.data[present_ids])
            # Missing
            else:
                last_position = self.last_position[present_ids]
                if last_position[0] > 4.27 and 1.5 < last_position[1] < 2.9:
                    index = np.where(self.human_id == present_ids)[0][0]
                    id_quit_the_room.append(index)
                # The person went to the small room if it his y position is less than 0.7 and his x position is between 2.1 and 3.1 (door hinges position) either exited or doing things
                elif last_position[1] < 0.7 and 2.1 < last_position[0] < 3.1:
                    self.add_to_missing(present_ids, last_position, big_room=False, small_room=True, server_room=False)
                # The person went to the server room either working or sleeping
                elif (last_position[1] > 7.1 and 1.3 < last_position[0] < 2.0) or\
                     (last_position[1] > 4.8 and 2.1 < last_position[0] < 3.0):
                    self.add_to_missing(present_ids, last_position, big_room=False, small_room=False, server_room=True)
                # The person is missing inside the big room
                else:
                    self.add_to_missing(present_ids, last_position, big_room=True, small_room=False, server_room=False)
        if id_quit_the_room:
            self.human_id = np.delete(self.human_id, id_quit_the_room, axis=0)

    def check_for_new_human_id(self):
        for detected_ids in self.data:
            #Check if it is a new human id recorded in the room
            if self.missing_found_id_dict.keys():
                id_name = self.missing_found_id_dict.keys()[0]
                self.id = np.concatenate((self.id, [self.missing_found_id_dict[id_name][0]]), axis=0)
                self.human_id = np.concatenate((self.human_id, [self.missing_found_id_dict[id_name][0]]), axis=0)
                self.last_position[id_name] = self.missing_found_id_dict[id_name][1]
                self.reset_missing_buffer(id_name)
            self.missing_found_id_dict = OrderedDict()
            if detected_ids not in self.human_id:
                # print 'Check for: ', detected_ids     
                try:
                    # If the id is in eject list just pass it
                    if detected_ids in self.eject_human_id:
                        pass
                    #Update the buffer if already there
                    elif detected_ids in self.buffer_dict.keys():                       
                        self.buffer(ids=detected_ids, ids_pose=self.data[detected_ids], verify='false', update='true')
                    else:
                        self.detect_in_room(detected_ids, self.data[detected_ids])
                except:                    
                    self.detect_in_room(detected_ids, self.data[detected_ids])
            else:
                pass

    def detect_in_room(self, ids, pose):
        # First Check if it's the corresponding missing one:
        if self.missing_flag:
            flag = False
            for missing_id in self.missing_id_dict.keys():
                missing_last_position = self.missing_id_dict[missing_id]['last_position']
                if self.missing_id_dict[missing_id]['location'] == 'big_room':
                    if self.check_for_neighbor_missing(missing_id,missing_last_position, 
                                                        ids,pose):
                        flag = True
                        break                    
                elif self.missing_id_dict[missing_id]['location'] == 'server_room':
                    if self.check_for_neighbor_missing(missing_id, missing_last_position,
                                                       ids, pose):
                        flag = True
                        break
                    elif self.check_gateway(missing_id, missing_last_position, ids, pose):
                        flag = True
                        break
                elif self.missing_id_dict[missing_id]['location'] == 'small_room':
                    if self.check_for_neighbor_missing(missing_id, missing_last_position, ids, pose, location='small_room'):
                        flag = True
                        try:
                            self.missing_match[missing_id].append(ids)
                        except:
                            self.missing_match[missing_id] = [ids]
                        self.buffer(ids, pose, door='back', verify='false', update='false')
                        
            if not flag:
                if self.check_for_new_entrance_front_door(ids, pose):
                    pass
                elif self.check_for_new_entrance_back_door(ids, pose):
                    pass
                else:
                    self.eject(ids)

        # if not just move check if it is a new person or the door being open or close
        else:
            # No missing person so check if the corresponding is either a door or a human to keep it in a buffer

            # Buffer it if between the boundaries of the front entrance
            if self.check_for_new_entrance_front_door(ids, pose):
                pass
            # Buffer it if between the boundaries of the back entrance
            elif self.check_for_new_entrance_back_door(ids, pose):
                pass
            # Eject it if it is not between the boundaries of none of the doors
            else:
                self.eject(ids)

    def check_for_new_entrance_front_door(self, ids,pose):
        '''Check for new human id near the front door
        '''
        if (entrance_corners_x_limit[0] <=pose[0]<=entrance_corners_x_limit[1] and\
            entrance_corners_y_limit[0] <=pose[1]<=entrance_corners_y_limit[1]) :
            # Put it in the buffer for further analysis in the next steps            
            self.buffer(ids,ids_pose=pose, door='front', verify='false',update= 'false')   
            return True       
        else:
            return False

    def check_for_new_entrance_back_door(self, ids, pose):
        ''' Check if we have an entrance at the back door or if it is the missing id
        '''
        if (2.1 <= pose[0] <= 3.0 and 0.1 <= pose[1] <= 0.9):
            self.buffer(ids, ids_pose=pose, door='back',verify='false', update='false')
            return True
        else:
            return False
        
    def check_for_neighbor_missing(self, missing_id=None, missing_last_position=None, ids=None, ids_pose=None, location='None'):
        neighbor_boundary = 0.9
        if fabs(ids_pose[0] - missing_last_position[0]) <= neighbor_boundary and\
           fabs(ids_pose[1] - missing_last_position[1]) <= neighbor_boundary:
            if not location == 'small_room':
                self.missing_found_id_dict[missing_id] = [ids, ids_pose]
            return True
        else:
            return False
    
    def check_gateway(self, missing_id, missing_last_position, ids, pose):
        if (2.1 < missing_last_position[0] < 3.1 and 4.8 < missing_last_position[1]< 5.2) and\
            (1.2 < pose[0] < 2.0 and 6.7 < pose[1] < 7.7):
            self.missing_found_id_dict[missing_id] = [ids, pose]
            return True
        elif (1.2 < missing_last_position[0] < 2.0 and missing_last_position[1] > 7.1) and \
             (2.1 < pose[0] < 3.0 and 4.5 < pose[1] < 5.2):
            
            self.missing_found_id_dict[missing_id] = [ids, pose]
            return True
        else:
            return False
    
    def eject(self, ids):                           
        try:
            if ids not in self.eject_human_id: 
                self.eject_human_id = np.concatenate((self.eject_human_id, [ids]), axis=0)
        except:
            self.eject_human_id = np.array([ids])  

    def update_last_position(self, ids, pose):
        # if ids in self.last_position.columns.values:
        #     for length in range(len(self.last_position)):
        #                 for n in range(len(self.last_position.loc[length, ids])):
        #                     self.last_position.loc[length, ids][n] = pose[n]
        # else:
        self.last_position[ids] = pose

    def add_to_missing(self, ids, last_pose, big_room=True, small_room=False, server_room=False):
        if ids not in self.missing_id_dict.keys():
            self.missing_id_dict[ids] = OrderedDict()
            if big_room:
                self.missing_id_dict[ids]['location'] = 'big_room'
            elif small_room:
                self.missing_id_dict[ids]['location'] = 'small_room'
            elif server_room:
                self.missing_id_dict[ids]['location'] = 'server_room'
            self.missing_id_dict[ids]['step'] = self.step
            self.missing_id_dict[ids]['last_position'] = last_pose
            try:
                self.missing_human_id = np.concatenate((self.missing_human_id, [ids]), axis=0)
            except:
                self.missing_human_id = np.array([ids])
        self.missing_flag = True

    def buffer(self, ids=None, ids_pose=None, door='front', verify='false', update='false'):
        '''A function to check if it is a door or human or hanging id in the back door
        '''
        # When putting the id in the first time in the buffer
        if update == 'false' and verify == 'false':
            self.buffer_dict[ids] = self.step
            if door == 'front':
                self.front_door_buffer[ids] = np.array([ids_pose])
                self.front_door_buffer_last_pose[ids] = ids_pose
            elif door == 'back':
                self.back_door_buffer[ids] = np.array([ids_pose])
                self.back_door_buffer_last_pose[ids] = ids_pose
        # Update if already in the dictionary buffer
        if update == 'true' and verify == 'false':
                if ids in self.front_door_buffer.keys():
                    self.front_door_buffer[ids] = np.concatenate((self.front_door_buffer[ids], [ids_pose]), axis=0)
                    self.front_door_buffer_last_pose[ids] = ids_pose
                elif ids in self.back_door_buffer.keys():
                    self.back_door_buffer[ids] = np.concatenate((self.back_door_buffer[ids], [ids_pose]), axis=0)
                    self.back_door_buffer_last_pose[ids] = ids_pose        
        #Verify the state
        if verify == 'true' and update == 'false':
            front_process_flag = []
            back_process_flag = []

            for ids in self.front_door_buffer.keys():
                ids_pose = self.front_door_buffer_last_pose[ids]                   
                if ids_pose[0] <= 3.0:
                    front_process_flag.append(ids)
                    self.human_id = np.concatenate((self.human_id, [ids]), axis=0)
                    self.id = np.concatenate((self.id, [ids]), axis=0)
                    self.last_position[ids] = ids_pose
                else:
                    if self.front_door_buffer[ids].shape[0] + 4000 < (self.step - self.buffer_dict[ids]) or\
                        self.front_door_buffer[ids].shape[0] > 24000:
                            front_process_flag.append(ids)
                            self.eject(ids)
            if front_process_flag:
                for ids in front_process_flag:
                    del self.front_door_buffer[ids]
                    del self.front_door_buffer_last_pose[ids]
                    del self.buffer_dict[ids]

            for ids in self.back_door_buffer.keys():
                ids_pose = self.back_door_buffer_last_pose[ids]
                if ids_pose[1] >= 0.65 and 2.0 <=ids_pose[0]<= 3.0:
                    back_process_flag.append(ids)
                    self.human_id = np.concatenate((self.human_id, [ids]), axis=0)
                    self.id = np.concatenate((self.id, [ids]), axis=0)
                    self.last_position[ids] = ids_pose
                else:
                    if self.back_door_buffer[ids].shape[0] + 4000 < (self.step - self.buffer_dict[ids]) or\
                        self.back_door_buffer[ids].shape[0] > 24000:
                        back_process_flag.append(ids)
                        self.eject(ids)
            if back_process_flag:                
                for ids in back_process_flag:
                    id_to_del = []
                    missing_flag_find = False
                    del self.back_door_buffer[ids]
                    del self.back_door_buffer_last_pose[ids]
                    del self.buffer_dict[ids]
                    if ids in self.id:
                        if self.missing_match.keys():
                            for missing_ids in self.missing_match.keys():
                                if ids in self.missing_match[missing_ids]:
                                    id_to_del.append(missing_ids)
                        if id_to_del:
                            for id_garb in id_to_del[1:]:
                                self.missing_match[id_garb].remove(ids)    
                            del self.missing_match[id_to_del[0]]    
                            self.reset_missing_buffer(id_to_del[0])
                    elif ids in self.eject_human_id:
                        if self.missing_match.keys():
                            for missing_ids in self.missing_match.keys():
                                if ids in self.missing_match[missing_ids]:
                                    id_to_del.append(missing_ids)                      
                        if id_to_del:
                            for miss_ids in id_to_del:
                                    self.missing_match[miss_ids].remove(ids)
                             
    def verify_new_human_id(self):
        self.buffer(verify='true', update='false')
    
    def reset_missing_buffer(self, ids):
        del self.missing_id_dict[ids]
        self.missing_human_id = np.delete(self.missing_human_id, np.where(self.missing_human_id == ids)[0][0], axis=0)
        if self.missing_human_id.shape[0] == 0:
            self.missing_flag = False

if __name__ == '__main__':
    Extraction().extract_id()

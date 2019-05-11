#!/usr/bin/env	python
import os
import psutil

import sys
import numpy as np
import csv
import pandas as pd
from pandas import Series, DataFrame
from math import fabs 
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from time import sleep
from rulo_utils.csvconverter import csvconverter, convert_list_to_int, convert_lists_to_int
# path = '/media/mtb/39A7D6837693DAFB/tracking_data/'
# path = '/media/mtb/005114C404F86E90/tracking data/11-14_11-16/HumanPositions/'
import affinity

from multiprocessing import Process, Queue, Lock

from threading import Thread
# os.system("taskset -p 0xff %d" % os.getpid())

tracking_folder = '/home/barry/hdd/tracking/'
# day_folder = ['21','22', '23', '24', '25', '26', '28', '29', '30']
day_folder = ['12-4/','12-5/','12-6/', '12-6/','12-7/','12-8/', '12-8/']
# day_folder = ['12-/11', '12-12/', '12-12/', '12-13/', '12-13/', '12-14/', '12-14/']
folder_name = ['noon','noon','morning','noon','noon','morning', 'noon']
# folder_name = ['noon', 'morning', 'noon', 'morning', 'noon', 'morning', 'noon']
filename = ['/HumanPositions/Real-time Data.csv']
start_line_list = [310721,9639,16315, 181277, 4178,744, 531]
# start_line_list = [5850, 2248, 404, 4555, 896, 2079, 2190]
id_start = [np.array(['14151100']),np.array(['12033900']),np.array(['21164400']), np.array(['12000201']) , np.array(['9140600']), np.array(['18270100','18271300', '18285400']), np.array(['12145100'])]
# id_start = [np.array(['10185900']), np.array(['19174400']), np.array(['10552702']), np.array(
    # ['22015600']), np.array(['10293500', '10294300']), np.array(['20221302', '20221304', '20221303', '20221400', '20221402']), np.array(['11054500', '11155600'])]


pose_start = [np.array([[4.285, 1.939]]), np.array([[1.746,7.105]]),np.array([[3.903,2.426]]), np.array([[4.233,1.964]]), np.array([[1.832, 7.163]]),np.array([[3.953,1.806],[0.165,6.538], [1.722,-1.702]]), np.array([[0.438,5.007]])]

# pose_start = [np.array([[1.506, 6.404]]), np.array([[2.619, 4.879]]), np.array([[0.614, 5.095]]), np.array([[2.721, 4.895]]), np.array(
    # [[0.212, 5.322], [-1.079, 5.257]]), np.array([[3.953, 1.806], [0.165, 6.538], [1.722, -1.702]]), np.array([[1.328, -3.250], [0.140, 5.323]])]
# save_path = [tracking_folder + 'dataframe/' + str(i)+ '.csv' for i in range(len(day_folder))]
save_path = [tracking_folder + 'id/' + str(day_folder[i][:-1])+'_'+folder_name[i] +'.npy'
                 for i in range(len(day_folder))]
save_csv = [tracking_folder + 'id/' + str(day_folder[i][:-1])+'_'+folder_name[i] +'.csv'
                 for i in range(len(day_folder))]
pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
entrance_corners_x_limit = [3.2, 4.5]
entrance_corners_y_limit = [1.3, 2.9]
# entrance_corners_x_limit = [2.4, 4.2]
# entrance_corners_y_limit = [1.0, 4.2]
class Tracking():
    def __init__(self):
        pass      
    
    def follow_id(self):
            # lock.acquire()  
            # self.filename = [tracking_folder + day_folder[i] + folder_name[i] + filename[0] for i in range(len(day_folder))]
            self.filename = ['/home/mtb/Desktop/Real-time Data.csv']
            # print save_path
            # for k in range(0, len(self.filename)-3): 
            for k in range(len(day_folder)):                 
                # with open(self.filename[k], 'r') as csvfile:
                with open(self.filename[0], 'r') as csvfile:
                    self.data = OrderedDict()
                    csvreader = csv.reader(csvfile)                
               
                    time_list_unique = []                   
                    # start_line = start_line_list[k] 
                    # self.human_id = id_start[k]   
                    # self.human_pose = pose_start[k] 

                    start_line = start_line_list[3] 
                    self.human_id = id_start[3]   
                    self.human_pose = pose_start[3] 

                    
                    self.human_id_dataframe =DataFrame()
                    self.last_position = DataFrame()
                    self.front_door_buffer = DataFrame()
                    self.front_door_buffer_last_pose = DataFrame()
                    self.back_door_buffer_last_pose  = DataFrame()
                    self.missing_found_id_dataframe = DataFrame()
                    self.back_door_buffer = DataFrame()
                    # self.big_room = DataFrame()
                    # self.small_room = DataFrame()   
                    # self.server_room = DataFrame()

                    self.missing_id_dict = OrderedDict()  
                    self.missing_found_id_dict = OrderedDict()                 
                    self.buffer_dict= OrderedDict()
                    self.replacement = OrderedDict()

                    self.buffer_list = []          
                    
                    self.missing_flag= False
                    # self.small_room_num_human = 0

                    try:
                        self.missing_human_id = np.delete(self.missing_human_id, range(0, self.missing_human_id.shape[0]), axis=0)
                    except:
                        pass
                    
                    try:
                        self.eject_human_id = np.delete(self.eject_human_id, range(0, self.eject_human_id.shape[0]), axis=0)
                    except:
                        pass
                    
                    self.number_of_human = self.human_id.shape[0]
                    for i in range(self.number_of_human):
                        self.human_id_dataframe[self.human_id[i]] =[self.human_pose[i]]   
                        self.last_position[self.human_id[i]] = [self.human_pose[i]]
                
                    self.step = 0
                    for row in csvreader:
                        self.step +=1                   
                        if self.step >=start_line:                           
                            if float(row[0]) not in time_list_unique:
                                self.data = OrderedDict()                                
                                self.data[float(row[0])] =OrderedDict()
                                                           
                                del time_list_unique[:]
                                time_list_unique.append(float(row[0]))
                                id_list = []
                                coordinates =[]
                            self.data[float(row[0])]['id'] = []
                            self.data[float(row[0])]['pose']  =  []   
                            if int(row[2]) > 0 :    # Number of people                     
                                    for elem in row:
                                        if elem == '':
                                            row.remove(elem)
                                    start = 4
                                    step = 7
                                    num_tracked = int(row[2])
                                    for i in np.linspace(start, len(row), num_tracked +1 , dtype=np.int32):                                                                  
                                        try:
                                            if int(row[i]) == 0:    # human detection                                                                          
                                                x = (float(row[i+3])/float(1000))
                                                y = -(float(row[i +2]) / float(1000))
                                                pose = np.array([x,y])
                                                # if str(row[i + 1]) in self.human_id:
                                                self.data[float(row[0])]['id'].append(str(row[i+1]))  
                                                self.data[float(row[0])]['pose'].append(np.array(pose))
                                                # self.human_id_dataframe = self.human_id_dataframe.append({str(row[i+1]): pose}, ignore_index=True)                                                                    
                                        except :
                                            break 
                                    # os.system("taskset -p 0xff %d" % os.getpid())
                                    # print psutil.Process().cpu_affinity()
                                    # thread = Thread(target=self.fill_the_list, args=(5))
                                    # thread.start()
                                    # thread.join()
                                    # self.fill_the_list()
                                    self.process()
                                    # print self.data[float(row[0])]
                        if  self.step >=  360000:
                            sys.exit(0)
                        else:
                            pass                         
                        print str(self.filename[k]) + ': ' + str(self.step) 
                        self.human_id_dataframe.to_csv()
    def process(self):
        self.time_key = self.data.keys()[-1]
        for ids in self.data[self.time_key]['id']:
            indice = self.data[self.time_key]['id'].index(ids)
            pose = self.data[self.time_key]['pose'][indice]
            self.human_id_dataframe = self.human_id_dataframe.append({ids: pose}, ignore_index=True)
            # print self.human_id_dataframe
            
        
    def fill_the_list(self): 
           
        self.time_key = self.data.keys()[-1]
        self.new_id =OrderedDict()
        self.missing_found_id_dict = OrderedDict()
        id_quit_the_room = []
        self.new_human= []
        self.missing_updated = []

        for present_ids in self.human_id: 
            # Present in the room       
            if present_ids in self.data[self.time_key]['id']:  
                # The person is still in the room, append to dataframe and update his position    
                #If he was lost somewhere then delete it from the missing array and dictionary
                self.add_to_human_dataframe(present_ids, self.data[self.time_key]['pose'][self.data[self.time_key]['id'].index(present_ids)])
            else:
                # 4 choices : left the room , in the small room, in the server room,in the big room either sitting or bending 
                # First check where it was first seen meaning his last location                                                         
                last_position = self.last_position[present_ids][0]
                # The person has exited the room if it his x position is greater than 4.27 and his y position is between 1.5 and 2.9 (door hinges position)
                if last_position[0] > 4.27 and  1.5<last_position[1]<2.9:                                      
                    index = np.where(self.human_id == present_ids)[0][0]
                    id_quit_the_room.append(index)   
                # The person went to the small room if it his y position is less than 0.7 and his x position is between 2.1 and 3.1 (door hinges position) either exited or doing things 
                elif last_position[1] < 0.7 and  2.1<last_position[0]<3.1:
                    self.add_to_missing(present_ids, last_position, big_room = False, small_room = True, server_room= False)  
                # The person went to the server room either working or sleeping
                elif (last_position[1] > 7.1 and  1.3<last_position[0]<2.0) or (last_position[1] > 4.8 and  2.1<last_position[0])<3.0  :                    
                    self.add_to_missing(present_ids, last_position, big_room = False, small_room = False, server_room= True)                                    
                # The person is missing inside the big room
                else:  
                    self.add_to_mising(present_ids, last_position, big_room = True, small_room = False, server_room= False)
                                  
        if id_quit_the_room:
            self.human_id = np.delete(self.human_id, id_quit_the_room, axis=0) 

        # For the new detected ids check if it's a new person or the substituant to the missing person
        for detected_ids in self.data[self.time_key]['id']:     
            #Check if it is a new human id recorded in the room       
            if detected_ids not in self.human_id:
                # print 'Check for: ', detected_ids
                #Index for the position of the id
                indice = self.data[self.time_key]['id'].index(detected_ids)
                try:                                       
                    # If the id is in eject list just pass it
                    if detected_ids in self.eject_human_id :
                        pass                    
                    #Update the buffer if already there
                    elif detected_ids in self.buffer_dict.keys():
                        print 'Updating the buffer'         
                        self.buffer(ids=detected_ids, ids_pose= self.data[self.time_key]['pose'][indice], verify='false', update='true')   
                    # elif detected_ids in self.possible_human:
                    #     self.check_for_neighbor_missing(ids= detected_ids, ids_pose=self.data[self.time_key]['pose'][indice], update = True)
                    #Completely a new id search if it is a new human or the door or a hanging id left over
                    else:
                        self.detect_in_room(detected_ids, self.data[self.time_key]['pose'][indice])
                except:  
                    print 'find in room'                 
                    self.detect_in_room(detected_ids, self.data[self.time_key]['pose'][indice])
            else:
                pass            
            self.refresh_missing_list() 
                
                
       
        self.update_human_id()
        #Check the buffer for the expected result if it is the door or human
        if self.buffer_dict.keys():
            # print 'buffer list: ',self.buffer_list
            self.buffer(verify='true', update='false')


        # if self.new_human:
        #     for humans in self.new_human:
        #         self.human_id = np.concatenate((self.human_id, [humans]))
        #    
        # self.check_found_missing()

        # try:
        #     if self.missing_human_id.shape[0] ==0:                
        #         self.missing_flag = False
        # except:
        #     pass
        print 'Human: ', self.human_id_dataframe.columns.values
       
    def detect_in_room(self,ids, pose):               
        # First Check if it's the corresponding missing one:
        if self.missing_flag:     
            flag=  False  
            eject = False    
            for m in range(self.missing_human_id.shape[0]):
                # detected_ids_pose = pose        
                missing_last_position = self.last_position[self.missing_human_id[m]][0]                
                if self.missing_id_dict[self.missing_human_id[m]]['location'] == 'big_room':
                    if self.check_for_neighbor_missing(self.missing_human_id[m],missing_last_position,ids, pose): 
                        flag = True
                        break
                    else:
                        pass                                 
                elif self.missing_id_dict[self.missing_human_id[m]]['location'] == 'server_room':
                    if self.check_for_neighbor_missing(self.missing_human_id[m],missing_last_position,ids, pose):
                        flag = True
                        break
                    elif self.check_gateway(self.missing_human_id[m], missing_last_position, ids, pose):  
                        flag = True
                        break  
                elif self.missing_id_dict[self.missing_human_id[m]]['location'] == 'small_room':
                    if self.check_for_neighbor_missing(self.missing_human_id[m],missing_last_position,ids, pose):
                        flag = True
                        self.buffer(ids, pose, door='back', verify='false', update='false')
                        if not self.missing_human_id[m] in self.replacement.keys():
                            self.replacement[self.missing_human_id[m]] = [ids]
                        else:
                            self.replacement[self.missing_human_id[m]].append(ids)
                        break

                # elif self.missing_id_dict[self.missing_human_id[m]]['location'] == 'small_room':
                #     if self.check_for_neighbor_missing(self.missing_human_id[m],missing_last_position,ids, pose):
                #         flag = True
                #         break
                    
            if not flag:
                if self.check_for_new_entrance_front_door(ids, pose): pass
                elif self.check_for_new_entrance_back_door(ids, pose):  pass
                else:self.eject(ids)
                
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

    def check_gateway(self, missing_id, missing_last_position,ids, pose):        
        if (2.1 <missing_last_position[0]<3.1 and missing_last_position[1] > 4.8) and (1.2<pose[0]<2.0 and 6.7<pose[1]<8.0) :
            #Front door of the server room
            # id_not_used_flag = True
            # for keys_id in self.new_id:
            #     if self.new_human
            if  missing_id not in self.new_id.keys():                
                self.new_id[missing_id] = [ids, pose]
            return True
        elif (1.2 <missing_last_position[0]<2.0 and missing_last_position[1] > 7.1) and (2.1<ids_pose[0]<3.0 and 4.0<ids_pose[1]< 5.1):
            if  missing_id not in self.new_id.keys(): 
                self.new_id[missing_id] = [ids, pose]
            return True
        else:
            return False
                   
    def check_for_neighbor_missing(self, missing_id=None, missing_last_position=None,ids=None, ids_pose=None):#, update= False):
        neighbor_boundary = 0.9
        # if not update:
        if fabs(ids_pose[0] - missing_last_position[0])<= neighbor_boundary and\
           fabs(ids_pose[1] - missing_last_position[1])<= neighbor_boundary:
            # if missing_id not in self.missing_found_id_dict.keys():
                self.missing_found_id_dict[missing_id] = [ids, ids_pose]
                # self.missing_found_id_dict[missing_id] = OrderedDict()
                # self.missing_found_id_dict[missing_id]['id'] = ids
                # # self.missing_found_id_dict[missing_id]['step'] = [self.step]
                # self.missing_found_id_dict[missing_id]['pose'] = ids_pose
                # try:
                #     self.possible_human.append(ids_pose)
                # except:
                #     self.possible_human = [ids_pose]
                # self.missing_found_id_dataframe = self.missing_found_id_dataframe.append({ids: ids_pose}, ignore_index=True)
            # else:
            #     self.missing_found_id_dict[missing_id]['id'].append(ids)
            #     self.missing_found_id_dict[missing_id]['step'].append(self.step)
            #     self.missing_found_id_dict[missing_id]['pose'].apppend(ids_pose)
                # self.missing_found_id_dataframe = self.missing_found_id_dataframe.append({ids: ids_pose}, ignore_index=True)
                return True
                
        else:
            # Outside the neighbor missing
            return False
      
    def eject(self, ids):                           
        try:
            if not len(np.where(self.eject_human_id == ids)[0]) == 1: 
                self.eject_human_id = np.concatenate((self.eject_human_id, [ids]), axis=0)
        except:
            self.eject_human_id = np.array([ids])  
      
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
        if (2.1 <=pose[0]<=3.0 and 0.1<=pose[1]<=0.9):         
            self.buffer(ids , ids_pose= pose, door='back', verify='false',update='false')
            return True            
        else:
            return False
                        
    def buffer(self, ids=None, ids_pose= None, door='front', verify='false', update='false'):    
        '''
        A function to check if it is a door or human or hanging id in the back door
        '''   
        # When putting the id in the first time in the buffer
        if update=='false' and verify =='false':
            self.buffer_dict[ids] = self.step
            if door == 'front':
                self.front_door_buffer = self.front_door_buffer.append({ids: ids_pose}, ignore_index=True)
                self.front_door_buffer_last_pose[ids] = [ids_pose]
            elif door == 'back':
                self.back_door_buffer = self.back_door_buffer.append({ids: ids_pose}, ignore_index=True)
                self.back_door_buffer_last_pose[ids] = [ids_pose]           
        # Update if already in the dictionary buffer
        if update=='true' and verify =='false':            
                if ids in self.front_door_buffer.columns.values:
                    self.front_door_buffer = self.front_door_buffer.append({ids: ids_pose}, ignore_index=True)                
                    for length in range(len(self.front_door_buffer_last_pose)):
                                for n in range(len(self.front_door_buffer_last_pose.loc[length,ids])):                       
                                    self.front_door_buffer_last_pose.loc[length,ids][n]=ids_pose[n]

                elif ids in self.back_door_buffer.columns.values:
                    self.back_door_buffer = self.back_door_buffer.append({ids: ids_pose}, ignore_index=True)                      
                    for length in range(len(self.back_door_buffer_last_pose)):
                                for n in range(len(self.back_door_buffer_last_pose.loc[length,ids])):                       
                                    self.back_door_buffer_last_pose.loc[length,ids][n]=ids_pose[n]

        if verify == 'true' and update=='false':
            front_process_flag = []
            back_process_flag = []
            replacement_list = []
            for ids in self.front_door_buffer.columns.values:
                # if ids !=None:
                    ids_pose = self.front_door_buffer_last_pose[ids][0]
                    # print 'columns: ',self.front_door_buffer.columns.values
                    if ids_pose[0] <= 3.0:
                        # print True
                        # self.new_human.append(ids)
                        front_process_flag.append(ids)
                        self.human_id = np.concatenate((self.human_id, [ids]), axis=0)
                        # self.human_id_dataframe = pd.concat([self.human_id_dataframe, self.front_door_buffer], axis=1)
                        for positions in self.front_door_buffer[ids].dropna().values:
                            self.human_id_dataframe = self.human_id_dataframe.append({ids: positions}, ignore_index=True)
                            # self.save_pose(positions)
                        self.last_position[ids]=[ids_pose]                        
                    else:
                        dataframe_of_ids_index_list = self.front_door_buffer[ids].dropna().index.values
                        length_dataframe_index = len(dataframe_of_ids_index_list)
                        if length_dataframe_index + 4000 < self.step - self.buffer_dict[ids] or length_dataframe_index > 24000 :
                                front_process_flag.append(ids)
                                self.eject(ids)      
            if  front_process_flag:
                for ids in front_process_flag:
                    self.front_door_buffer = self.front_door_buffer.drop(ids, axis=1)
                    self.front_door_buffer_last_pose = self.front_door_buffer_last_pose.drop(ids, axis=1)
                    # self.buffer_list.remove(ids)
                    del self.buffer_dict[ids]
                    # if ids not in self.new_human:
                    #     self.eject(ids)   


            for ids in self.back_door_buffer.columns.values:
                # if ids !=None:
                    ids_pose = self.back_door_buffer_last_pose[ids]
                    if ids_pose[1] > 0.60 and 1.8<=ids_pose[0]<=3.1:
                        # self.new_human.append(ids)
                
                        if self.replacement.keys():
                            for back_missing_id in self.replacement.keys():
                                if ids in self.replacement[back_missing_id]:
                                    self.reset_missing_buffer(ids)
                                replacement_list.append(back_missing_id)

                        if replacement_list:
                            for elem in replacement_list:
                                del self.replacement[elem]
                        back_process_flag.append(ids)
                        self.human_id = np.concatenate((self.human_id, [ids]), axis=0)
                        # self.human_id_dataframe = pd.concat([self.human_id_dataframe, self.front_door_buffer], axis=1)
                        for positions in self.back_door_buffer[ids].dropna().values:
                            self.human_id_dataframe = self.human_id_dataframe.append({ids: positions}, ignore_index=True)
                            # self.save_pose(positions)
                        self.last_position[ids]=[ids_pose]                        
                    else:
                        dataframe_of_ids_index_list = self.back_door_buffer[ids].dropna().index.values
                        length_dataframe_index = len(dataframe_of_ids_index_list)
                        if length_dataframe_index +4000 < self.step - self.buffer_dict[ids] or length_dataframe_index > 24000 :
                                back_process_flag.append(ids)
                                self.eject(ids)
                                            
            if  back_process_flag:
                    for ids in back_process_flag:
                        self.back_door_buffer = self.back_door_buffer.drop(ids, axis=1)
                        self.back_door_buffer_last_pose = self.back_door_buffer_last_pose.drop(ids, axis=1)
                        # self.buffer_list.remove(ids)
                        del self.buffer_dict[ids]
                        # if ids not in self.new_human:

                                

        # print 'front door buff : ', self.front_door_buffer    
                          
    def save_pose(self, pose):
        try:
            self.pose = np.concatenate((self.pose,[pose]))
        except:
            self.pose = np.array([pose])

    def remove_from_buffer(self, ids):
        del self.buffer_dict[ids]
        self.buffer_list.remove(ids)
        self.front_door_buffer = self.front_door_buffer.drop(ids, axis=1)
        self.eject(ids)
        # print 'Deleted ' + str(ids)
       
    def add_to_human_dataframe(self, ids, pose):
        self.human_id_dataframe = self.human_id_dataframe.append({ids: pose}, ignore_index=True)
        if ids in self.last_position.columns.values:
            for length in range(len(self.last_position)):
                        for n in range(len(self.last_position.loc[length,ids])):                       
                            self.last_position.loc[length,ids][n]=pose[n]
        else:
            self.last_position[ids] = [pose]
        
        try:
            if ids in self.missing_id_dict.keys():
                self.reset_missing_buffer(ids)                
        except:
            pass

    def reset_missing_buffer(self, ids):
        del self.missing_id_dict[ids]
        self.missing_human_id = np.delete(self.missing_human_id, np.where(self.missing_human_id == ids)[0][0], axis=0)
        if self.missing_human_id.shape[0] == 0:
            self.missing_flag = False
        # try:
        #     del self.missing_found_id_dict[ids]
        # except:
        #     pass
        # try:
        #     self.missing_found_id_dataframe = self.missing_found_id_dataframe.drop(ids, axis=1)
        # except:
        #     pass
    
    def refresh_missing_list(self):
        # refresh_flag = False
        if self.new_id.keys():
            missing_id = self.new_id.keys()[-1]
            # if missing_id in self.missing_id_dict.keys():
            #     missing_id = missing_id           
            self.reset_missing_buffer(missing_id)

        if self.missing_found_id_dict.keys():
            missing_id = self.missing_found_id_dict.keys()[-1]
            # if missing_id in self.missing_id_dict.keys():
            #     missing_id = missing_id
            self.reset_missing_buffer(missing_id)
        
    def add_to_missing(self, ids,last_pose, big_room = True, small_room = False, server_room= False):
        if ids not in self.missing_id_dict.keys():
            self.missing_id_dict[ids] = OrderedDict() 
            if big_room:self.missing_id_dict[ids]['location'] = 'big_room'          
            elif small_room: self.missing_id_dict[ids]['location'] = 'small_room'
            elif server_room:self.missing_id_dict[ids]['location'] ='server_room'
            self.missing_id_dict[ids]['step'] = self.step
            self.missing_id_dict[ids]['last_position'] = last_pose                                    
            try:
                self.missing_human_id = np.concatenate((self.missing_human_id, [ids]), axis=0)                            
            except:
                self.missing_human_id = np.array([ids])           
        self.missing_flag = True
    
    def update_human_id(self):
        if self.new_id.keys():
            for ids in self.new_id.keys():
                self.human_id= np.concatenate((self.human_id, [self.new_id[ids][0]]), axis=0)
                self.add_to_human_dataframe(self.new_id[ids][0], self.new_id[ids][1])
                # self.delete_from_missing(ids)        
        if self.missing_found_id_dict.keys():
            for ids in self.missing_found_id_dict.keys():
                self.human_id= np.concatenate((self.human_id, [self.missing_found_id_dict[ids][0]]), axis=0)
                self.add_to_human_dataframe(self.missing_found_id_dict[ids][0], self.missing_found_id_dict[ids][1])
                # self.delete_from_missing(ids)
        # for ids in self.new_human:
        #     self.human_id= np.concatenate((self.human_id, [ids]), axis=0)
            # self.add_to_human_dataframe(ids, pose)
            
        # self.new_id = OrderedDict()

    def delete_from_missing(self,ids):
        self.missing_human_id = np.delete(self.missing_human_id , np.where(self.missing_human_id == ids)[0], axis= 0 )
        if self.missing_human_id.shape[0] == 0:
            self.missing_flag = False
    
        
    def check_found_missing(self):
        key_to_delete = []
        if self.missing_found_id_dict.keys():
            for ids in self.missing_found_id_dict.keys():
                if self.step - self.missing_found_id_dict[ids]['step'][0] > 10:
                    if len(self.missing_found_id_dict[ids]['id']) == 1:
                        self.delete_from_missing(ids)
                        del self.missing_id_dict[ids]
                        new_id = self.missing_found_id_dict[ids]['id'][0]
                        self.human_id = np.concatenate((self.human_id,[new_id]))
                        for positions in self.missing_found_id_dataframe[new_id].dropna().values:
                            self.human_id_dataframe = self.human_id_dataframe.append({new_id: positions}, ignore_index=True)
                        self.last_position[new_id] = [positions]
                        key_to_delete.append(ids)
                    # else:
                    #     self.check_motion(self.)


        
            

if __name__ == '__main__':
    Tracking().follow_id()
    # os.system("taskset -p 0xff %d" % os.getpid())
    # print affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)
    # print affinity.get_process_affinity_mask(0)
    # p = psutil.Process(os.getpid())
    # print p.cpu_affinity()
    # lock = Lock()
    # P1 = Process(target= Tracking().follow_id, args=(lock,))
    # P1.start()
    # p = psutil.Process()
    # cpu_num = len(p.get_cpu_affinity())
    # all_cpu = list(range(cpu_num -2))
    # print all_cpu
    # p.set_cpu_affinity(range(cpu_num -2))
    # t1= Process(target= Tracking().follow_id)
    # t2= Process(target= Tracking().follow_id)
    # t3= Process(target= Tracking().follow_id)
    # t4= Process(target= Tracking().follow_id)
    # t5= Process(target= Tracking().follow_id)
    # t6= Process(target= Tracking().follow_id)
    # t7= Process(target= Tracking().follow_id)
    # t8= Process(target= Tracking().follow_id)
    # t9= Process(target= Tracking().follow_id)
    # t10= Process(target= Tracking().follow_id)
    # t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    # t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join();

    # Tracking().follow_id() 
    # so = ['poi/']
    # print so[0][:-1] + 'hello'
    # array = np.array(['4','6','6'])
    # if '5' in array: print True
    # a = 1
    # if a in {1,2}:
    #     print a


    # a = np.array([[5, 4]]) 
    # a = np.concatenate((a, [[5,8],[7,8]]))
    # # print a
    # b = DataFrame()
    # b = b.append({'c': np.array([5,6])}, ignore_index=True)
    # b = b.append({'d': np.array([3,4])}, ignore_index=True)
    # for pose in np.array([[7,2],[7,8],[4,5]]):
    #     b = b.append({'c': pose }, ignore_index=True)
    
    
    # c = b['c'].dropna()
    # for pose in c.values:
    #     try:
    #         k = np.concatenate((k,[pose]))
    #     except:
    #         k = np.array([pose])
    # k = np.delete(k, range(0,k.shape[0]) , axis=0)
    
    # print k
    # a = np.delete(a,0,axis=0)
    # print a
    # print np.where(a==4)[0][0]
    # a = np.delete(a,np.where(a==5)[0][0],axis=0)
    # print a.shape[0]
    # a = np.concatenate((a, [6]), axis=0) 
    # a  = np.delete(a,[0,1],axis=0 )
    # print a.shape[0]
    # print a.shape[0]
    # pbrint np.bool([5])[0]

    # for val in a :
    #     print val
    # a = DataFrame({ 'a':[[5,4]],'b':[[6,7]] })
    # a = DataFrame({ 'a':[[5,4]],'b':[[6,7]]})
    # replace_dict ={'a': [[6,8]]}
    # # a['a'].replace({5:6}, inplace= True)
    # for keys in range(len(a)):
    #     for i in range(len(a.loc[keys,'a'])):
    #         a.loc[keys,'a'][i] =  replace_dict['a'][0][i]
    # a['c'] = [[5,6]]
    # print a['c'][0][0]

    # a = [1,2,3,4,5,6]

    # for elem in a :
    #     if elem ==3:
    #         pass
    #     else:
    #         print elem
    #     print 'a'

    # a = a.append({'a': [5,5]}, ignore_index=True)
    # print a
    # a = a.append({'c': [6,6]}, ignore_index=True)
    # a['c'] = [[5,7]]
    # print a

    # a = OrderedDict()
    # a['a'] = 50
    # del a['a']
    # print a

    # a = a.append({'a': [8,10]}, ignore_index= True)
    # a = a.append({'b': [8,16]}, ignore_index= True)
    # a = a.append({'b': [8,16]}, ignore_index= True)
    # a = a.append({'b': [8,16]}, ignore_index= True)
    # c= DataFrame({'c': [[5,6],[5,8],[5,7],[9,6],[6,9],[89,6]]})
    # c = c.append({'d': [8,16]}, ignore_index= True)
    # print c['c'].dropna().values[0]
    # c = c.drop('c', axis = 1)
    # print c.columns.values

    # print c
    # a = pd.concat([a,c], axis=1)
    # a = pd.merge(a,c)
    # print a
    # print c.loc[[c.index.values[-1]],'d']
    # print c['d'][6][0]
    # c = c.drop('c', axis=1)
    # q = c['c'].dropna()
    # print q
    # print c['c'].dropna()[c['c'].dropna().index.values[-1]]

#!/usr/bin/env	python
import os
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

tracking_folder = '/home/barry/hdd/tracking/'
# day_folder = ['21','22', '23', '24', '25', '26', '28', '29', '30']
# day_folder = ['12-4/','12-5/','12-6/', '12-6/','12-7/','12-8/', '12-8/']
day_folder = ['12-/11', '12-12/', '12-12/', '12-13/', '12-13/', '12-14/', '12-14/']
# folder_name = ['noon','noon','morning','noon','noon','morning', 'noon']
folder_name = ['noon', 'morning', 'noon', 'morning', 'noon', 'morning', 'noon']
filename = ['/HumanPositions/Real-time Data.csv']
# start_line_list = [310721,9639,16315, 181277, 4178,744, 531]
start_line_list = [5850, 2248, 404, 4555, 896, 2079, 2190]
# id_start = [np.array(['14151100']),np.array(['12033900']),np.array(['21164400']), np.array(['12000201']) , np.array(['9140600']), np.array(['18270100','18271300', '18285400']), np.array(['12145100'])]
id_start = [np.array(['10185900']), np.array(['19174400']), np.array(['10552702']), np.array(
    ['22015600']), np.array(['10293500', '10294300']), np.array(['20221302', '20221304', '20221303', '20221400', '20221402']), np.array(['11054500', '11155600'])]


# pose_start = [np.array([[4.285, 1.939]]), np.array([[1.746,7.105]]),np.array([[3.903,2.426]]), np.array([[4.233,1.964]]), np.array([[1.832, 7.163]]),np.array([[3.953,1.806],[0.165,6.538], [1.722,-1.702]]), np.array([[0.438,5.007]])]

pose_start = [np.array([[1.506, 6.404]]), np.array([[2.619, 4.879]]), np.array([[0.614, 5.095]]), np.array([[2.721, 4.895]]), np.array(
    [[0.212, 5.322], [-1.079, 5.257]]), np.array([[3.953, 1.806], [0.165, 6.538], [1.722, -1.702]]), np.array([[1.328, -3.250], [0.140, 5.323]])]
# save_path = [tracking_folder + 'dataframe/' + str(i)+ '.csv' for i in range(len(day_folder))]
save_path = [tracking_folder + 'id/' + str(day_folder[i][:-1])+'_'+folder_name[i] +'.npy'
                 for i in range(len(day_folder))]
save_csv = [tracking_folder + 'id/' + str(day_folder[i][:-1])+'_'+folder_name[i] +'.csv'
                 for i in range(len(day_folder))]
pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
entrance_corners_x_limit = [3.2, 4.3]
entrance_corners_y_limit = [1.8, 3.2]
# entrance_corners_x_limit = [2.4, 4.2]
# entrance_corners_y_limit = [1.0, 4.2]
class Tracking():
    def __init__(self):
       
        self.centers=[]
       
    def get_center(self):
        '''
        Returns a list of centers position list
        ---
        self.centers = [[0.5,0.5],[1.0,1.0],...]
        '''
        x_start, y_start = corners[0][0], corners[0][1]
        self.centers.append([x_start + grid_size / 2.0, y_start + grid_size / 2])
        center_start = self.centers[0]
        list_x, list_y = deque(), deque()
        for [x, y] in corners:
            list_x.append(x)
            list_y.append(y)
        increment = 0.0
        while(self.centers[-1][1] < max(list_y)):
             while(self.centers[-1][0] < max(list_x)):
                 val = [self.centers[-1][0] + grid_size, self.centers[-1][1]]
                 if (val[0] < max(list_x)):
                     self.centers.append(val)
                 else:
                    break
             if self.centers[-1][1] + grid_size < max(list_y):
                 self.centers.append(
                     [self.centers[0][0], self.centers[-1][1] + grid_size])
             else:
                break
        del list(list_x)[:]
        del list(list_y)[:]
        return self.centers

    def get_rectangle_corners(self):
        '''
        Returns a list of centers coordinates and a dictionnary of corners coordinates 
        --- 
        self.corners = {'[0.5,0.5]':[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],...}
        '''
        self.corners = OrderedDict()
        self.get_center()
        for center in self.centers:
            self.corners[str(center)] = [[center[0] - grid_size / 2, center[1] - grid_size / 2],
                                         [center[0] + grid_size / 2,
                                             center[1] - grid_size / 2],
                                         [center[0] + grid_size / 2,
                                             center[1] + grid_size / 2],
                                         [center[0] - grid_size / 2, center[1] + grid_size / 2]]
        return self.corners

    def get_grid_tracking_frequency(self):              
        self. get_rectangle_corners()
        print 'Get grid frequency'
         
        
        for files in os.listdir('/home/barry/hdd/tracking/id-12-22/np/'):
            pose = np.load(files)  
            tracking_number = pose.shape[0]
            self.grid = OrderedDict()
            for key_center in self.corners.keys():
                print 'cell:  ', self.corners.keys().index(key_center)
                
                x_list = []
                y_list = []
                data_used_list = []    
                self.grid[key_center] = 0
                for value in self.corners[key_center]:
                    x_list.append(value[0])
                    y_list.append(value[1])
                
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)
    
                for i in range(tracking_number):    
                    if (pose[i][0] > x_min and pose[i][0] < x_max) and (pose[i][1] > y_min and pose[i][1] < y_max):                      
                            self.grid[key_center] +=1
                #             data_used_list.append(i)                
                # # pose = np.delete(pose, data_used_list,0)
            # print self.grid.values()
            csvwriter('/home/barry/hdd/tracking/id-12-22/csv/' + files[:-4] + '.csv',
                      headers=['grid_freq_val'],
                       rows=[self.grid.values()])
    
    def get_corners(self):
        self.get_rectangle_corners()
        data = csvread(pose_filename)
        self.corn = OrderedDict()
        for str_pose in data['pose']:
            self.corn[str_pose] = self.corners[str_pose]
       
        return self.corn           
    
    def follow_id(self):
            # self.filename = [tracking_folder + day_folder[i] + folder_name[i] + filename[0] for i in range(len(day_folder))]
            self.filename = ['/home/mtb/Desktop/Real-time Data.csv']
            # print save_path
            # for k in range(0, len(self.filename)-3): 
            for k in range(len(day_folder)):  
                with open(self.filename[0], 'r') as csvfile:
                    self.data = OrderedDict()
                    csvreader = csv.reader(csvfile)                
                    iterations = 0
                    time_list_unique = []                   
                    # start_line = start_line_list[k] 
                    # self.id_human = id_start[k]   
                    # self.human_pose = pose_start[k] 

                    start_line = start_line_list[3] 
                    self.id_human = id_start[3]   
                    self.human_pose = pose_start[3] 

                    
                    self.dataframe =DataFrame()
                    self.missing_dataframe = OrderedDict()
                    self.last_position = DataFrame()
                    self.buffer_dataframe = DataFrame()
                    self.buffer_diction = OrderedDict()
                    self.buffer_list = []
                    self.id_to_check_entry_exit = []
                    # self.missing_human_id = OrderedDict()
                    
                    self.missing = False
                    self.eject_flag = False
                    self.buffer_id = []
                    self.human_flag = []
                    self.individual = []
                    self.new_id = []
                    self.new_pose = []
                    self.number_of_human = self.id_human.shape[0]
                   
                    try:
                        self.pose = np.delete(self.pose, range(0, self.pose.shape[0]), axis=0)
                    except:
                        pass
                    try:
                        self.missing_human_id = np.delete(self.missing_human_id, range(
                            0, self.missing_human_id.shape[0]), axis=0)
                    except:
                        pass
                    
                    try:
                        self.eject_id = np.delete(self.eject_id, range(
                            0, self.eject_id.shape[0]), axis=0)
                    except:
                        pass
                    for i in range(self.number_of_human):
                        # self.dataframe[self.id_human[i]] =[self.human_pose[i]]   
                        try:
                            self.pose = np.concatenate((self.pose, [self.human_pose[i]])) 
                        except:
                            self.pose = np.array([self.human_pose[i]])

                        self.last_position[self.id_human[i]] = [self.human_pose[i]]
                    # print self.last_position
                    self.step = 0
                    for row in csvreader:
                        iterations +=1
                        self.step +=1
                        # print iterations
                        # if iterations >=start_line:   
                        if iterations >=start_line:
                            # if  iterations > stop_line_debug   :
                            #      break          
                            if float(row[0]) not in time_list_unique:
                                # print self.data.keys(), iterations
                                # if len(self.data.keys())  == 2:
                                    # self.check_human_presence()
                            
                                self.data = OrderedDict()                                
                                self.data[float(row[0])] =OrderedDict()
                                                           
                                del time_list_unique[:]
                                time_list_unique.append(float(row[0]))
                                id_list = []
                                coordinates =[]
                            self.data[float(row[0])]['id'] = list()
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
                                                # if str(row[i + 1]) in self.id_human:
                                                self.data[float(row[0])]['id'].append(str(row[i+1]))  
                                                self.data[float(row[0])]['pose'].append(np.array(pose))
                                                # self.dataframe = self.dataframe.append({str(row[i+1]): pose}, ignore_index=True)                                                                    
                                        except :
                                            break 
                                    self.fill_the_list()
                        # if  iterations > 400000:
                        #     sys.exit(0)
                        else:
                            pass                         
                        print str(self.filename[k]) + ': ' + str(iterations) 

        
                # np.save(save_path[k], self.save_pose)
                np.save(save_path[k], self.pose)
                csvwriter(save_csv[k], ['id'], self.individual)
    
    def add_to_human_list(self, ids):
        if ids not in self.individual:
            self.individual.append(ids)
        

    def fill_the_list(self):
        
        self.time_key = self.data.keys()[-1]
        # print self.data[self.time_key]['id']
        id_quit_the_room = []
        for present_ids in self.id_human: 
            self.add_to_human_list(present_ids)   
            # Present in the room       
            if present_ids in self.data[self.time_key]['id']:                
                # It's not lost still in the room
                # print self.data[self.time_key]['pose'][self.data[self.time_key]['id'].index(present_ids)]
                # self.dataframe = self.dataframe.append({present_ids: self.data[self.time_key]['pose'][self.data[self.time_key]['id'].index(present_ids)]}, ignore_index=True)
                self.save_pose(self.data[self.time_key]['pose'][self.data[self.time_key]['id'].index(present_ids)])
                #If the missing id just reappeared, delete it from the missing list
                try:
                    if present_ids in self.missing_human_id:
                        del self.missing_dataframe[present_ids]
                        self.missing_human_id = np.delete(self.missing_human_id, np.where(self.missing_human_id == present_ids)[0][0], axis=0)
                except:
                    pass

                    # Register the last positions
                for length in range(len(self.last_position)):
                    for n in range(len(self.last_position.loc[length,present_ids])):                       
                        self.last_position.loc[length,present_ids][n]=self.data[self.time_key]['pose'][self.data[self.time_key]['id'].index(present_ids)][n]
                
                
            else:
                #Lost either sitted in the room or left it

                # First check where it was first seen                                                           
                last_position = self.last_position[present_ids][0]
                # print present_ids, 'last position: ', last_position
                # print last_position
                if last_position[0] > 4.27:# Has Exited                    
                    index = np.where(self.id_human == present_ids)[0][0]
                    id_quit_the_room.append(index)                     
                else:  
                     #Sitted somewhere  
                    if present_ids not in self.missing_dataframe.keys():
                            self.missing_dataframe[present_ids] = OrderedDict() 
                            self.missing_dataframe[present_ids]['id'] = OrderedDict()
                            # print self.missing_dataframe                        
                    try:
                        if not len(np.where(self.missing_human_id == present_ids)[0]) ==1:
                            self.missing_human_id =   np.concatenate((self.missing_human_id, [present_ids]), axis=0)                            
                    except:
                        self.missing_human_id = np.array([present_ids])                     
                    self.missing = True

        if id_quit_the_room:
            self.id_human = np.delete(self.id_human, id_quit_the_room, axis=0) 

        # Check the buffer  for door or human
        # self.check_buffer(self.data[self.time_key]['id'],self.data[self.time_key]['pose'])      

        # For the new detected ids check if it's a new person or the substituant to the missing person
        for detected_ids in self.data[self.time_key]['id']:            
            if not len(np.where(self.id_human == detected_ids)[0]) == 1  :
                # print 'Check for: ', detected_ids
                indice = self.data[self.time_key]['id'].index(detected_ids)
                try:
                    # print 'try'
                    # pass if in the eject list
                    if detected_ids in self.eject_id :
                        # print str(detected_ids) +  ' already in eject'
                        pass
                                   
                    #update the buffer if already there
                    elif detected_ids in self.buffer_list:
                        # print 'adding to buffer'         
                        self.buffer(detected_ids, self.data[self.time_key]['pose'][indice])
                    # elif detected_ids in self.identifiable:
                    #     pass         
                    else:
                        self.detect_in_room(detected_ids, self.data[self.time_key]['pose'][indice])
                except:  
                    # print 'find in room'                 
                    self.detect_in_room(detected_ids, self.data[self.time_key]['pose'][indice])
            else:
                pass
       

        key_to_del = []
        # try:            
        #     if self.identifiable.shape[0]>0:
        if self.missing_dataframe.keys():
            try:
                for missing_id in self.missing_dataframe.keys():
                    # if len(self.missing_dataframe[missing_id]['id'].keys())==1:                    
                        index = np.where(self.id_human == missing_id)[0][0]
                        new_id = self.missing_dataframe[missing_id]['id'].keys()[0]
                        # indice = self.missing_dataframe.keys().index(id_keys)
                        self.id_human = np.delete(self.id_human, [index], axis=0)
                        self.id_human = np.concatenate((self.id_human, [new_id]), axis=0)
                        # self.dataframe = self.dataframe.append({new_id:self.missing_dataframe[missing_id]['id'][new_id][0]}, ignore_index=True)
                        self.save_pose(self.missing_dataframe[missing_id]['id'][new_id][0])
                        self.last_position[new_id]= [self.missing_dataframe[missing_id]['id'][new_id][0]]
                        self.missing_human_id = np.delete(self.missing_human_id, np.where(self.missing_human_id == missing_id)[0][0], axis=0)
                        key_to_del.append([ missing_id,new_id])
            except:
                pass
        if key_to_del:
            for [old, new] in key_to_del:
                del self.missing_dataframe[old]        
        try:
            if self.missing_human_id.shape[0] == 0:
                self.missing = False
        except:
            pass
        # print self.id_human
        # # print self.missing_human_id
        # # print self.eject_id
        # print self.missing_dataframe
        # print self.dataframe
        # print ' Human present: ', self.id_human
        # try:
        #     print ' Human missing: ', self.missing_human_id
            
        # except:
        #     pass
        # print ' Human eject: ', self.eject_id
        # print 'Buffer: ' , self.buffer_list
        
        # print self.dataframe.columns
        # print self.missing_dataframe 
        print 'Shape : ' + str(self.pose.shape)

    def detect_in_room(self,detected_ids, pose):               
        # First Check if it's the corresponding missing one:
        if self.missing:           
            print 'Checking the missing' 
            for m in range(self.missing_human_id.shape[0]):
                detected_ids_pose =  pose        
                missing_last_position = self.last_position[self.missing_human_id[m]][0]

                if self.check_for_neighbor_missing(self.missing_human_id[m],detected_ids,detected_ids_pose, missing_last_position): pass #The missing person then halt
                else: # there is a person missing but this detected ids doesn't seem to match the missing id position . So CHECK IF IT IS THE DOOR OR A NEW PERSON
                    if self.check_for_new_entrance(detected_ids, pose):
                        pass
                    
                    else: # It is not neither a missing person nor a new person nor a door so a defective detected id .SO EJECT IT
                            
                            self.eject(detected_ids)    

        # if not just move check if it is a new personn or the door being open or close
        else:
            
            
            # No missing personn so check if the corresponding is either a door or a human to keep it in a buffer

            # Buffer it if between the boundaries of the entrance 
            if self.check_for_new_entrance(detected_ids, pose):
                pass
                # print 'Added to Buffer'          
           # A random id Put it in a eject list   
            else:
                
                # print 'ejecting: ', detected_ids
                self.eject(detected_ids)
        
    def check_for_neighbor_missing(self, missing_id,detected_ids, detected_ids_pose, missing_last_position):
        neighbor_boundary = 0.9
        if fabs(detected_ids_pose[0] - missing_last_position [0])<= neighbor_boundary and fabs(detected_ids_pose[1] - missing_last_position[1]) <= neighbor_boundary:
            # print missing_id, detected_ids, fabs(detected_ids_pose[0] - missing_last_position [0]), fabs(detected_ids_pose[1] - missing_last_position[1])
            # if 'id' not in self.missing_dataframe[missing_id].keys():
            #     self.missing_dataframe[missing_id]['id'] =  OrderedDict()    

            if detected_ids not in  self.missing_dataframe[missing_id]['id'].keys():           
                self.missing_dataframe[missing_id]['id'][detected_ids] =[detected_ids_pose]
            
            else:
                self.missing_dataframe[missing_id]['id'][detected_ids].append(detected_ids_pose)
                
                                
            # else:
            #     if detected_ids not in self.missing_dataframe[missing_id]['id'].keys():
            #         # self.missing_dataframe[self.missing_human_id[m]]['id'].append(detected_ids)
            #         self.missing_dataframe[missing_id]['id'][detected_ids].append(detected_ids_pose)            
            # try:
            #     if not len(np.where(self.identifiable == detected_ids)[0]) ==1:
            #         self.identifiable = np.concatenate((self.identifiable, [detected_ids]), axis=0)
            # except:
            #     self.identifiable= np.array([detected_ids])
            # return True

        else:
            # Outside the neighbor missing
             return False
        
    
    def eject(self, detected_ids):                           
        try:
            if not  len(np.where(self.eject_id == detected_ids)[0]) == 1:
                self.eject_id = np.concatenate((self.eject_id, [detected_ids]), axis=0)
        except:
            self.eject_id = np.array([detected_ids])  
        
        # print 'Ejected :   ', self.eject_id     
        
        # self.eject_dataframe = self.eject_dataframe.append({detected_ids: })

    def check_for_new_entrance(self, detected_ids,pose):
        # Check for the new entry exit
        if entrance_corners_x_limit[0] <=pose[0]<=entrance_corners_x_limit[1] and\
            entrance_corners_y_limit[0] <=pose[1]<=entrance_corners_y_limit[1]:
            # print 'Door is either opened or the person just get in. Save in the buffer'

            # Put it in the buffer for further analysis in the next steps            
            self.buffer(detected_ids,pose)   
            return True       
        else:
            return False
    
    def buffer(self, detect_ids, pose ):       
        if detect_ids not in self.buffer_list:
            self.buffer_diction[detect_ids] = self.step
            self.buffer_list.append(detect_ids)
            self.buffer_dataframe = self.buffer_dataframe.append({detect_ids: pose}, ignore_index=True)
            # self.buffer_dict[detect_ids]['pose'] =[pose]
            # self.buffer_dict[detect_ids]['step'] =[self.step, self.step]         
        else:
            self.buffer_dataframe = self.buffer_dataframe.append({detect_ids: pose}, ignore_index=True)
            if pose[0] < 3.1:
                self.id_human = np.concatenate((self.id_human, [detect_ids]), axis=0)
                # self.dataframe = pd.concat([self.dataframe, self.buffer_dataframe], axis=1)
                for positions in self.buffer_dataframe[detect_ids].dropna().values:
                    self.save_pose(positions)
                self.last_position[detect_ids]=[pose]
                self.buffer_dataframe = self.buffer_dataframe.drop(detect_ids, axis=1)
                self.buffer_list.remove(detect_ids)
                del self.buffer_diction[detect_ids]
            else:
                dataframe_of_ids_index_list = self.buffer_dataframe[detect_ids].dropna().index.values
                length_dataframe_index = len(dataframe_of_ids_index_list)

                if length_dataframe_index > 100:
                    # The door has disappeared
                    if length_dataframe_index +2000 < self.step - self.buffer_diction[detect_ids] :
                        self.remove_from_buffer(detect_ids)                    
                    #The door gets stuck in the screen
                    if length_dataframe_index > 3000:
                        self.remove_from_buffer(detect_ids)  
    def save_pose(self, pose):
            # print self.pose
        try:
            self.pose = np.concatenate((self.pose,[pose]))
        except:
            self.pose = np.array([pose])
        # print  self.pose.shape[0]
                        

                
   
        # print self.buffer_dataframe
    
    def remove_from_buffer(self, ids):
        del self.buffer_diction[ids]
        self.buffer_list.remove(ids)
        self.buffer_dataframe = self.buffer_dataframe.drop(ids, axis=1)
        self.eject(ids)
        # print 'Deleted ' + str(ids)


if __name__ == '__main__':
    Tracking().follow_id() 
    # so = ['poi/']
    # print so[0][:-1] + 'hello'




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

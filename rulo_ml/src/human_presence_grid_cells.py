#!/usr/bin/env	python
import os
import sys
import numpy as np
import csv
import pandas as pd
from pandas import Series, DataFrame
from collections import deque, OrderedDict
from math import fabs
from rulo_utils.csvwriter import csvwriter
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

# front_entrance_x = [3.2, 4.5]
# front_entrance_y = [1.3, 2.9]
front_entrance_x = [3.6, 4.5]
front_entrance_y = [1.5, 2.3]

# back_entrance_x = [2.2, 3.1]
# back_entrance_y = [0.0, 1.0]

back_entrance_x = [2.2, 3.1]
back_entrance_y = [0.0, 0.8]

front_server_entrance_x = [2.0, 3.0]
front_server_entrance_y = [4.8, 5.1]

back_server_entrance_x = [1.3, 2.0]
back_server_entrance_y = [7.2, 8.0]

time_list = [1512590402.0]
maxsize = 12000


class IdExtractor:
    def get_id(self):
        self.filename = ['/home/mtb/Desktop/12-6_noon.csv']        
        for k in range(1):
            self.chunk = pd.read_csv(filepath_or_buffer=self.filename[0], index_col=[0, 1],
                                     header=[0, 1], chunksize=maxsize)
            
            # self.id = [id_start[k][0]]
            self.id = [id_start[3][0]]
            self.human = [id_start[3][0]]
            self.eject = []
            self.missing_dict = OrderedDict()
            self.missing_flag = False
            self.buffer = []   
            self.id_timestamp = OrderedDict()   
            self.replacement = OrderedDict()
            self.time = np.array([])
            self.last_id_dataframe = DataFrame()
            # try:
            #     self.pose = np.delete(self.pose, range(0, self.pose.shape[0]), axis=0)
            # except:
            #     pass  
            step = 0
            for chunk in self.chunk:                
                step +=1
                print self.filename[0], ' step: ', step
                self.process(chunk)  
            csvwriter(id_save_path,headers=self.replacement.keys(), row=self.replacement.values())         

    def process(self,chunk):          
        if self.buffer:
            self.data =pd.concat([self.buffer[-1], chunk])     
            del self.buffer[:]            
        else:
            self.data = chunk  
            for ids_start in self.id:
                self.id_timestamp[ids_start] = self.data[ids_start].dropna(how='all').index.values[0]
                self.replacement[ids_start] = [ids_start]
        time_start = self.data.index.values[0]   
        # print self.data.index.vales[0]
        # print self.id_timestamp
        # print 'start: ', time_start
        if len(self.data.index.values) >= maxsize:
            self.process_chunk(time_start,indice=2000,terminate = False)
        else:
            self.process_chunk(time_start,indice=-1, terminate=True)
        
    def process_chunk(self, time_start,indice=2000,terminate = False):
        for line_index in self.data.index[:-indice]:   
            id_list_step = self.data.loc[line_index].dropna(how='all').index.get_level_values(0).drop_duplicates(keep='first')
            id_to_remove = []
            for ids in self.id:
                if ids in id_list_step:
                    pass
                    # print 'next: ',np.where(self.data.index[0] == line_index[0])
                else:
                    # if self.time_list[-1] - line_index[0]>=2.0:
                        if self.data.loc[line_index[0]:, ids].dropna(how='all').empty:
                            # self.missing_dict[ids]=[]
                            # print 'missing: ', ids
                            p_x = self.data.loc[previous_index,ids].dropna(how='all')['x']
                            p_y = self.data.loc[previous_index,ids].dropna(how='all')['y']
                            pose = [round(p_x, 4), round(p_y, 4)]
                            # print pose
                            if pose[0] >=4.27 and  1.5<=pose[1]<=2.8:
                                # print 'Quit via the front door'
                                id_to_remove.append(ids) 
                            elif pose[1] <= 0.6 and 2.1 <= pose[0] <= 3.0: 
                                # print 'Quit via the back door'
                                id_to_remove.append(ids) 
                            elif 4.9<=pose[1]<=5.4 and 2.1 <= pose[0] <= 2.9:
                                # print 'Quit via the front server door'
                                id_to_remove.append(ids)
                            elif pose[1] >= 7.5 and 1.3 <= pose[0] <= 2.0:
                                # print 'Quit via the back server door'
                                id_to_remove.append(ids)
                            elif -2.2 <= pose[0] <= 0.9 and 1.0 <= pose[1] <= 2.9: 
                                self.missing_dict[ids] =OrderedDict()
                                self.missing_dict[ids]['last_location'] = 'sofa'
                                self.missing_dict[ids]['last_pose'] = pose
                                id_to_remove.append(ids)
                                # print 'sofa'
                            else:
                                self.missing_dict[ids] = OrderedDict()
                                self.missing_dict[ids]['last_location'] = 'big_room'
                                self.missing_dict[ids]['last_pose'] = pose
                                id_to_remove.append(ids)
                                # print 'big room'
                        else:
                            pass
                    # else:
                    #     self.reexamine.append(ids)
            
            if id_to_remove:
                for elem in id_to_remove:
                    self.id.remove(elem)
            self.new_id = []
            self.new_eject =[]
            self.replaced = []
            for detected_ids in id_list_step:
                self.found_ids = []
                if detected_ids not in self.id and detected_ids not in self.eject:
                    # print detected_ids                        
                    if not self.missing_dict.keys():
                        p_x = self.data.loc[line_index, detected_ids].dropna(how='all')['x']
                        p_y = self.data.loc[line_index, detected_ids].dropna(how='all')['y']
                        pose = [round(p_x, 4), round(p_y, 4)]
                        if self.check_front_door_entrance(detected_ids,pose):pass
                        elif self.check_back_door_entrance(detected_ids,pose):pass
                        elif self.check_front_server_entrance(detected_ids,pose):pass
                        elif self.check_back_server_entrance(detected_ids,pose):pass                       
                        else: self.new_eject.append(detected_ids)
                    if self.missing_dict.keys():
                        # print 'missing: ', self.missing_dict
                        missing_found_flag = False
                        p_x = self.data.loc[line_index, detected_ids].dropna(how='all')['x']
                        p_y = self.data.loc[line_index, detected_ids].dropna(how='all')['y']
                        detected_pose = [round(p_x, 4), round(p_y, 4)]
                        for missing_ids in self.missing_dict.keys():
                            if self.missing_dict[missing_ids]['last_location'] == 'big_room':
                                if self.check_person_neighbor(missing_ids, self.missing_dict[missing_ids]['last_pose'],detected_ids,detected_pose):
                                    self.found_ids.append(missing_ids)
                                    missing_found_flag = True
                                    break
                            elif self.missing_dict[missing_ids]['last_location'] == 'sofa':
                                if self.check_sofa_neighbor(missing_ids, detected_ids, detected_pose,line_index):
                                    self.found_ids.append(missing_ids)
                                    missing_found_flag = True
                                    break
                        if not missing_found_flag:
                            if self.check_front_door_entrance(detected_ids,detected_pose):pass
                            elif self.check_back_door_entrance(detected_ids,detected_pose):pass
                            elif self.check_front_server_entrance(detected_ids,detected_pose):pass
                            elif self.check_back_server_entrance(detected_ids,detected_pose):pass                       
                            else: self.new_eject.append(detected_ids)                     
                if self.found_ids:
                    del self.missing_dict[self.found_ids[0]]
            if self.new_id:
                for elem in self.new_id:
                    if elem not in self.id:
                        self.id.append(elem)
            if self.new_eject:
                for elem in self.new_eject:
                    if elem not in self.eject:
                        self.eject.append(elem)
            previous_index = line_index
            # print self.time
        # self.get_time(time_start, previous_index,indice)
        if not terminate:
            self.buffer.append(self.data.iloc[-indice:])
        else:
            self.buffer = []

    def get_time(self, chunk_time_start, chunk_time_end,indice):
        buffer_list=[]
        print chunk_time_start
        print chunk_time_end
        print self.replacement
        print self.last_id_dataframe
        for ids_keys, ids_val in self.replacement.items():
            buffer_list.append(ids_val[-1])
            if ids_keys in self.last_id_dataframe.columns.values:
                dataframe =  pd.concat([self.last_id_dataframe[ids_keys],self.data.loc[self.data.index.values[:-indice],ids_val].dropna(how='all')])
            else:
                dataframe =  self.data.loc[self.data.index.values[:-indice],ids_val].dropna(how='all')
            time_list = dataframe.index.values
            print time_list
            last_recorded_time = dataframe.index.values[-1]
            for miss_ids in ids_val:
                dataframe =  self.data.loc[self.data.index.values[:-indice],miss_ids].dropna(how='all')   
                for d in range(dataframe.index.values.shape[0]):
                    try:         
                        self.time = np.concatenate((self.time,[(time_list[1][0] - time_list[0][0]) * 1000.0 + (time_list[1][1] - time_list[0][1])]), axis=0)
                        try:
                            self.pose = np.concatenate((self.pose, [[round(dataframe.loc[time_list[0], 'x'], 4),
                                                                round(dataframe.loc[time_list[0], 'y'], 4)]]), axis=0)
                        except:
                            self.pose = np.array([[round(dataframe.loc[time_list[0], 'x'], 4),
                                                              round(dataframe.loc[time_list[0], 'y'], 4)]])
                    except:
                        pass
                    time_list = np.delete(time_list,[0], axis = 0)
            
            print self.data[miss_ids].dropna(how='all')
            # last_pose = [round(self.data[miss_ids].dropna(how='all').loc[:-indice]['x'],4), 
            #              round(self.data[miss_ids].dropna(how='all').loc[:-indice ]['y'], 4)]
            # print last_pose
        #     self.last_id_dataframe[miss_ids] = DataFrame(data=[last_pose], index=[[last_recorded_time[0]],[last_recorded_time[1]]], columns=[[miss_ids, miss_ids], ['x', 'y']])
        # print self.last_id_dataframe
        # self.replacement = OrderedDict() 
        # for ids_buff in buffer_list:
        #     self.replacement[ids_buff]=[ids_buff]    
          
           
    def check_front_door_entrance(self, ids,pose):
        if front_entrance_x[0]<=pose[0]<=front_entrance_x[1] and\
           front_entrance_y[0] <= pose[1] <= front_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 3.0)[0]) > 0 and \
              len(np.where(self.data[ids].dropna(how='all')['y'].values >= 2.8)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
               self.id_timestamp[ids] = self.data[ids].dropna(how='all').index.values[0]
               self.replacement[ids]=[ids]
               return True
            else:
                return False
        else:
            return False
        
    def check_back_door_entrance(self,ids,pose):
        if back_entrance_x[0] <= pose[0] <= back_entrance_x[1] and\
           back_entrance_y[0] <= pose[1] <= back_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 1.7)[0]) > 0 and \
              len(np.where(self.data[ids].dropna(how='all')['y'].values >= 1.5)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
               self.id_timestamp[ids] = self.data[ids].dropna(how='all').index.values[0]
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False
    
    def check_front_server_entrance(self,ids ,pose):
        if front_server_entrance_x[0] <= pose[0] <= front_server_entrance_x[1] and\
           front_server_entrance_y[0] <= pose[1] <= front_server_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 1.6)[0]) > 0 and \
              len(np.where(self.data[ids].dropna(how='all')['y'].values <= 4.2)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
               self.id_timestamp[ids] = self.data[ids].dropna(how='all').index.values[0]
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False
    
    def check_back_server_entrance(self,ids,pose):
        if back_server_entrance_x[0] <= pose[0] <= back_server_entrance_x[1] and\
           back_server_entrance_y[0] <= pose[1] <= back_server_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 0.6)[0]) > 0 and \
              len(np.where(self.data[ids].dropna(how='all')['y'].values <= 6.8)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
               self.id_timestamp[ids] = self.data[ids].dropna(how='all').index.values[0]
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False
    
    def check_person_neighbor(self,missing_ids, missing_pose, detected_ids, detected_pose):
        if fabs(detected_pose[0] -missing_pose[0]) <=1.0 and fabs(detected_pose[1] - missing_pose[1])<= 1.0:
            self.new_id.append(detected_ids)
            self.human.append(detected_ids)
            self.id_timestamp[detected_ids] = self.data[detected_ids].dropna(how='all').index.values[0]
            for keys, values in self.replacement.items():
                if missing_ids in values:
                    self.replacement[keys].append(detected_ids)
                    break                
            return True
        else:
            return False
    
    def check_sofa_neighbor(self, missing_ids, detected_ids, detected_pose, line_index):
        sofa_dict = OrderedDict() 
        sofa_dict['x']= self.data.loc[line_index:, detected_ids].dropna(how='all')['x'].values
        sofa_dict['y']= self.data.loc[line_index:, detected_ids].dropna(how='all')['y'].values
        if -2.4<=detected_pose[0]<=1.1 and 0.6<=detected_pose[1]<=3.8:
            if len(np.where(sofa_dict['x'] >= 1.0)[0]) > 0 or len(np.where(sofa_dict['y'] <= 0.6)[0])>0 or\
                len(np.where(sofa_dict['y'] >= 4.0)[0])>0:
                self.new_id.append(detected_ids)
                self.human.append(detected_ids)
                self.id_timestamp[detected_ids] = self.data[detected_ids].dropna(how='all').index.values[0]
                for keys, values in self.replacement.items():
                    if missing_ids in values:
                        self.replacement[keys].append(detected_ids)
                        break
                return True
            else:
                return False            
        else:
            return False

if __name__ == '__main__':
    IdExtractor().get_id()

    # a = DataFrame(data=[[5,2]], index =[['a'],[1]],columns=[['id','id'],['x','y']])
    # print a

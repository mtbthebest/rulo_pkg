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
maxsize = 32000


class IdExtractor:
    def get_id(self):
        self.filename = ['/home/mtb/Desktop/12-6_noon.csv']
        for k in range(1):
            self.chunk = pd.read_csv(filepath_or_buffer=self.filename[0], index_col=[0, 1], header=[0, 1], chunksize=maxsize)

            # self.id = [id_start[k][0]]
            self.id = [id_start[3][0]]
            self.human = [id_start[3][0]]
            self.eject = []
            self.missing_dict = OrderedDict()
            self.missing_flag = False
            self.buffer = []
            self.replacement = OrderedDict()
            
            self.time = np.array([])            
            try:
                self.pose = np.delete(self.pose, range(0, self.pose.shape[0]), axis=0)
            except:
                pass
            step = 0
            for chunk in self.chunk:
                step += 1
                print self.filename[0], ' step: ', step
                self.process(chunk)
                # if step >2:
            #         np.save('/home/mtb/pose.npy', self.pose)
            #         np.save('/home/mtb/time.npy', self.time)
            #         sys.exit(0)
            # csvwriter(id_save_path, headers=self.replacement.keys(), row=self.replacement.values())

    def process(self, chunk):        
        if self.buffer:
            self.data = pd.concat([self.buffer[-1], chunk])
            del self.buffer[:]
        else:
            self.data = chunk
            for ids_start in self.id:
                self.replacement[ids_start] = [ids_start]
        time_start = self.data.index.values[0]
        # print 'Time start: ', time_start
        if len(self.data.index.values) >= maxsize:
            self.process_chunk(time_start, indice=2000, terminate=False)
        else:
            self.process_chunk(time_start, indice=1, terminate=True)
   
    def process_chunk(self, time_start, indice=2000, terminate=False):               
        self.id_time_stamp = OrderedDict()
        self.disappearance = OrderedDict()
        id_list_step = self.data[:-indice].dropna(how='all', axis=1).columns.get_level_values(0).drop_duplicates(keep='first')
        analyze_lim_indice = self.data[:-indice].index.values
        for ids in id_list_step:
            self.id_time_stamp[ids] = [float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[0][0]) + float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[0][1]) / 1000.0,
                                       float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[-1][0])+ float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[-1][1] )/ 1000.0]
            if ids in self.id:
                self.disappearance[ids] = float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[-1][0])+ \
                                          float(self.data.loc[analyze_lim_indice, [ids]].dropna(how='all').index.values[-1][1] )/ 1000.0
        
        for detected_ids in self.id_time_stamp:
            self.new_id = []
            self.new_eject = []
            self.remove=[]
            self.found_ids = OrderedDict()
            if detected_ids not in self.id and detected_ids not in self.eject:
                id_miss_flag = False
                for present_ids in self.disappearance:                    
                    if self.id_time_stamp[detected_ids][0] > self.disappearance[present_ids]:
                        state = self.get_state(present_ids)
                        if  state == 'sofa' or state == 'big_room':
                            id_miss_flag = True

                if not id_miss_flag:
                    #New person or door or hanging
                    p_x,p_y = self.data[detected_ids].dropna(how='all').iloc[0]['x'], self.data[detected_ids].dropna(how='all').iloc[0]['y']
                    pose = [round(p_x,4), round(p_y,4)]
                    if self.check_front_door_entrance(detected_ids, pose):
                        pass
                    elif self.check_back_door_entrance(detected_ids, pose):
                        pass
                    elif self.check_front_server_entrance(detected_ids, pose):
                        pass
                    elif self.check_back_server_entrance(detected_ids, pose):
                        pass
                    else:
                        self.new_eject.append(detected_ids)
                elif id_miss_flag:
                    replacement = False
                    detected_pose = [round(self.data[detected_ids].dropna(how='all').iloc[0]['x'], 4),
                                     round(self.data[detected_ids].dropna(how='all').iloc[0]['y'], 4)]
                    for missing_ids in self.missing_dict:
                        missing_pose = self.missing_dict[missing_ids]['last_position']                        
                        if self.missing_dict[missing_ids]['last_location'] == 'big_room':
                            if self.check_person_neighbor(missing_ids, missing_pose, detected_ids, detected_pose):
                                replacement = True
                                self.found_ids[missing_ids] = detected_ids
                                break
                        elif self.missing_dict[missing_ids]['last_location'] == 'sofa':
                            if self.check_sofa_neighbor(missing_ids, detected_ids, detected_pose):
                                replacement = True
                                self.found_ids[missing_ids] = detected_ids
                                break
                    if not replacement:
                        if self.check_front_door_entrance(detected_ids, detected_pose):
                            pass
                        elif self.check_back_door_entrance(detected_ids, detected_pose):
                            pass
                        elif self.check_front_server_entrance(detected_ids, detected_pose):
                            pass
                        elif self.check_back_server_entrance(detected_ids, detected_pose):
                            pass
                        else:
                            self.new_eject.append(detected_ids)

            if self.new_id:
                for elem in self.new_id:
                    if elem not in self.id:
                        self.id.append(elem)
                        self.update_disappearance(elem,analyze_lim_indice)
            if self.remove:
                for rmv_ids in self.remove:
                    if rmv_ids in self.id:
                        self.id.remove(rmv_ids)
                    if rmv_ids in self.disappearance:
                        del self.disappearance[rmv_ids]                    
                
            if self.found_ids.keys():
                new_id = self.found_ids.values()[0]
                old_id = self.found_ids.keys()[0]
                self.id.append(new_id)
                self.id.remove(old_id)
                # print self.disappearance.keys()
                # print self.id
                # if self.found_ids.keys()[0] in self.id:
                #     self.id.remove(self.found_ids.keys()[0])
                if old_id in self.disappearance:
                        del self.disappearance[old_id]
                del self.missing_dict[old_id]
                self.update_disappearance(new_id,analyze_lim_indice)
                
            if self.new_eject:
                for eject_ids in self.new_eject:
                    self.eject.append(eject_ids)
        
        self.get_pose_and_time_cell(analyze_lim_indice)
        self.reset()
        if not terminate:
            self.buffer.append(self.data[-indice-1:])
        else:
            self.buffer = []
        
    def check_front_door_entrance(self, ids, pose):
        if front_entrance_x[0] <= pose[0] <= front_entrance_x[1] and\
           front_entrance_y[0] <= pose[1] <= front_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 3.0)[0]) > 0 or \
               len(np.where(self.data[ids].dropna(how='all')['y'].values >= 2.8)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
            #    self.update_disappearance(ids, analyze_lim_indice)
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False

    def check_back_door_entrance(self, ids, pose):
        if back_entrance_x[0] <= pose[0] <= back_entrance_x[1] and\
           back_entrance_y[0] <= pose[1] <= back_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 1.7)[0]) > 0 or \
               len(np.where(self.data[ids].dropna(how='all')['y'].values >= 1.2)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
            #    self.update_disappearance(ids, analyze_lim_indice)
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False

    def check_front_server_entrance(self, ids, pose):
        if front_server_entrance_x[0] <= pose[0] <= front_server_entrance_x[1] and\
           front_server_entrance_y[0] <= pose[1] <= front_server_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 1.6)[0]) > 0 or \
               len(np.where(self.data[ids].dropna(how='all')['y'].values <= 4.4)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
            #    self.update_disappearance(ids, analyze_lim_indice)
               self.replacement[ids] = [ids]
               return True
            else:
                return False
        else:
            return False

    def check_back_server_entrance(self, ids, pose):
        if back_server_entrance_x[0] <= pose[0] <= back_server_entrance_x[1] and\
           back_server_entrance_y[0] <= pose[1] <= back_server_entrance_y[1]:
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 0.6)[0]) > 2 and \
               len(np.where(self.data[ids].dropna(how='all')['y'].values <= 6.8)[0]) > 0:
               self.new_id.append(ids)
               self.human.append(ids)
               self.replacement[ids] = [ids]
            #    self.update_disappearance(ids, analyze_lim_indice)
               return True
            else:
                return False
        else:
            return False

    def check_person_neighbor(self, missing_ids, missing_pose, detected_ids, detected_pose):
        if fabs(detected_pose[0] - missing_pose[0]) <= 1.0 and fabs(detected_pose[1] - missing_pose[1]) <= 1.0:
            # self.new_id.append(detected_ids)
            self.human.append(detected_ids)
            for keys, values in self.replacement.items():
                if missing_ids in values:
                    self.replacement[keys].append(detected_ids)
                    break
            return True
        else:
            return False

    def check_sofa_neighbor(self, missing_ids, detected_ids, detected_pose):
        if -2.4 <= detected_pose[0] <= 1.1 and 0.6 <= detected_pose[1] <= 3.8:
            if len(np.where(self.data[detected_ids].dropna(how='all')['x'].values >= 1.2)[0]) > 0 or \
                len(np.where(self.data[detected_ids].dropna(how='all')['y'].values <= 0.6)[0]) > 0 or\
                len(np.where(self.data[detected_ids].dropna(how='all')['y'].values >= 4.0)[0]) > 0:
                # self.new_id.append(detected_ids)
                self.human.append(detected_ids)
                for keys, values in self.replacement.items():
                    if missing_ids in values:
                        self.replacement[keys].append(detected_ids)
                        break
                return True
            else:
                return False
        else:
            return False
    
    def get_state(self,ids):
        position = [round(self.data[ids].dropna(how='all').iloc[-1]['x'],4),
                    round(self.data[ids].dropna(how='all').iloc[-1]['y'], 4)]
    
        if position[0] >= 4.27 and 1.5 <= position[1] <= 2.8:
                                # print 'Quit via the front door'
            # id_to_remove.append(ids)
            self.remove.append(ids)
            return 'leave'
        elif position[1] <= 0.6 and 2.1 <= position[0] <= 3.0: 
            # print 'Quit via the back door'
            # id_to_remove.append(ids)
            self.remove.append(ids)
            return 'leave'
        elif 4.9<=position[1]<=5.4 and 2.1 <= position[0] <= 2.9:
            # print 'Quit via the front server door'
            # id_to_remove.append(ids)
            self.remove.append(ids)
            return 'leave'
        elif position[1] >= 7.5 and 1.3 <= position[0] <= 2.0:
            # print 'Quit via the back server door'
            # id_to_remove.append(ids)
            self.remove.append(ids)
            return 'leave'
        elif -2.2 <= position[0] <= 0.9 and 1.0 <= position[1] <= 2.9: 
            if ids not in self.missing_dict:
                self.missing_dict[ids] =OrderedDict()
                self.missing_dict[ids]['last_location'] = 'sofa'
                self.missing_dict[ids]['last_position'] = position
            # id_to_remove.append(ids)
            return 'sofa'
            # print 'sofa'
        else:
            if ids not in self.missing_dict:
                self.missing_dict[ids] = OrderedDict()
                self.missing_dict[ids]['last_location'] = 'big_room'
                self.missing_dict[ids]['last_position'] = position
            # id_to_remove.append(ids)
            return 'big_room'

    def update_disappearance(self, ids,analyze_lim_indice):
        self.disappearance[ids] = float(self.data.loc[analyze_lim_indice, [ids]].dropna(
             how='all').index.values[-1][0]) + float(self.data.loc[analyze_lim_indice, [ids]].dropna(
             how='all').index.values[-1][1]) / 1000.0
    
    def get_pose_and_time_cell(self,analyze_lim_indice):
        # print 'replacement: ', self.replacement
        for ids, ids_list in self.replacement.items():
            for s in range(len(ids_list)):
                dataframe = self.data.loc[analyze_lim_indice,ids_list[s]].dropna(how='all')
                # print dataframe.values
                time_list = dataframe.index.values
                for d in range(dataframe.index.values.shape[0]):
                    try:
                        self.time = np.concatenate((self.time, [(
                                     time_list[d+1][0] - time_list[d][0]) * 1000.0 + (time_list[d+1][1] - time_list[d][1])]), axis=0)
                        try:
                             self.pose = np.concatenate((self.pose, [[round(dataframe['x'].iloc[d], 4),
                                                                     round(dataframe['y'].iloc[d], 4)]]), axis=0)

                        except:
                            self.pose = np.array([[round(dataframe['x'].iloc[d], 4),  round(dataframe['y'].iloc[d], 4)]])
                    except:
                        try:
                            next_time =self.data.loc[analyze_lim_indice,ids_list[s+1]].dropna(how='all').index.values[0]
                            self.time = np.concatenate((self.time, [(next_time[0] - time_list[d][0]) * 1000.0 + (next_time[1] - time_list[d][1])]), axis=0)
                            self.pose = np.concatenate((self.pose, [[round(dataframe['x'].iloc[d], 4),
                                                                     round(dataframe['y'].iloc[d], 4)]]), axis=0)
                            # print 'a: ',self.time[-1], self.pose[-1]
                        except:
                            last_time = time_list[d]
                            last_pose = [round(dataframe['x'].iloc[d], 4),round(dataframe['y'].iloc[d], 4)]
            # print last_pose
            # print last_time
            # print self.time[-1], self.pose[-1]
            # np.save(filename, self.pose)
            # np.save(filename, self.time)
            # print self.time[0], self.pose[0]
            # print self.time[-1], self.pose[-1]
            # print self.pose.shape, self.time.shape
        try:
            self.pose = np.delete(self.pose, range(
                0, self.pose.shape[0]), axis=0)
        except:
            pass
        
        print self.pose, self.time.shape
    def reset(self):
        # remove_id = []
        if self.id:
            for ids in self.id:
                self.replacement[ids] = [ids]

    def get_center(self):
        '''
        Returns a list of centers position list
        ---
        self.centers = [[0.5,0.5],[1.0,1.0],...]
        '''
        self.centers = []
        x_start, y_start = corners[0][0], corners[0][1]
        self.centers.append(
            [x_start + grid_size / 2.0, y_start + grid_size / 2])
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

        self.get_rectangle_corners()
        # pose = np.load(tracking_file)
        print 'Get grid frequency'
        # pose = [self.arrange.values()[i][j] for i in range(len(self.arrange.values()))
        #                                     for j in range(len(self.arrange.values()[i]))]
        # list_to_examine = pose
        # print pose[0]
        # print len(pose)
        # pose.remove(pose[0])
        # print len(pose)
        # init =   self.arrange.shape[0]
        # init_list = []

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
                            self.grid[key_center] += 1
                #             data_used_list.append(i)
                # # pose = np.delete(pose, data_used_list,0)
            # print self.grid.values()
            csvwriter('/home/barry/hdd/tracking/id-12-22/csv/' + files[:-4] + '.csv',
                      headers=['grid_freq_val'],
                      rows=[self.grid.values()])

if __name__ == '__main__':
    IdExtractor().get_id()
    # a = np.load('/home/mtb/pose.npy')
    # b = np.load('/home/mtb/time.npy')
    # print a[9998:10004]
    # print b[9998:10004]
    # path = ['/home/mtb/Desktop/test']
    if not os.path.isdir(path[0]):
        os.mkdir(path[0])
   

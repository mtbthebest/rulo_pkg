#!/usr/bin/env	python
import numpy as np
import csv
import pandas as pd
from pandas import Series, DataFrame
from collections import deque, OrderedDict
from math import fabs

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

front_entrance_x = [3.2, 4.5]
front_entrance_y = [1.3, 2.9]

back_entrance_x = [2.2, 3.1]
back_entrance_y = [0.0, 1.0]

front_server_entrance_x = [2.0,3.0]
front_server_entrance_y = [4.8,5.1]


back_server_entrance_x = [1.2,2.0]
back_server_entrance_y = [7.0,8.0]

time_list = [1512590402.0]

class IdExtractor:
    def get_id(self):
        self.filename = ['/home/mtb/Desktop/12-6_noon.csv']
        for k in range(1):
            self.chunk = pd.read_csv(filepath_or_buffer= self.filename[0],index_col=[0,1], header=[0,1],chunksize=11000)
            self.id_list = []
            # self.id = [id_start[k][0]]
            self.id = [id_start[3][0]]
            self.human = self.id
            self.eject = []
            self.missing_dict =OrderedDict()
            self.missing_flag = False
            # self.candidate = OrderedDict()
            #     self.candidate['server_room'] = OrderedDict()
        #     self.candidate['big_room'] = OrderedDict()
            for chunk in self.chunk:
                self.data = chunk
                break
            # ids_start = id_start[k][0]
            ids_start = id_start[3][0]
            time_start = time_list[0]#1512590402.0
            
            # print self.data[self.data.columns.values[72]].loc[time_start]
            # print self.data['12000201'][['x','y']].dropna(axis=1, how='all').loc[time_start,649.0]['x'] + 1
            # print self.data.loc[time_start,ids_start]['x'][649.0]
            # self.data.columns.get_level_values(0).drop_duplicates(keep ='first')
            # self.id_list=[] np.where(self.data.columns.get_level_values(0).drop_duplicates(keep ='first') == ids_start)[0][0]
            # print b
            # print self.data.columns.values
            # start_index = self.data[self.data]
            if not self.id_list:
                for j in range(np.where(self.data.columns.get_level_values(0) == ids_start)[0][0],len(self.data.columns.values)):
                    if self.data.columns.values[j][0] not in self.id_list:
                        self.id_list.append(self.data.columns.values[j][0])
      
            self.time_list = self.data.index.get_level_values(0).drop_duplicates(keep='first')
            finish_row = self.time_list[self.time_list.shape[0] - 1]
        
    
    #     # # print np.where(self.time_list == 1512586042.0)[0][0]
            exit = False
    #     # # print self.data.iloc[-1,:].columns.values
    #     flag = 0
    #     
    #     # print self.time_list
        for times in self.time_list[:-2]:
            self.new_id = []
            self.new_eject =[]
            
            if times >= time_start:
                present_time = times
                dataframe = self.data.loc[times].dropna(axis=1, how='all')
                detected_ids_list= dataframe.columns.get_level_values(0).drop_duplicates(keep='first')
                # print detected_ids_list
    #             # print dataframe
    #             # if flag == 1:
    #             #     exit =True
                remove_id =[]
                for present_ids in self.id:
                    if present_ids in detected_ids_list:
                        pass
                    else:  
                        pass
    #                     print present_ids, present_time            
                        present_time_indice = np.where(self.time_list == times)[0][0]
                        if self.data.loc[self.time_list[present_time_indice], present_ids].dropna(how='all').empty:
    #                         #CHECK IF HE HAS NOT LEFT THE ROOM 
                            pose_x, pose_y= self.data.loc[last_time][present_ids].iloc[-1]
                            pose =[round(pose_x,4),round(pose_y,4)]
                            if self.check_front_door_exit(pose) or self.check_back_door_exit(pose) or \
                                self.check_front_server_exit(pose) or self.check_back_server_exit(pose):
                                    pass
                            else:
                                #HAS NOT LEFT THE ROOM EITHER IN THE SERVER ROOM OR IN THE BIG ROOM
                                self.missing_flag = True
                                self.missing_dict[present_ids]= OrderedDict()                                
                                self.missing_dict[present_ids]['last_pose']= pose
                                self.missing_dict[present_ids]['time']= last_time
                                self.missing_dict[present_ids]['time_indice'] = present_time_indice
                                if  -2.5 <=self.missing_dict[present_ids]['last_pose']<= 0.95 and    0.89<=missing_last_pose[1]<=3.1:
                                    self.missing_dict[present_ids]['last_location'] = 'sofa'
                                else:
                                    self.missing_dict[present_ids]['last_location'] = 'big_room'
                            remove_id.append(present_ids)
                            
                        else:
                           
                            pass
                
                if remove_id:
                    for ids_garb in remove_id:
                        self.id.remove(ids_garb)  
                print self.missing_dict
    #             #Not missing just check the front door or the back door                
                if not self.missing_flag:
                    for detected_ids in detected_ids_list:                                                
                        if detected_ids not in self.id and detected_ids not in self.eject:                 
                            if self.check_front_door_entrance(detected_ids,dataframe):pass
                                                                       
                            elif self.check_back_door_entrance(detected_ids, dataframe):pass

                            elif self.check_front_server_entrance(detected_ids, dataframe):pass

                            elif self.check_back_server_entrance(detected_ids, dataframe):pass
                                                     
                            else: self.new_eject.append(detected_ids)         

                if self.missing_flag:
                    # print self.missing_dict
                    for detected_ids in detected_ids_list:
                        self.match_find = OrderedDict()
                        if detected_ids not in self.id and detected_ids not in self.eject:                 
                            ids_pose_x, ids_pose_y =  self.data[detected_ids].dropna(how='all').iloc[0]
                            ids_pose = [round(ids_pose_x,4), round(ids_pose_y,4)]
                            if self.missing_dict.keys():                             
                                for missing_ids in self.missing_dict:
                                    missing_last_pose = self.missing_dict[missing_ids]['last_pose'] 
                                    # print self.data[detected_ids].dropna(how='all').iloc[0]['x'] 
                                    if self.missing_dict[missing_ids]['last_location'] == 'sofa':
                                        if self.check_near_sofa_neighbor(detected_ids,missing_ids): break
                                    elif self.missing_dict[missing_ids]['last_location'] == 'big_room':
                                        print self.missing_dict
                                        if self.check_person_neighbor(missing_ids, missing_last_pose,detected_ids,ids_pose,present_time):
                                            print missing_ids, detected_ids
                                            break
                            
                            if not self.match_find.keys():
                                if self.check_front_door_entrance(detected_ids,dataframe):pass
                                                                       
                                elif self.check_back_door_entrance(detected_ids, dataframe):pass

                                elif self.check_front_server_entrance(detected_ids, dataframe):pass

                                elif self.check_back_server_entrance(detected_ids, dataframe):pass
                                                        
                                else: self.new_eject.append(detected_ids)
                        
                        if self.match_find.keys():
                            del self.missing_dict[self.match_find.keys()[0]]
    #         #Update present human at that time
            self.id = self.id + self.new_id
            self.eject = self.eject + self.new_eject
            if not self.missing_dict.keys():
                self.missing_flag = False                
            last_time = times
            # print self.human
            print dataframe.columns.values
    #         # print self.missing_dict, present_time
    #         # print self.id, self.eject,detected_ids_list,self.missing_flag
          
   
    
    def check_front_door_entrance(self,ids,dataframe):        
        if front_entrance_x[0] <= dataframe[ids].iloc[0]['x'] <= front_entrance_x[1] and\
                front_entrance_y[0] <= dataframe[ids].iloc[0]['y'] <= front_entrance_y[1]:
                #Front door check if his x_Pose is above 3.1
            if len(np.where(self.data[ids].dropna(how='all')['x'].values <= 3.0)[0])==0:
                #If not near the door put it to eject else put it to human id
                return False
            else:
                # It is a new human save it
                self.new_id.append(ids)
                self.human.append(ids)
                return True
        else:
            return False

    def check_back_door_entrance(self,ids,dataframe):
        if back_entrance_x[0] <= dataframe[ids].iloc[0]['x'] <= back_entrance_x[1] and\
                back_entrance_y[0] <= dataframe[ids].iloc[0]['y'] <= back_entrance_y[1]:
                #Front door check if his y_Pose is above 0.7
            if not np.where(self.data[ids].dropna(how='all')['y'].values >= 0.7)[0]:
                #If not near the door put it to eject else put it to human id
                return False
            else:
                # It is a new human save it
                self.new_id.append(ids)
                self.human.append(ids)
                return True
        else:
            return False
    
    def check_front_server_entrance(self,ids, dataframe):
        if front_server_entrance_x[0] <= dataframe[ids].iloc[0]['x'] <= front_server_entrance_x[1] and\
                front_server_entrance_y[0] <= dataframe[ids].iloc[0]['y'] <= front_server_entrance_y[1]:
                #Front door check if his y_Pose is above 0.7
            array_x =  self.data[ids].dropna(how='all')['x'].values
            array_y = self.data[ids].dropna(how='all')['y'].values
            front_server_flag = False
            for p_y in array_y:
                p_x = array_x[np.where(array_y == p_y)[0][0]]
                if fabs(p_y - array_y[0])>=1.2 or fabs(array_x[0] - p_x)>=1.2 :
                    front_server_flag = True
                    break            
            if front_server_flag:
                self.new_id.append(ids)
                self.human.append(ids)
                return True
            else:
                return False
        else:
            return False
                 
    def check_back_server_entrance(self,ids, dataframe):
        if back_server_entrance_x[0] <= dataframe[ids].iloc[0]['x'] <= back_server_entrance_x[1] and\
                back_server_entrance_y[0] <= dataframe[ids].iloc[0]['y'] <= back_server_entrance_y[1]:
                #Front door check if his y_Pose is above 0.7
            array_x =  self.data[ids].dropna(how='all')['x'].values
            array_y = self.data[ids].dropna(how='all')['y'].values
            back_server_flag = False
            for p_y in array_y:
                p_x = array_x[np.where(array_y == p_y)[0][0]]
                if fabs(p_y - array_y[0]) >=1.0 or fabs(array_x[0] - p_x)>=1.0 :
                    back_server_flag = True
                    break            
            if back_server_flag:
                self.new_id.append(ids)
                self.human.append(ids)
                return True
            else:
                return False
        else:
            return False 

    def check_front_door_exit(self,ids_pose):
        if ids_pose[0] >= 4.270 and 1.476 <= ids_pose[1] <= 2.37:
            #Quit the room via the front door
            return True
        else:
            return False
    
    def check_back_door_exit(self,ids_pose):
        if ids_pose[1]<= 0.6 and  back_entrance_x[0]<=ids_pose[0]<=back_entrance_x[1]:
            #Quit the room via the back door
            return True
        else:
            return False
    
    def check_front_server_exit(self,ids_pose):
        if ids_pose[1]>= 4.95 and front_server_entrance_x[0]<=ids_pose[0]<=front_server_entrance_x[1]:
            #Quit the room via the back door
            return True
        else:
            return False

    def check_back_server_exit(self,ids_pose):
        if ids_pose[1]>= 7.3 and 1.3<=ids_pose[0]<=back_server_entrance_x[1]:
            #Quit the room via the back door
            return True
        else:
            return False

    def check_near_sofa_neighbor(self, ids,missing_ids):
        pose_x, pose_y =  self.data[ids].dropna(how='all').iloc[0]
        if -2.5<=pose_x<=1.0 and  0.89<=pose_y<=3.1:
            array_x =  self.data[ids].dropna(how='all')['x'].values
            array_y = self.data[ids].dropna(how='all')['y'].values
            sofa_flag = False
            for p_y in array_y:
                p_x = array_x[np.where(array_y == p_y)[0][0]]
                if fabs(p_y - array_y[0])>=1.5 or fabs(array_x[0] - p_x)>=1.5 :
                    sofa_flag = True
                    break            
            if sofa_flag:
                self.new_id.append(ids)
                self.human.append(ids)
                self.match_find[missing_ids] = ids
                return True
            else:
                return False
        else:
            return False
            
    def check_person_neighbor(self,missing_ids, missing_last_pose,ids,ids_pose,present_time):        
        if fabs(ids_pose[0] - missing_last_pose[0]) <=0.8 and fabs(ids_pose[1] - missing_last_pose[1])<=0.8:
            # if len(self.data.loc[present_time].dropna(how='all', axis=1).columns.get_level_values(0).drop_duplicates(keep='first')):
                self.human.append(ids)
                self.new_id.append(ids)
                self.match_find[missing_ids] = ids
                return True
        else:
            return False
            

        
if __name__ == '__main__':
    IdExtractor().get_id()
    # x= np.array([1,4,2,3,5,6])
   
    


    # # n = np.where(a==b)
    # # print n

    # c= np.where(x<6)[0]
    # d = np.where(x>2)[0]
    # print c,d
    # max_shape = max(c.shape, d.shape)
    # # print max_shape
    # res = []
    # for i in range(max_shape[0]):
    #     if c[i] in d:

    #         res.append(c[i])
    # y = np.array([1,2,3,4,5,6])
    # for elem in res:
       
      
    #     if len(np.where(y[elem]<6)[0])== 1 and \
    #      len(np.where(y[elem]>4)[0])==1:
    #         print elem
            
    # # print np.bool(l)
    # # for elem in c:
    # #     if elem in d:
    # #         print True
    # # print k
    # # print np.bool(k.all())

    

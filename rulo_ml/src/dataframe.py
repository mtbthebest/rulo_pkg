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
save_path = [tracking_folder + 'dataframe/' + str(i)+ '.csv' for i in range(len(day_folder))]
save_path = [tracking_folder + 'id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.npy'
             for i in range(len(day_folder))]
save_csv = [tracking_folder + 'id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.csv'
            for i in range(len(day_folder))]


id_save_path = [tracking_folder + 'id-12-24/id/' + str(day_folder[i][:-1]) + '_' + folder_name[i] + '.npy'
                for i in range(len(day_folder))]
pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
# entrance_corners_x_limit = [3.2, 4.5]
# entrance_corners_y_limit = [1.3, 2.9]

front_entrance_x = [3.2, 4.5]
front_entrance_y = [1.3, 2.9]

back_entrance_x = [2.2, 3.1]
back_entrance_y = [0.0, 1.0]
# entrance_corners_x_limit = [2.4, 4.2]
# entrance_corners_y_limit = [1.0, 4.2]


class Extraction:
    def dataframe(self):
          
            # self.filename = [tracking_folder + day_folder[i] + folder_name[i] + filename[0] for i in range(len(day_folder))]
            self.filename = ['/home/mtb/Desktop/Real-time Data.csv']
            
            # print save_path
            # for k in range(0, len(self.filename)-3):
            for k in range(1):
                step_print = 1
                with open(self.filename[k], 'r') as csvfile:
                # with open(self.filename[0], 'r') as csvfile:
                   
                    csvreader = csv.reader(csvfile)
                    time_list_unique = []
                    # start_line = start_line_list[k]
                    # self.human_id = id_start[k]
                    # self.human_pose = pose_start[k]

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
                                try:
                                    # print self.data
                                    self.rewrite()
                                    # index = [[time_list_unique[-1]]*len(self.data['time_us']), self.data['time_us']]
                                    # b = DataFrame(self.dict, index=index)
                                    # self.buffer_list.append(b)
                                except:
                                    pass
                               
                                self.data = OrderedDict()
                                self.data['time_us'] = []
                                self.data['time_s'] = [float(row[0])]
                                self.data['id'] = []
                                self.data['pose'] = OrderedDict()
                                del time_list_unique[:]
        
                                time_list_unique.append(float(row[0]))              
                            
                            if int(row[2]) > 0:    # Number of people
                                    for elem in row:
                                        if elem == '':
                                            row.remove(elem)
                                    start = 4
                                    step = 7
                                    num_tracked = int(row[2])
                                    if float(row[1]) not in self.data['time_us']:
                                        self.data['time_us'].append(float(row[1]))
                                    for i in np.linspace(start, len(row), num_tracked + 1, dtype=np.int32):
                                        try:
                                                        
                                                x = (
                                                    float(row[i + 3]) / float(1000))
                                                y = - \
                                                    (float(
                                                        row[i + 2]) / float(1000)) 
                                                                                     
                                                if str(row[i + 1]) not in self.data['id']:
                                                    self.data['id'].append(str(row[i + 1]))   
                                                    self.data['pose'][str(row[i + 1])] = OrderedDict()                    
                                                self.data['pose'][str(row[i + 1])][float(row[1])] = [x,y]
                                                
                                                # print self.data['pose']
                                                # print x,y
                                                # print str(row[i+1])                           
                                        except:
                                            break  
                                    
                                
                                     
                                                                    
                    
                        # if self.step >= start_line +1500:
                        #     # print len(self.buffer_list)
                        #     a = pd.concat(self.buffer_list,join='outer')
                        #     print a['12000100']
                        #     a.to_csv('/home/mtb/Desktop/dat.csv' )
                        #     sys.exit(0)
                        
                        if step_print * 100000 < self.step:
                            print str(self.filename[k]) + ': ' + str(self.step)
                            step_print +=1
                            # print self.id
                        # print str(self.filename[k]) + ': ' + str(self.step)
                        # print len(self.buffer_list)
                self.dataframe = pd.concat(self.buffer_list,join='outer')
                self.dataframe.to_csv('/home/mtb/Desktop/barry.csv')

                      
    def rewrite(self):
        self.dict = OrderedDict()
        for ids in self.data['id']:
            self.dict[ids] = []
            for times in self.data['time_us']:
                if times in self.data['pose'][ids]:
                    self.dict[ids].append(self.data['pose'][ids][times])
                else:
                     self.dict[ids].append(np.nan) 
        res = {}
        # dataframe = DataFrame()
        for ids in self.data['id']:
        #     res[ids]= Series(data=self.dict[ids], index=[self.data['time_s'] * len(self.data['time_us']),
        #    self.data['time_us']], name=ids)
            res[ids] = DataFrame(data=self.dict[ids], index=[self.data['time_s'] * len(self.data['time_us']),
                                                      self.data['time_us']], columns=[[ids, ids], ['x', 'y']])
        
        t_concat = pd.concat(res.values(), axis=1)
        
        # print t_concat['12000100']['x']
        # print t_concat
        # print t_concat['12000201']['x']
        # dataframe = DataFrame(t_concat,columns=[self.dict.keys(),[]])
        # print dataframe
        # self.buffer_list.append(DataFrame(t_concat))
        # print self.data['time_us']      
        self.buffer_list.append(t_concat)
    

if __name__ == '__main__':
    Extraction().dataframe()
    # result = OrderedDict([('time_us', [649.0, 679.0, 698.0, 723.0, 748.0, 772.0, 798.0, 823.0, 848.0, 872.0, 898.0, 922.0, 947.0, 972.0]), ('id', ['12000100', '12000200', '12000201']), ('pose', OrderedDict([('12000100', OrderedDict([(649.0, [3.713, 1.669]), (679.0, [3.703, 1.66]), (698.0, [3.694, 1.657]), (723.0, [3.688, 1.654]), (748.0, [3.682, 1.651]), (772.0, [3.682, 1.649]), (798.0, [3.679, 1.646]), (823.0, [3.68, 1.646]), (848.0, [3.677, 1.644]), (872.0, [3.682, 1.643]), (898.0, [3.686, 1.643]), (922.0, [3.685, 1.643]), (947.0, [3.682, 1.644]), (972.0, [3.684, 1.647])])), ('12000200', OrderedDict([(649.0, [3.892, 1.08]), (679.0, [3.902, 1.092]), (698.0, [3.903, 1.089]), (723.0, [3.914, 1.091]), (748.0, [3.922, 1.091]), (772.0, [3.936, 1.086]), (798.0, [3.937, 1.087]), (823.0, [3.924, 1.106]), (848.0, [3.917, 1.112]), (872.0, [3.915, 1.111]), (898.0, [3.916, 1.112]), (922.0, [3.909, 1.108]), (947.0, [3.903, 1.112]), (972.0, [3.891, 1.121])])), ('12000201', OrderedDict([(649.0, [4.233, 1.964]), (679.0, [4.232, 1.965]), (698.0, [4.224, 1.976]), (723.0, [4.214, 1.988]), (748.0, [4.205, 2.0]), (772.0, [4.194, 2.013]), (798.0, [4.183, 2.029]), (823.0, [4.173, 2.054]), (848.0, [4.162, 2.085]), (872.0, [4.153, 2.113]), (898.0, [4.144, 2.146]), (922.0, [4.132, 2.174]), (947.0, [4.129, 2.195]), (972.0, [4.117, 2.217])]))]))])
    # print result['time_us']

    
    # index = [[1547896351]*len(result['time_us']), result['time_us']]
    # obj = []
    # for i in range(len(result['id'])):
    #     columns = [result['id'][i]]
    #     a = Series(data =result['pose'][columns[0]].values() ,
    #                 index=index)    
    #     b = DataFrame(a,columns=columns)
    #     obj.append(b)
        

    # for i in range(len(obj)-1):
    #     print i
    #     try:
           
    #         f = pd.merge(f,obj[i+1],how='outer', left_index=True, right_index=True)
    #         print 'try'
    #     except:
    #         print 'except'
    #         f = pd.merge(obj[i],obj[i+1],how='outer', left_index=True, right_index=True)
    # print f
    # columns2 = [result['id'][1]]
    # c = Series(data =result['pose'][columns2[0]].values() ,
    #                 index=index)
    
    # d = DataFrame(c,columns=columns2)
    # # print b,d
    # f = pd.merge(d,b,how='outer', left_index=True, right_index=True)
    # columns3 = [result['id'][0]]
    # j = Series(data =result['pose'][columns3[0]].values() ,
    #                 index=index)
    
    # k = DataFrame(j,columns=columns3)
    # l = pd.merge(f,k,how='outer', left_index=True, right_index=True)
    # l.as_matrix
    # o = np.array(l.as_matrix)
    # p = []
    # print np.matrix(o).tolist()[0][0][649.0]
    # p.append(l)
    # print p
    # q = np.asarray([[p]])
    # time_s =[1547889]
    # time_ms = [10.0,20.0,30.0,40.0,50.0]
    # index=[time_s * len(time_ms), time_ms]

    # columns = [['id1','id1'],['x','y']]
    # # print Series([[5,2],[2,4],[3,4]])
    # a = DataFrame( data=[[5,2],[2,4],[3,4],[4,9],[3,4]], index=index, columns= columns)
    # # print a
    # # columns2 = [['id2','id2'],['x','y']]
    # # time_s =[1547889]
    # time_ms2 = [10.0,20.0,30.0,60.0]
    # index2 = ndex=[time_s * len(time_ms2), time_ms2]
    # b = DataFrame( data=[[5,2],[2,4],[3,4],[2,3]], index=index2,columns=columns2)
    # # print b
    # # c = pd.merge(a,b, left_index = True , right_index= True, how='outer')
    # # print c

    # columns3 = ['id3','keys']    
    # time_s =[1547889]
    # time_ms3 = [10.0]
    # # index3=[time_s * len(time_ms3), time_ms3]
    # f = Series(data = [[5,2],[2,4],[3,4]], index=[['a','a','a'],[2,1,3]],name= 'id1')
    # # o = Series(data = [1,2,3,4], index=[['a','a','a','a'],[1,2,3,4]], name = 'keys')
    # z = Series(data = [[5,2],[2,3],[10,4],[1,5]], index=[['a','a','a','a'],[1,2,3,4]], name = 'id2')
    # y = pd.concat([f,z], axis = 1) 
    # # print y
    # # g = DataFrame(y)

    # g = Series(data = [[5,2],[2,4],[3,4]], index=[['b','b','b'],[5,2,3]],name= 'id1')
    # # h = Series(data = [5,6,7,8], index=[['b','b','b','b'],[1,2,3,4]], name = 'keys')
    # k = Series(data = [[5,2],[2,3],[10,4],[1,5],[1,2]], index=[['b','b','b','b','b'],[1,2,3,4,5]], name = 'id2')
    # x = pd.concat([g,k], axis = 1) 
    # # print x

    # q = pd.concat([y,x], join='outer')

    # f = Series(data = [[5,2],[2,4],[3,4]], index=[['a','a','a','a'],[1,2,3,4]],name= 'id1')
    # g  = Series(data = [[5,2],[2,4],[3,4],[2,5]], index=[['a','a','a','a'],[1,2,3,4]],name= 'id2')
    # print g
    
    # a = OrderedDict([('12000100', [[3.713, 1.669], [3.703, 1.66], [3.694, 1.657], [3.688, 1.654], [3.682, 1.651], [3.682, 1.649], [3.679, 1.646], [3.68, 1.646], [3.677, 1.644], [3.682, 1.643], [3.686, 1.643], [3.685, 1.643], [3.682, 1.644], [3.684, 1.647]]), ('12000200', [[3.892, 1.08], [3.902, 1.092], [3.903, 1.089], [3.914, 1.091], [3.922, 1.091], [3.936, 1.086], [3.937, 1.087], [3.924, 1.106], [3.917, 1.112], [3.915, 1.111], [3.916, 1.112], [3.909, 1.108], [3.903, 1.112], [3.891, 1.121]]), ('12000201', [[4.233, 1.964], [4.232, 1.965], [4.224, 1.976], [4.214, 1.988], [4.205, 2.0], [4.194, 2.013], [4.183, 2.029], [4.173, 2.054], [4.162, 2.085], [4.153, 2.113], [4.144, 2.146], [4.132, 2.174], [4.129, 2.195], [4.117, 2.217]])])
    
    # f = DataFrame(a, index=[['a']*len(a['12000100']),[649.0, 679.0, 698.0, 723.0, 748.0, 772.0, 798.0, 823.0, 848.0, 872.0, 898.0, 922.0, 947.0, 972.0]])
    # # print f

    # b =OrderedDict([('12000100', [[3.689, 1.648], [3.697, 1.656], [3.704, 1.66], [3.709, 1.663], [3.714, 1.667], [3.712, 1.67], [3.712, 1.676], [3.714, 1.68], [3.713, 1.682], [3.712, 1.681], [3.71, 1.686], [3.71, 1.689], [3.708, 1.694], [3.71, 1.703], [3.711, 1.704], [3.707, 1.704], [3.708, 1.709], [3.703, 1.711], [3.697, 1.723], [3.676, 1.728], [3.672, 1.732], [3.671, 1.745], [3.661, 1.748], [3.655, 1.749], [3.649, 1.751], [3.646, 1.75], [3.641, 1.75], [3.636, 1.75], [3.632, 1.751], [3.632, 1.753], [3.635, 1.757], [3.632, 1.757], [3.629, 1.758], [3.624, 1.767], [3.62, 1.768], [3.616, 1.77], [3.612, 1.773], [3.605, 1.773], [3.602, 1.778], [3.596, 1.785], [3.601, 1.775]]), ('12000200', [[3.882, 1.13], [3.864, 1.13], [3.846, 1.144], [3.832, 1.147], [3.824, 1.153], [3.822, 1.157], [3.82, 1.163], [3.822, 1.171], [3.821, 1.181], [3.814, 1.187], [3.805, 1.182], [3.796, 1.188], [3.792, 1.19], [3.789, 1.193], [3.785, 1.195], [3.782, 1.197], [3.777, 1.199], [3.773, 1.202], [3.714, 1.211], [3.749, 1.142], [3.818, 1.018], [3.842, 0.973], [3.849, 0.947], [3.858, 0.922], [3.866, 0.896], [3.873, 0.875], [3.883, 0.86], [3.891, 0.851], [3.905, 0.84], [3.911, 0.838], [3.939, 0.838], [3.941, 0.832], [3.941, 0.832], [3.928, 0.833], [3.934, 0.844], [3.956, 0.856], [3.991, 0.872], [4.083, 0.917], [4.211, 1.001], [4.281, 1.047], [4.302, 1.055]]), ('12000201', [[4.107, 2.237], [4.098, 2.25], [4.096, 2.264], [4.096, 2.284], [4.088, 2.295], [4.082, 2.301], [4.075, 2.309], [4.078, 2.321], [4.075, 2.335], [4.084, 2.357], [4.088, 2.374], [4.087, 2.384], [4.084, 2.393], [4.087, 2.412], [4.084, 2.424], [4.087, 2.439], [4.087, 2.453], [4.085, 2.468], [4.074, 2.481], [4.06, 2.491], [4.051, 2.499], [4.05, 2.523], [4.034, 2.541], [4.024, 2.567], [4.019, 2.596], [4.017, 2.622], [4.011, 2.647], [4.005, 2.664], [3.997, 2.684], [3.991, 2.698], [3.983, 2.713], [3.967, 2.723], [3.945, 2.733], [3.921, 2.745], [3.901, 2.753], [3.882, 2.759], [3.861, 2.769], [3.84, 2.78], [3.83, 2.792], [3.815, 2.8], [3.798, 2.812]])])
    # g = DataFrame(b,index=[['b']*len(b['12000100']),[4.0, 24.0, 48.0, 74.0, 98.0, 122.0, 145.0, 170.0, 195.0, 221.0, 246.0, 272.0, 296.0, 322.0, 347.0, 373.0, 397.0, 421.0, 447.0, 471.0, 496.0, 521.0, 546.0, 571.0, 598.0, 621.0, 646.0, 671.0, 696.0, 721.0, 751.0, 772.0, 797.0, 821.0, 848.0, 872.0, 897.0, 921.0, 947.0, 972.0, 996.0]]) 
    # # print g

    # x = pd.concat([f,g], join = 'outer')
    # print x

    # d =pd.read_csv('/home/mtb/Desktop/dat.csv')
    # print d

    # a = pd.read_csv('/home/mtb/Desktop/dat.csv',index_col=[0,1],header=[0,1])
    # print a.loc[1512590402.0]['12000200'][['x']]
    # print a.loc['1512590405.0']
    # print a['12000100'].index.values
    # print a.loc[1512590402.0,['x','y']]
    
    

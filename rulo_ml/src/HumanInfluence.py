#!/usr/bin/env	python
import rospy
import os
import sys
import tensorflow as tf
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.graph_plot import Plot
from rulo_base.markers import VizualMark,Line
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from collections import OrderedDict


dirt_sensor_file = '/home/mtb/Documents/data/dirt_extraction_2/data/'
result_file = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/'
cells_file = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv'

reject = [447,962, 967,1178,1200, 1201]

cleaning_cycle = ['12-5.csv', '12-7.csv', '12-9.csv', '12-12.csv',  '12-13.csv',
                  '12-14.csv','12-15.csv', '12-19.csv', '12-20.csv','12-21.csv']
air_cells = [838, 839, 840, 841, 842,
             865, 866, 867, 868, 869]
cells_num = 617


class HumanInfluence:
    def __init__(self, sensor='high'):
        self.sensor = sensor
        self.epoch = OrderedDict()
        self.cells = []
        self.marker = 'r' if self.sensor == 'high' else 'g'
    
    def makedir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def extract_time(self):
        file_list = ['12-4_noon.csv', '12-5_noon.csv', '12-6_morning.csv', '12-6_noon.csv', '12-7_noon.csv',
                     '12-8_morning.csv', '12-8_noon.csv', '12-9_morning.csv', '12-9_noon.csv','12-11_noon.csv',
                     '12-12_morning.csv', '12-12_noon.csv', '12-13_morning.csv', '12-13_noon.csv', '12-14_morning.csv',
                     '12-14_noon.csv', '12-15_morning.csv', '12-15_noon.csv', '12-16_noon.csv', '12-17_noon.csv',
                     '12-18_morning.csv','12-18_noon.csv','12-19_morning.csv', '12-19_noon.csv', '12-20_noon.csv',
                     '12-21_morning.csv','12-21_noon.csv']
        for files in file_list:
            data = csvread(dirt_sensor_file + 'time/'+ files)
            self.epoch[files] = [float(data['start'][0]), float(data['finish'][0])]
    
    def  get_cells(self):
        data = pd.read_csv(cells_file,index_col=None)
        cells_partition = [data.columns.values[0]]
        for elem in data.values:
            cells_partition.append(elem[0])
        for elem in cells_partition:
            mylist =[]
            for val in elem.split(':'):
                mylist.append(int(val))
            for d in range(mylist[0], mylist[1]):
                self.cells.append(d)
             
    def rewrite_human_file(self,rewrite=False):
        self.get_cells()
        self.order_file = OrderedDict()    
        self.order_file['test2017-12-05_12h25.csv'] = ['12-4_noon.csv']
        self.order_file['test2017-12-07_10h15.csv'] = ['12-5_noon.csv', '12-6_morning.csv', '12-6_noon.csv']
        self.order_file['test2017-12-09_11h10.csv'] = ['12-7_noon.csv','12-8_morning.csv', '12-8_noon.csv', '12-9_morning.csv']
        self.order_file['test2017-12-12_11h05.csv'] = ['12-9_noon.csv','12-11_noon.csv','12-12_morning.csv']
        self.order_file['test2017-12-13_10h18.csv'] = ['12-12_noon.csv', '12-13_morning.csv']
        self.order_file['test2017-12-14_11h10.csv'] = ['12-13_noon.csv', '12-14_morning.csv']
        self.order_file['test2017-12-15_14h30.csv'] = ['12-14_noon.csv', '12-15_morning.csv']
        self.order_file['test2017-12-19_13h40.csv'] = ['12-15_noon.csv', '12-16_noon.csv', '12-17_noon.csv','12-18_morning.csv', '12-18_noon.csv', '12-19_morning.csv']
        self.order_file['test2017-12-20_20h30.csv'] = ['12-19_noon.csv', '12-20_noon.csv']
        self.order_file['test2017-12-21_14h38.csv'] = ['12-21_morning.csv','12-21_noon.csv']
      
        # for files in sorted(os.listdir(dirt_sensor_file + self.sensor + '/'))[-10:]:
        #     # for cells in self.cells :
        #     print files
        if rewrite:
            for keys, values in self.order_file.items():
                human_pres = []
                for val in values:
                    data = csvread(dirt_sensor_file + 'human presence/'+val)['grid_freq_val']
                    for i in range(len(data)):
                        try:
                            human_pres[i] += float(data[i])
                        except:
                            human_pres.append(float(data[i]))
                csvwriter(result_file + 'human_time_in_cells/' + keys,
                          headers=['human_frequency'], rows=[human_pres])
    
    def get_cells_param(self, save=False):
        # self.rewrite_human_file(rewrite=False)
        self.get_cells()
        self.dirt_cells = OrderedDict()

        for cells in self.cells:
            self.dirt_cells[str(cells)] = OrderedDict()
        
        for files in sorted(os.listdir(dirt_sensor_file + self.sensor + '/dirt_value/'))[:]:
            
            dirt_data = csvread(dirt_sensor_file + self.sensor + '/dirt_value/' + files)
            # human_data = csvread(result_file + 'human_time_in_cells/' + files)
            for cells in self.dirt_cells:
                try:
                    self.dirt_cells[cells]['dirt_level'].append(int(dirt_data['dirt_level'][int(cells)]))
                    self.dirt_cells[cells]['wall_time'].append(float(dirt_data['wall_time'][int(cells)]))
                    self.dirt_cells[cells]['cleaning_duration'].append(float(dirt_data['cleaning_duration'][int(cells)]))
                    # self.dirt_cells[cells]['human_frequency'].append(float(human_data['human_frequency'][int(cells)]))              
                except:
                    self.dirt_cells[cells]['dirt_level'] = [int(dirt_data['dirt_level'][int(cells)])]
                    self.dirt_cells[cells]['wall_time'] =[float(dirt_data['wall_time'][int(cells)])]
                    # self.dirt_cells[cells]['human_frequency'] =[float(human_data['human_frequency'][int(cells)])]
                    self.dirt_cells[cells]['cleaning_duration'] =[float(dirt_data['cleaning_duration'][int(cells)])]
        # print self.dirt_cells['116']
        if save :
            for cells in self.dirt_cells:
                # dataframe_human = DataFrame(data=self.dirt_cells[cells]['human_frequency'],
                #                             index=self.dirt_cells[cells]['wall_time'], columns=['human_frequency'])
                dataframe_dirt = DataFrame(data=self.dirt_cells[cells]['dirt_level'],
                                        index=self.dirt_cells[cells]['wall_time'], columns=['dirt_level'])
                dataframe_cycle = DataFrame(data=self.dirt_cells[cells]['cleaning_duration'],
                                           index=self.dirt_cells[cells]['wall_time'], columns=['cleaning_duration'])
                # dataframe = pd.merge(dataframe_dirt,right_index=True, left_index=True)    
                dataframe = pd.merge(dataframe_dirt, dataframe_cycle,right_index=True, left_index=True)    

                path = self.makedir('/home/mtb/Documents/data/dirt_extraction_2/result_2/' + 'human_data_arranged/'+self.sensor +'/')
                dataframe.to_csv(path +cells +'.csv')
                
    def get_last_cleaning_time(self):
        self.get_cells()
        selected = []
        self.last_wall_time = {}
        for files in sorted(os.listdir(dirt_sensor_file + self.sensor + '/'),reverse=True)[10:]:
            data = csvread(dirt_sensor_file + self.sensor +'/' + files)
            wall_time = data['wall_time']            
            for i in self.cells: 
                if str(i) not in self.last_wall_time:  
                    if float(wall_time[i]) >0.0:
                        self.last_wall_time[str(i)] = float(wall_time[i])    
        for t in self.cells:
            if str(t) not in self.last_wall_time.keys() and t not in reject:
                selected.append(t)
        for elem in selected:
            self.last_wall_time[str(elem)] = 0.0 
    
    def write_param(self):
        self.get_last_cleaning_time()
        for cells in self.cells :
            if cells not in reject:
                df = pd.read_csv(result_file + 'human_data_arranged/' +
                                 self.sensor + '/' + str(cells) + '.csv', index_col=[0])
                last_cleaning_cycle = self.last_wall_time[str(cells)]
                dirt_level = []
                duration = []
                human_pres = []
                cleaning_duration = []
                buffer_hum =0.0               
                for k in range(df.index.values.shape[0]):
                    if df.index.values[k] >0.0:
                        dirt_level.append(df.values[k][0])
                        cleaning_duration.append(df.values[k][2])
                        duration.append(df.index.values[k]-last_cleaning_cycle)
                        human_pres.append(buffer_hum + df.values[k][1])
                        last_cleaning_cycle = df.index.values[k]
                        buffer_hum =0.0
                    else: 
                        buffer_hum += df.values[k][1]                        
                dataframe = DataFrame({'dirt_level': dirt_level, 'human_frequency': human_pres,'cleaning_duration':cleaning_duration}, index=duration)
                dataframe.to_csv(result_file+ 'human_data/'+self.sensor +'/data/'+str(cells)+'.csv')
          
    def get_human_dirt(self):
        self.get_cells()        
        res=OrderedDict()
        data_path = result_file + 'human_data/' + self.sensor + '/data/'
        for cells in self.cells:            
            if cells not in reject and cells not in air_cells:
                df = pd.read_csv(data_path + str(cells)+'.csv',index_col=[0])
                res[str(cells)] = df
        return res 

    def plot_human_dirt_cell(self,scaling=False):
        plot_path = result_file + 'human_data/' + self.sensor + '/plot/'
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
        data =self.get_human_dirt()
        duration =[]
        dirt_level =[]
        for keys, val in data.items():
            for i in range(val.index.values.shape[0]):
                duration.append(val.index.values[i])
                if scaling:
                    dirt_level.append(float(val.values[i][1]) / float(val.values[i][0]))
                else:
                    dirt_level.append(val.values[i][1])                    
        Plot().scatter_plot(x_value= duration, y_value = dirt_level, marker=self.marker + 'o', 
                            title='Dirt sucked in cells that human passed through by '+self.sensor+' level sensor',
                            labels =['Cleaning_cycle', 'Dirt_Level'], save_path=plot_path +self.sensor+'.png',show =True )

    def get_weights(self):
        estimation_path = result_file +'estimation/' + self.sensor +'/'
        data = self.get_human_dirt()
        duration =[]
        dirt_level = []
        weights =[]
        for keys, val in data.items():
            for i in range(val.index.values.shape[0]):
                duration.append(val.index.values[i])
                dirt_level.append(val.values[i][1])
                if float(val.values[i][2] / val.index.values[i]) / 1000.0 <= 1.0:
                    weights.append((0.001 * float(val.values[i][2]))/ val.index.values[i])
                else:
                    weights.append(1.0)
        dataframe =DataFrame({'dirt_level': dirt_level, 'weights':weights}, index=duration)
        dataframe.to_csv(estimation_path + 'human_cell_weights.csv')

    def plot_dirt_cells(self):
        j =0
        for files in os.listdir('/home/mtb/Documents/data/dirt_extraction/human_influence/clean_cycle_hum_pres/low/data/'):
            print files
            data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction/human_influence/clean_cycle_hum_pres/low/data/'+ files, index_col=[0])
            a = []
            j +=1
            for i in range(data.index.values.shape[0]):
                a.append(data.values[i][1]/(1000.0*data.index.values[i]))
            # print max(a)
            Plot().scatter_plot(
                a, list(data.values[:, 0]), marker=self.marker +'o',save_path='/home/mtb/Documents/data/dirt_extraction/human_influence/clean_cycle_hum_pres/low/plot/'+files[:-4]+'.png')

    def get_params(self):
        folder = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/'
        for files_num in range(1198,1200):
            if os.path.isfile(folder + str(files_num)+'.csv'):
                # filename = raw_input('Enter_filename: ')            
                data = pd.read_csv(folder + str(files_num)+'.csv',index_col=[0])   
                print data['ratio'].values
                while True:
                    print files_num
                    step = float(raw_input('Define the step: ')) *1000.0        
                    times = OrderedDict()
                    dirt = OrderedDict()
                    for k in range(6):
                        times[step * float(k+1)] = []
                        dirt[step * float(k + 1)] = 0.0
                    # print step
                    for i in range(data.index.values.shape[0]):            
                        for keys in times:
                            indice = times.keys().index(keys)
                            if step *float(indice) <=data.index.values[i]<=keys:
                                dirt[keys] += data['ratio'].values[i]
                                times[keys].append(i)
                                break                                 
                    for keys in dirt:
                        if len(times[keys]) >0:
                            dirt[keys] = dirt[keys] / len(times[keys])    
                    # print dirt               
                    Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
                    
                    saving = raw_input('Do you wanna save?: ')
                    if str(saving) == 'y':
                        elem = raw_input('Enter skip id: ')
                        skip_id = [int(val) for val in elem.split(',')]
                        # print skip_id

                        csvwriter('/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_high_ratio.csv',
                                    headers=['cells','step','k','skip_id'], 
                                    rows = [[files_num],[step],[k+1],[skip_id]])
                        # self.learn(files_num,step,k+1,skip_id )
                        
                        self.get_weibull_distribution(files_num, skip_id ,dirt)
                        break
                    else:
                        continue

                    # if files[:-4] == '116':
                    #     # self.get_weibull_distribution(files[:-4],dirt)
                    #     Plot().scatter_plot(dirt.keys(), dirt.values(), show=True) 
                    #     break
        
            else:
                pass
    
    def learn(self,files_num,step,k,skip_id):
        # file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/low/weibull.csv'
        folder = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/'
        # data = pd.read_csv(file)
        # print data['cells']
        # while True:
        # for p in range(0,len(data['cells'])):
           
        cells = files_num #data['cells'][p]
        step = step #float(data['step'][p])
        ranges = 6
        skip_id = skip_id #map(int, data['skip_id'][p][1:-1].split(','))
        data_1 = pd.read_csv(folder + str(cells) + '.csv', index_col=[0])
        # print cells
        times = OrderedDict()
        dirt = OrderedDict()

        if int(cells) == 1204:
            for k in range(6):
                        times[step * float(k + 1)] = []
                        dirt[step * float(k + 1)] = 0.0
            for i in range(data_1.index.values.shape[0]):            
                        for keys in times:
                            indice = times.keys().index(keys)
                            if step *float(indice) <=data_1.index.values[i]<=keys:
                                dirt[keys] += data_1['dirt'].values[i]
                                times[keys].append(i)
                                break                                 
            for keys in dirt:
                        if len(times[keys]) >0:
                            dirt[keys] = dirt[keys] / len(times[keys])
            # print dirt            
            # Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
            #     prompt_1 = raw_input('Do you wanna learn?')
            #     if str(prompt_1) =='y':
            #         skip_id = []
            #         prompt_2 = raw_input('Enter skip id: ')
            #         for elem in str(prompt_2).split(','):
            #             skip_id.append(int(elem))
            #         # try:
            self.get_weibull_distribution(cells, skip_id, dirt)
            #         # except KeyboardInterrupt:
            #         #     pass
            #         break

            #     elif str(prompt_1) == 'c':
            #         sys.exit(0)
            #     # elif 
            #     else:
            #         break
                    
    def get_weibull_distribution(self,cells,skip_id=[],*args):
        t = tf.placeholder(tf.float32)
        dirt_data= tf.placeholder(tf.float32)
        beta = tf.Variable(initial_value= 2.0)
        eta = tf.Variable(initial_value=3.0)
        A = tf.Variable(initial_value=50.0)
        f1 = tf.divide(beta, tf.pow(eta, beta) )
        f2 = tf.pow(t, (beta -1))
        f3 = tf.exp(-(t/eta)**beta)
        f_t =  A * f1 * f2 * f3
       
        loss = dirt_data - f_t
        cost = tf.reduce_mean(tf.pow(loss,2))
        optimize_BE = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
        optimize_A = tf.train.GradientDescentOptimizer(learning_rate =0.01)

        grads_A = optimize_A.compute_gradients(cost,[A])
        grads_BE = optimize_BE.compute_gradients(cost, [beta, eta])
        opt_A = optimize_A.apply_gradients(grads_A)
        opt_B = optimize_BE.apply_gradients(grads_BE)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # times_param = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys())) ]
            # y  = [args[0].values()[i] for i in range(len(args[0].values()))]
            #if if i not in [4]

            x_val = []
            y_val = []
            for i in range(len(args[0].keys())):
                if i not in skip_id:
                    x_val.append(args[0].keys()[i])
                    y_val.append(args[0].values()[i])

            cells_cycle = max(x_val)
            times_param = [ time_val / cells_cycle for time_val in x_val   ]
            y  = y_val
            # print args[0].values()
            # print y
            # print times_param
            error_list = [0.0]
            try:
                i =0
                while True:            
                
                    # for i in range(10000):
                        i +=1
                        opt1,opt2 , error,  var1, var2 ,var3=  sess.run([opt_A,opt_B, cost,A, beta,eta], {t: times_param, dirt_data: y })                    
                        print cells,np.sqrt(error), var1,var2, var3
                        # if np.sqrt(error) < 0.2:break
                        if i %1000 == 0:
                            print cells,np.sqrt(error), var1,var2, var3
                        #     if np.sqrt(error) == error_list[0] or np.isnan(np.sqrt(error)):break                        
                        #     else: error_list[0] = np.sqrt(error)
            except KeyboardInterrupt:
                   pass
                    # prompt_3 = raw_input('Do you wanna save the parameters? ')
                    # if str(prompt_3) == 'y':
                    #     prompt_4 = raw_input('Enter success or error')
                    # pass
            csvwriter(
                        '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_learn_high_2.csv',
                        ['cells', 'error', 'A', 'beta', 'eta', 'mode','cycle'], [[cells], [np.sqrt(error)], [var1], [var2], [var3], ['s'],[cells_cycle]])
                                
                # else:
                #     pass
                    
        

            # A, beta, eta =A ,3.67853 ,0.505138
            # f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A for t in times_param]
            # plt.plot(times_param, y,'ro', times_param, f_t , 'b-')
            # plt.show()

    def plot_estimation(self):
        est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/params.csv'
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/'
        data_1 = pd.read_csv(est_file)
        a = np.array(data_1['cells'].values,dtype='str')
        for files in os.listdir(dirt_fold):
            data_2 = pd.read_csv(dirt_fold + files,index_col=[0])         
            print files  
            try:     
                index =  np.where(int(files[:-4]) == data_1['cells'].values)[0][0]               
                [A, beta, eta, cycle]= [data_1.iloc[index]['A'], 
                                     data_1.iloc[index]['beta'], 
                                     data_1.iloc[index]['eta'],
                                     data_1.iloc[index]['cycle']]               
                dirt = data_2['ratio'].values
                duration = data_2.index.values
                x_val = []
                y_val = []            
                if np.amax(duration) > cycle:
                    cycle_time =int(np.amax(duration) / cycle)
                    for m in range(cycle_time + 3):
                        times_par = np.linspace(m*cycle, (m+1)*cycle, 10)
                        times_param = []
                        for elem in times_par:                           
                            times_param.append(elem / ((m + 1) * cycle))                     
                        if m ==0:
                            f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A for t in times_param]                        
                        for i in range(len(f_t)):
                            x_val.append(times_par[i])
                            y_val.append(f_t[i])
                    plt.plot(duration, dirt, 'ro',label='Experiments data')
                    plt.plot(x_val, y_val,label='Estimated density function')
                    plt.xlabel('Cleaning_cycle')
                    plt.ylabel('Dirt')
                    plt.title('Dirt estimation for cells ' + files[:-4])
                    plt.legend()
                    # plt.show()
                    plt.savefig(
                        '/home/mtb/Documents/data/dirt_extraction_2/result/ratio/plot/' + files[:-4] +'.png')
                    plt.close()
                    
            except:
                pass
                    
                    # plt.show()
                # break

                # Plot().scatter_plot(list(duration), list(dirt),show=True)
    
    def get_mean_performance(self):
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/human_data/high/data/'
        mean_result = OrderedDict()
        max_ = -1.0
        for files in os.listdir(dirt_fold):
            data_1 = pd.read_csv(dirt_fold + files,index_col=[0])
            mean_result[files[:-4]] = OrderedDict()
            
            mean_result[files[:-4]]['mean'] =  np.mean(data_1['dirt_level'].values)
            if mean_result[files[:-4]]['mean'] >max_:
                max_ = mean_result[files[:-4]]['mean']
        #Mean and ratio during cleaning
        for keys in mean_result:
            mean_result[keys]['ratio'] = mean_result[keys]['mean'] / max_

        
        day_clean_folder = '/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/'
        mean_day = OrderedDict()
        mean_by_cleaning = OrderedDict()
        for i in range(1296):
            mean_day[str(i)] = OrderedDict()
            mean_day[str(i)]['mean'] = []
            mean_day[str(i)]['ratio'] = []
            mean_by_cleaning[str(i)]=OrderedDict()

        for files in os.listdir(day_clean_folder):
            data_2 = pd.read_csv(day_clean_folder + files,index_col=[0])
            for i in range(1296):                
                mean_day[str(i)]['mean'].append(float(data_2['dirt_level'].values[i]))
                mean_day[str(i)]['ratio'].append(float(data_2['dirt_level'].values[i])/float(np.max(data_2['dirt_level'].values)))
        max_ = 0.0
        keep = []
        for keys in mean_day:
            if keys in mean_result:
                mean_by_cleaning[keys]['day_mean']= np.mean(mean_day[keys]['mean'])
                mean_by_cleaning[keys]['ratio']= np.mean(mean_day[keys]['ratio'])
                if mean_by_cleaning[keys]['day_mean'] >max_:
                    max_ = mean_by_cleaning[keys]['day_mean']
                keep.append(keys)
        result=OrderedDict()
        all_result = []
        for keys in mean_by_cleaning:
            if keys in keep:
                mean_by_cleaning[keys]['proportion_mean'] = mean_by_cleaning[keys]['day_mean'] / max_
                error = (np.abs(mean_by_cleaning[keys]['ratio'] -mean_by_cleaning[keys]['proportion_mean']))/mean_by_cleaning[keys]['proportion_mean']
                dataframe = DataFrame({'mean':mean_by_cleaning[keys]['day_mean'], 'ratio_by_day':mean_by_cleaning[keys]['ratio'],
                                        'estimation_ratio':mean_by_cleaning[keys]['proportion_mean'], 'error': error}, index=[keys])
                all_result.append(dataframe)        
        b = pd.concat(all_result,axis=0)        
        b.to_csv('/home/mtb/Documents/data/dirt_extraction_2/result/mean/estimated_mean.csv')

    def plot_mean_exp(self):
        data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/result/mean/estimated_mean.csv',index_col=[0])
        positions = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/all_cells_pose.npy')
        pose =[]
        color =[]
        id_list = []
        for cells in (data.index.values):      
            a = float(data.loc[cells,'estimation_ratio'] )/0.8
            if a >=1.0:
                a=1.0
            b = a*245.0
            c = 255.0-b          
            color.append(int(c))
            pose.append(list(positions[cells]))
            id_list.append(cells)
        rospy.init_node('path')
        VizualMark().publish_marker(pose, sizes=[[0.25,0.25,0.0]]* len(pose), color=color,convert= True, texture='Red', id_list =id_list)

    def get_weibull_performance(self):
        est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/params.csv'
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/'
        data_1 = pd.read_csv(est_file)
        a = np.array(data_1['cells'].values,dtype='str')
        all_errors = []
        dataframe_list = []
        for files in os.listdir(dirt_fold):
            data_2 = pd.read_csv(dirt_fold + files,index_col=[0])
            try:      
                        index =  np.where(int(files[:-4]) == data_1['cells'].values)[0][0]
                        A, beta, eta, cycle= data_1.iloc[index]['A'], data_1.iloc[index]['beta'], data_1.iloc[index]['eta'], data_1.iloc[index]['cycle']
                        dirt = data_2['ratio'].values
                        duration = data_2.index.values
                        x_val = []
                        y_val = []
                # try:
                # if np.amax(duration) > cycle:
                        cycle_time =int(np.amax(duration) / cycle)
                        loop =[0.0]
                        for m in range(cycle_time + 3):
                            times_par = np.linspace(m*cycle, (m+1)*cycle, 10)
                            times_param = []
                            loop.append((m+1)*cycle)
                            for elem in times_par:                           
                                times_param.append(elem / ((m + 1) * cycle))
                            # print times_param
                            if m ==0:
                                f_t = []
                                for t in times_param:
                                    try:
                                        f_t.append((beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A)
                                    except:
                                        f_t.append(0.0)                       
                                                
                            for i in range(len(f_t)):
                                x_val.append(times_par[i])
                                y_val.append(f_t[i])
                            # print f_t

                        # plt.plot(duration, dirt, 'ro',markersize=3,label='Experiment Data')
                        # plt.plot(x_val, y_val,label='Estimated Distribution Curve')
                        z_val = []
                        w_val = []
                        for times in duration:
                            for i in range(len(loop)-1):
                                if  loop[i]<=times<loop[i+1]:
                                    t = times /loop[i+1]
                                    f_t = (beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A 
                                    w_val.append(times)
                                    z_val.append(f_t)
                                    break
                        error = [np.abs(z_val[i] - dirt[i])/ np.max([z_val[i], dirt[i]]) for i in range(len(z_val))]
                        dataframe = DataFrame({'error':[error]}, index =[files[:-4]])
                        dataframe_list.append(dataframe)
                        # print dataframe
                        # print w_val
                        # print error
                        # all_errors.append(np.mean(error))
                        # plt.plot(w_val, z_val, 'go',label='Dirt estimated from the distribution',markersize=3)
                        # plt.legend()
                        
                        # plt.xlabel('Cleaning_cycle')
                        # plt.ylabel('Dirt')
                        # plt.title('High Dirt estimation for cells ' + files[:-4])
                        # plt.show()
                        # plt.savefig(
                        #     '/home/mtb/Documents/data/dirt_extraction_2/result/estimation_fitting-error/' + files[:-4] +'.png')
                        # plt.close()

            except:
                    pass

        # print np.mean(all_errors)

        final_error = pd.concat(dataframe_list)
        final_error.to_csv('/home/mtb/Documents/data/dirt_extraction_2/result/ratio/error/ratio_error.csv')

    def get_average_error(self):
        data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/result/ratio/error/ratio_error.csv',index_col=[0])
        result = []
        classes = OrderedDict([(0.10,0),(0.25,0),(0.50,0),(0.75,0),(1.0,0)])
        for cells in data.index.values:
            trans = []
            for elem in data['error'][cells][1:-1].split(','):                     
                try:
                    if 0.0<=float(elem)<0.10:
                        classes[0.10] +=1
                    elif 0.10<=float(elem)<0.25:
                        classes[0.25] +=1
                    elif 0.25<=float(elem)<0.50:
                        classes[0.50] +=1
                    elif 0.50<=float(elem)<0.75:
                        classes[0.75] +=1
                    elif float(elem)>=0.75 :
                        classes[1.0] +=1
                        
                    trans.append(float(elem))                                            
                except:
                    pass                
            # result.append(trans) 
             
            dataframe = DataFrame({'error': np.mean(trans)},index=[cells])
            # print dataframe
            # print classes[0.1]
            result.append(dataframe)
        # all_result = pd.concat(result)
        # all_result.to_csv('/home/mtb/Documents/data/dirt_extraction_2/result/estimation_fitting-error/error/cells_mean_error.csv')
        # print np.mean(all_result.values) #Average error for all cells 0.566446526815
        # plt.plot(all_result.index,all_result.values,'ro')
        # plt.show()
        # print classes
        performance = OrderedDict()
        performance[90.0] = float(classes[0.1]) / sum(classes.values())
        performance[75.0] = float(classes[0.25])/ sum(classes.values())
        performance[50.0] = float(classes[0.50]) / sum(classes.values())
        performance[25.0] = float(classes[0.75]) / sum(classes.values())
        performance[0.0] = float(classes[1.0]) / sum(classes.values())

        csvwriter('/home/mtb/Documents/data/dirt_extraction_2/result/ratio/error/performance_high.csv',['trial_success','percentage'],[performance.keys(),performance.values()])

    def visualize_estimation(self):
        folder ='/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/'
        file_list =[]
        for files in sorted(os.listdir(folder)):
            file_list.append(files)
        test_file = '/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/test2017-12-07_10h15.csv'
        data_1 = pd.read_csv(test_file)
        file_to_examine =sorted(file_list[:14],reverse=False)
        last_time = OrderedDict()
        for cells in data_1.index.values:
            last_time[cells] = 2592000.0            
            for file in file_to_examine[::-1]:
                data_2 = pd.read_csv(folder + file)
                if data_2['wall_time'][cells] >15.0:                    
                    last_time[cells] = data_1['wall_time'][cells] - data_2['wall_time'][cells]
                    break
        
        # print last_time
        estimation_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_learn_2.csv'
        data_3 = pd.read_csv(estimation_file)
        cells_list =[]
        for files in os.listdir('/home/mtb/Documents/data/dirt_extraction_2/result/estimation_fitting-error/plot/'):
            cells_list.append(int(files[:-4]))
        data_4 =  list(data_3['cells'].values)
        data_5 = OrderedDict()
        iterations = 0
        for elem in sorted(cells_list):
            if elem in data_4:    
                # print elem            
                try:
                    data_5[elem] =OrderedDict()
                    data_5[elem]['dirt_data'] = data_1['dirt_level'][elem]
                    A = data_3[data_3['cells']==elem]['A'].values[0]
        
                    beta = data_3[data_3['cells']==elem]['beta'].values[0]
                    eta = data_3[data_3['cells']==elem]['eta'].values[0]
                    
                    cycle = data_3[data_3['cells']==elem]['cycle'].values[0]
                    recursion = int(4000000.0 /cycle)
                    last_cleaning_time = last_time[elem]
                    for m in range(recursion):
                    
                        if m * cycle<=last_cleaning_time<=(m+1)* cycle:
                            t = float(float(last_cleaning_time - (m * cycle)) / float((m+1)* cycle))
                            dirt_estimation = (beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A
                            # print elem , dirt_estimation , iterations, last_cleaning_time,beta,eta,A
                            # sys.exit(0)
                            data_5[elem]['dirt_estimation'] = dirt_estimation
                            break                            
                except:
                    continue        
            iterations +=1           
        positions = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/all_cells_pose.npy')
        pose = []
        id_list = []
        dirt_level = []
        for i in data_5.keys():
            if 'dirt_data' in data_5[i]:
                dirt_level.append(data_5[i]['dirt_data'])
                
            else:
                dirt_level.append(0)
            id_list.append(i)
        # print (dirt_level)
        

        # for i in id_list:
        #     pose.append(list(positions[i]))
                
        
        # color = [int(255 - float(dirt_level[i]) / float(max(dirt_level)) * 245)
        #                  for i in range(len(dirt_level))]
        # rospy.init_node('path')
        # VizualMark().publish_marker(pose, sizes=[[0.25,0.25,0.0]] * len(pose),color=color , id_list =id_list, convert= True, texture='Red' )
        
    def simulation(self):
        rospy.init_node('path')
        last_time = 1517832000  # 1513835007.0
        current_time = rospy.get_time() 
        # est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/low/weibull_learn_low.csv'
        est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/params.csv'
        # dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_low/'
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/'
        data_1 = pd.read_csv(est_file)
        a = np.array(data_1['cells'].values,dtype='str')
        result = OrderedDict()
        for files in os.listdir(dirt_fold):
            data_2 = pd.read_csv(dirt_fold + files,index_col=[0])            
            if str(files[:-4]) in a:            
                    index =  np.where(int(files[:-4]) == data_1['cells'].values)[0][0] 
                    # print index
                    A, beta, eta, cycle= data_1.iloc[index]['A'], data_1.iloc[index]['beta'], data_1.iloc[index]['eta'], data_1.iloc[index]['cycle']                    
                    # dirt = data_2['ratio'].values                   
                    # duration = data_2.index.values
                    x_val = []
                    y_val = []
                    v= (current_time - last_time) // cycle
                    u = np.int(v)
                    b = float(current_time - last_time)  - float(u) * cycle
                    t =  b / cycle
                    # print t
                    f_t = (beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A 
                    result[files[:-4]] = f_t
                
        # print result
        csvwriter('/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/high_02_19.csv', ['cells','dirt'],
                    [result.keys(), result.values()])
        # print result
                    
    def rewrite(self):
        
        for files in os.listdir('/home/mtb/Documents/data/dirt_extraction_2/data2/high/'):
            result = []
            data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/data2/high/' + files, index_col=[0])
            skip = False
            dirt_level = {'dirt': [],'ratio':[]}
            # print data['dirt_level']
            for i in range(data.index.values.shape[0]):               
                if data.index.values[i] >0.0 and len(result) == 0:
                    last_time = data.index.values[i]
                    result.append(0.0)
                    dirt_level['dirt'].append(data['dirt_level'].values[i])
                    if data['cleaning_duration'].values[i] <=1.0:
                         dirt_level['ratio'].append(data['dirt_level'].values[i])
                    else:
                        dirt_level['ratio'].append(float(data['dirt_level'].values[i]) /data['cleaning_duration'].values[i])

                elif data.index.values[i] > 0.0 and len(result)>=1:                  
                    value =  data.index.values[i] - last_time
                    result.append(value)
                    last_time = data.index.values[i]
                    dirt_level['dirt'].append(data['dirt_level'].values[i])
                    if data['cleaning_duration'].values[i] <=1.0:
                         dirt_level['ratio'].append(data['dirt_level'].values[i])
                    else:
                        dirt_level['ratio'].append(float(data['dirt_level'].values[i]) /data['cleaning_duration'].values[i])
                        
            # print dirt_level    
            dataframe = DataFrame(data=dirt_level, index=result)
            dataframe.to_csv(
                '/home/mtb/Documents/data/dirt_extraction_2/data2/final_high/' +files)

    def visualize_simulation(self):
        poses = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
        indices = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy')
        dirt_data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/high_02_19.csv')
        result = OrderedDict()
    
        
        # for i in range(dirt_data['cells'].values.shape[0]):
        #     if dirt_data['dirt'].values[i] >=1.0 or not np.isnan(dirt_data['dirt'].values[i]):
        #         result[dirt_data['cells'].values[i]] = dirt_data['dirt'].values[i]
        #     else:
        #         result[dirt_data['cells'].values[i]] = 0
        array = []
      
        # print dirt_data
        for elem in range(1296):
            # if elem in dirt_data['cells'].values:
                # a = np.where(dirt_data['cells'].values== elem)[0][0]
                # print a
                # print elem
                # print dirt_data.loc[0,'dirt']
                # sys.exit(0)
            try:
                array.append(dirt_data.loc[np.where(dirt_data['cells'].values== elem)[0][0],'dirt'] *10.0)
            except:
                array.append(0.0)

        # csvwriter('/home/mtb/Documents/data/dirt_extraction_2/data/high/test2017-02-19_estimation.csv', ['dirt_level'], [array])
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dirt_high_19.npy',np.array(array))

        # final_result = OrderedDict()

        # dirt = np.array(final_result.values())
        
        # filename='/home/mtb/Documents/data/dirt_extraction_2/data/low/dirt_value/test2017-02-01_low.csv'
        # real_data = pd.read_csv(filename)
        # error =0.0
        # for elem in indices:
        #     final_result[elem] = result[elem]  
        #     if np.where(indices == elem)[0][0] <=300:
        #         error +=np.square((final_result[elem] - real_data['dirt_level'][elem]))
        # print np.sqrt(error/300)
        # print real_data


        
        # print real_data
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dirt_low_01.npy',dirt)
        # color = [int(255 - (float(final_result[elem]) / max(final_result.values())) * 245)
        #                         for elem in final_result]
        # color = [color[i] if color[i]>=0.0 else 0 for i in range(len(color))]
        
        # pose =[[poses[i][0], poses[i][1]] for i in range(623)]
        # rospy.init_node('path')
        # VizualMark().publish_marker(pose, sizes=[[0.25,0.25,0.0]] *623,color=color,action='add',convert= True,texture='Red',publish_num = 2, id_list = final_result.keys())
        # cells = [12, 23, 24, 25, 36, 44, 37, 43, 52, 59, 74, 88, 102, 118, 134, 149, 165, 186, 207, 222, 208, 188, 189, 210, 223, 209, 224, 238, 252, 266, 279, 265, 264, 277, 290, 276, 262, 275, 274, 273, 259, 245, 231, 217, 203, 182, 162, 161, 146, 160, 179, 178, 177, 157, 176, 156, 175, 154, 139, 123, 108, 93, 78, 79, 80, 81, 96, 97, 83, 68, 54, 55, 71, 56, 48, 40, 32, 38, 30, 21, 20]

        # positions = []
        # for cell in cells:
        #     positions.append(pose[cell])
        # # range(1200)
        # Line().publish_marker(positions,id_list = range(20000,20000+len(cells)))
       # if cells in dirt_data['cells'].values:
        #         print dirt_data['dirt'].loc[cells]
        #         result[str(cells)] = dirt_data['dirt'].loc[cells]

    def finalize(self):
        folder = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn_server/02-01/high/summary.csv'
        data = pd.read_csv(folder, index_col=[0])
        max_ =np.amax(data['episode_reward']) 
        print max_

    def get_path(self):
        poses = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
        indices = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy')
        dirt_data = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/low/low_02_01.csv')
        result = OrderedDict() 
        print dirt_data
    
    def get_ratio_final_performance(self):
        data_1 = pd.read_csv('/home/mtb/planning_high_04.csv')
        # cells = data_1.loc[np.where(data_1['results'].values== True)[0], 'cells'].drop_duplicates(keep='first') 
        cells = [375, 399, 422, 398, 421, 445, 446, 468, 491, 508, 519, 529, 538, 552, 553, 554, 569, 585, 568, 583, 599, 582, 566, 567, 584, 601, 586, 603, 588, 571, 557, 541, 532, 520, 510, 492, 471, 448, 424, 401, 377, 378, 379, 404, 380, 405, 429, 454, 476, 498, 475, 452, 427, 451, 426, 403]

        all_poses = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/all_cells_pose.npy')
        data_2 = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/test2017-02-19_high.csv')   
        data_3 = pd.read_csv('/home/mtb/Documents/data/dirt_extraction_2/data/high/test2017-02-19_estimation.csv')     
        indices = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy')
        cells_indice = [indices[cell] for cell in cells]
        # print cells_indice  
        myindice = dict(zip(cells,cells_indice))
        # classification = {90:[],75:[],50:[],25:[]}   
        num = 0
        perf = 0
        # print cells
        # print indices
        # print myindice
        # print cells
        # print np.sum(data_2['cleaning_duration'].values)
        # print data_3['dirt_level']
        data_4 = pd.read_csv('/home/mtb/Documents/data/test2017-02-19_high.csv')
        print list(data_2['dirt_level'].values)
        # print sum(list(data_3['dirt_level'].values))/(21.0*60.0)
        print sum(list(data_2['dirt_level'].values))
        # dirt_values = [list(data_3['dirt_level'].values)[i] for i in myindice]
        # print  sum(list(dirt_values))
        # print list(data_2['dirt_level'].values)
        # print cells
        # limit = 50
        # p_x, p_y = data_4['p_x'].values[::limit], data_4['p_y'].values[::limit]
        # positions = np.dstack((p_x,p_y))
        # line_pose =[]
       
        # rospy.init_node('path')
        # for stat in positions[0]:
        #     line_pose.append(list(stat))
        # print cells.values
            
        # color =[]
        # poses =[]
        # for i in range(1296):
        #     if i in indices:
        #         color.append(data_3['dirt_level'].values[i]) 
        #         poses.append(all_poses[i])       
        # dirt_level = map(int , color)
        # del color[:]
        # thresh = 4000.0  # float(max(dirt_level))
        # color = [int(255 - 245 * float(dirt_level[i]) / thresh ) if dirt_level[i] < thresh else 10
        #                  for i in range(len(dirt_level))]
        # # 
        # VizualMark().publish_marker(pose =poses, sizes=[[0.25,0.25,0.0]]*len(color), color=color, convert=True, texture='Red', id_list =range(len(color)))
        # cells =[375,399, 398, 422, 447, 469, 491, 508, 519, 530, 539, 553, 567, 582, 598, 599, 597, 580, 579, 564, 563, 549, 548, 547, 546, 536, 526, 515, 514, 503, 481, 480, 458, 436, 411, 412, 413, 437, 438, 459, 460, 461, 482, 483, 484, 462, 463, 464, 465, 443]#[312, 328, 326, 343, 359, 357, 373, 398, 397, 395, 419, 444, 466, 488, 486, 485, 484, 483, 503, 502, 501, 500, 477, 454, 452, 426, 424, 447, 469, 491, 508, 519, 530, 539, 553, 568, 551, 550, 549, 548, 547, 535, 534, 533, 541, 554]
        # line_pose = [poses[cell] for cell in cells]
        # number = cells[0]
        # from PathPlanning import Path
        # for i in range(0,len(line_pose)):
            # print i 
            # VizualMark().publish_marker([line_pose[i]],sizes=[[0.25,0.25,0.0]], color=['Default'], action='delete',id_list =[cells[i]],publish_num =10)
        # # print len(line_pose) , len(list(cells.values))
            # try:
                # Line().publish_marker(pose=[line_pose[cells.index(number)],line_pose[i+1]], id_list=[number,cells[i+1] ],color='Blue',duration=2.0)
                # Path().initial_pose(line_pose[i])
                # number = cells[i+1]
            # except: pass
        # Line().publish_marker(pose=[line_pose[-1], line_pose[1]], id_list=[cells[-1], cells[0]], color='Blue', duration=2.0)
        # Line().publish_marker(pose=poses, id_list=range(len(poses)))
        # cleaned_cells= []
        # cleaned_poses = []
        # dirt_total_estimation = []
        # dirt_total_real = []
        # # classes ={90: 0, 75:0 , 50:0 ,25:0 ,0:0}
        # for i in range(1296):
        #     if i in cells_indice:
        #             # print i
        #             dirt_data = float(data_2['dirt_level'][i])#/float(data_2['cleaning_duration'][i])
        #             dirt_estimation = data_3['dirt_level'][i]  #* float(data_2['cleaning_duration'][i])     
        #             time = float(data_2['cleaning_duration'][i])
                    

        #             if time >=10.0:
        #                 dirt_total_estimation.append(dirt_estimation)
        #                 dirt_total_real.append(dirt_data)
        #                 cleaned_poses.append(all_poses[i])
        #                 cleaned_cells.append(myindice.keys()[myindice.values().index(i)])
        #                 num += 1
        #                 # print dirt_estimation, dirt_data, time
        #                 value = np.abs(dirt_data - dirt_estimation) / \
        #                     np.max([dirt_estimation, dirt_data])
        #                 if  value<=0.1:
        #                     classes[90] +=1
        #                 elif 0.1<value <=0.25:
        #                     classes[75] += 1
        #                 elif 0.25 < value <= 0.5:
        #                     classes[50] += 1
        #                 elif 0.5 < value <= 0.75:
        #                     classes[25] += 1
        #                 else:
        #                     classes[0] += 1
                        
        #                     # perf +=1
        #             elif 0.0<time<10.0 :
        # #                 cleaned_poses.append(all_poses[i])
        # #                 cleaned_cells.append(myindice.keys()[myindice.values().index(i)])
        #                 dirt_estimation = 0.1 * dirt_estimation * time
        #                 dirt_total_estimation.append(dirt_estimation)
        #                 dirt_total_real.append(dirt_data)
        #                 num += 1
        #                 value = np.abs(dirt_data - dirt_estimation) / \
        #                     np.max([dirt_estimation, dirt_data])
        #                 if value <= 0.1:
        #                     classes[90] += 1
        #                 elif 0.1 < value <= 0.25:
        #                     classes[75] += 1
        #                 elif 0.25 < value <= 0.5:
        #                     classes[50] += 1
        #                 elif 0.5 < value <= 0.75:
        #                     classes[25] += 1
        #                 else:
        #                     classes[0] += 1
                    # elif time ==0.0:
                    #     dirt_total_estimation.append(dirt_estimation)
                    #     dirt_total_real.append(dirt_data)
                        
                        
        #                 num +=1
        #                 value = np.abs(dirt_data - dirt_estimation) / \
        #                     np.max([dirt_estimation, dirt_data])
        #                 if value <= 0.1:
        #                     classes[90] += 1
        #                 elif 0.1 < value <= 0.25:
        #                     classes[75] += 1
        #                 elif 0.25 < value <= 0.5:
        #                     classes[50] += 1
        #                 elif 0.5 < value <= 0.75:
        #                     classes[25] += 1
        #                 else:
        #                     classes[0] += 1
        #                 pass
                    
        # #             else:
        # #                 print 'over'
        # #                 sys.exit(0)

                        
        #             # if dirt_data >0.0:
        #             #     error = np.abs(dirt_data)/ dirt_estimation
        #             #     if error>=0.75:
        #             #         perf +=1                            
        #             #         print dirt_estimation, dirt_data ,error
        #             # else:
        #             #     perf +=1
        #         # except:
        #         #     pass

        # # line_pose = [poses[cell] for cell in cleaned_cells]
        # # VizualMark().publish_marker(cleaned_poses,sizes=[[0.25,0.25,0.0]]*len(cleaned_poses), color=['Default']*len(cleaned_poses), action='delete',id_list =cleaned_cells)

        # # print perf , num , float(perf) / float(num)
        # print classes, num     
        # print sum(dirt_total_estimation) /1291.0, sum(dirt_total_real)/1291.0  

       
        # -----------------------------***********************----------------------------------#
#         b = [0.054,0.054,0.1891,0.21621,0.486]
#         a = [4,8,12,16,20]
        
#         Plot().plot_bar(a,b[::-1],x_labels=['0-25%','25-50%','50-75%','75-90%','90-100%'])
if __name__ == '__main__':
    # HumanInfluence().get_cells_param(save=True)
    # HumanInfluence(sensor='low').plot_dirt_cells()
    # HumanInfluence(sensor='high').get_cells_param(save=True)
    # HumanInfluence().get_params()
    # HumanInfluence().simulation()
    # HumanInfluence().learn()
    # HumanInfluence().get_path()
    # HumanInfluence().plot_estimation()
    # HumanInfluence().visualize_simulation()
    HumanInfluence().get_ratio_final_performance()
        # HumanInfluence().rewrite()
    # HumanInfluence().get_mean_performance()
    # HumanInfluence().plot_mean_exp()
    # HumanInfluence().finalize()
    # HumanInfluence().get_weibull_performance()
    # HumanInfluence().get_average_error()
    # HumanInfluence().rewrite()
    # HumanInfluence().visualize_estimation()
    # a= np.array([3836.8466017964661, 1389.9922162973833, 1801.9477915765306, 3859.1830240035788, 1755.9267817607829, 3098.8290655925034, 1681.5709404270483, 7522.0782516474965, 10894.951813388008, 312.3453905060353, 3688.1598712338332, 2441.5388948627524, 1126.5425021641925, 1891.1560367032121, 1271.4982477120616, 2445.4447738545778, 341.51465569016028, 2653.6173088594091, 2969.0448476150714, 1566.7823647296664, 3172.371692227005, 857.75549877929916, 2345.478721927017, 1853.5962219240453, 1122.2634063642174, 124.49342635407001, 1681.0868880490591, 4762.0882581609285, 707.83075045963608, 1030.8004436501883, 4962.2143793850519, 962.55425529664808, 2331.0963375694901, 3932.3171276844569, 252.93572644383707, 5.8873412785485897, 0.022727007182387068, 1988.491897381879, 3039.7839173750763, 55.305947165481363, 0.024144634818912665, 2223.8944359373877, 695.24024346188855, 15.710217936743225, 3368.3672800449162, 0, 2525.0344699819129, 7.89667440069538, 1575.2770346150153, 1669.5182918237433, 4047.2424747559194, 5055.6074437460356, 4899.7823311887569, 2644.9440007659546, 20805.583101714856, 1715.7216700254387, 256.34518983143357, 1046.1895922802635, 1476.0446007466714, 703.7500576000474, 78.989387638439808, 804.69868853313744, 1020.8146233574538, 1694.1113364909954, 1754.6091519098427, 3489.4707407773744, 2555.564969659426, 553.98110320843466, 4.9945446559638118e-06, 892.58437173712321, 1703.570978888446, 1240.9299004428995, 717.19689313277308, 3762.059221904336, 1015.6041494088462, 6449.981599432137, 2460.6154782627468, 1364.8755344045551, 1275.8260734608527, 1210.4106632444305, 0, 46.916906365689314, 598.88494359849938, 1260.2395449971789, 1047.5953829289142, 3031.0155848915811, 1148.8651137026184, 1.1741286618979283, 1111.3027819622203, 637.1792117830372, 4957.0773307509471, 4803.050804886102, 0, 0, 613.16716000698511, 1675.6492136161853, 1271.0694721778436, 3109.4608666926592, 681.45440403746215, 1785.3795182153772, 0.14209185103582964, 0.0022441611926224021, 129.16005550838989, 977.53565539235694, 394.55070854703462, 64.940912245067437, 2083.3482343473643, 1716.996358016364, 945.50599092495156, 811.17158037972467, 2.1470170268478199, 0.066115636684002749, 112.36388003609966, 148.1333985706801, 1559.1993339620835, 1246.8060873010088, 903.39498914342187, 0.00022282005047906549, 58.265122774258757, 895.26388784692267, 6.1513951365194979, 681.50998715040191, 917.22282687083407, 793.95918780640511, 958.9651112583565, 2595.701775994376, 1440.4992683452908, 699.46140943235207, 492.46996344863368, 5.9559340545804691, 2701.9127328950963, 959.40349466076225, 807.39163094001663, 536.27552524823841, 934.87254985878531, 1721.5436821748194, 1770.9545394101601, 0, 0, 0, 1386.3775959532272, 711.62923247497872, 581.83007896902325, 5.1550711454479012, 1841.3301064315883, 1457.6316319055222, 538.47175846200503, 0, 0, 0, 82.233935198675283, 1042.5976240489863, 1574.3659261619393, 901.48746594824183, 676.10061338463493, 1.4117169374041895e-09, 1636.5300428444546, 660.43021842021653, 0.00025921307906950702, 804.15722612466527, 4.8042328011213153e-11, 4.4397313057322748e-10, 1188.5318197003835, 1969.2493455533561, 0, 0.002221970103683944, 0, 140.12426954591933, 326.15205106890397, 2346.0960215522746, 5886.8204194241152, 633.887219685334, 1316.6527835964102, 6.5004083940454001, 2018.4830277804181, 4683.6392308291934, 1981.744439470501, 2006.5597216165729, 3620.4004356654546, 1804.1596361469074, 2171.0406139542397, 2548.2993436034612, 2314.9793202531378, 1538.3899087006619, 1783.1843680622949, 2649.6816138007489, 2857.3405907740912, 381.63824201218637, 4420.6066276562524, 2487.9843490065255, 932.69681779428765, 6044.029693605401, 2774.6486503107321, 2665.9773055446631, 2626.6660291591161, 2988.9265505167441, 3500.4487355750171, 5.6542380478597405e-17, 0, 0.2192031054246194, 4739.1952737007505, 0.028710947781432763, 1468.5931500433605, 3021.3418506483067, 2.913495279263041e-12, 2.5876009412777998, 2619.4083442572296, 1530.8531973922898, 5259.1312430296739, 4879.8664790032553, 11249.131328568559, 3020.9712666936371, 7722.1656453486476, 1109.5671425225164, 2391.2902037540512, 177.4609263085253, 92.949299113820942, 1087.5211425817288, 557.49481837256587, 0.79628268719822703, 1268.135417298008, 0.88240579098804028, 5483.9439408113667, 0, 3945.5146776774945, 2993.8206079850343, 0.003155679190694192, 0.23946793434823113, 3440.0322182887708, 5.6386106329193536e-05, 85.932684657053883, 1551.8776563149968, 2858.5386483880334, 1484.8265465429722, 497.97133026461734, 101.84024783586361, 2756.3881825724038, 0, 461.12488326816316, 554.05244816669506, 128.33748961987044, 1848.4661846569793, 327.25761022061346, 947.16079623983285, 1916.361540261347, 3440.2528854454317, 240.63849549533973, 1695.8798073706066, 3103.8702271225507, 6.8024941275203011e-06, 1.4491937814630089e-05, 0, 2.9539477943751238, 5181.8807590397046, 5001.7125741798, 130.00585792235663, 1890.4031127567798, 2.7545828071977567, 96.804993113614188, 3550.2077495111953, 2320.1360833081917, 0, 0, 0, 0, 1803.7715054302239, 4875.4804345354469, 1923.8189421437396, 2688.7744361123305, 1863.4034559409463, 1858.5694196631257, 2629.7942335750358, 1883.1107966247544, 0.017043841481795965, 1519.418194565138, 1871.5153057549715, 1091.9483805209866, 3564.3849433458363, 2811.5064403322281, 0, 4258.2147402281798, 1657.7529195574673, 564.02688377457207, 2786.6817557039194, 0, 2252.1017977442402, 2.3623748550396435, 0, 0, 0, 0, 0, 0, 0, 3069.6354573739259, 119.47266471458555, 3189.534610994951, 4734.7872167501173, 2705.4861761979187, 3567.643060294447, 153.83045917062262, 2052.7745870572153, 3194.9404812796938, 277.45440768615657, 312.45560465361945, 505.62565447897998, 2805.1874613928544, 2854.7624993428713, 2601.7084675862361, 5349.3648026293431, 3072.0579755372269, 2768.600349890552, 4064.8160522726348, 3469.7028632433444, 2243.0278258686308, 1437.8679494227763, 3719.4989893115999, 721.70783726078412, 1128.3272390391305, 918.16941083300651, 1544.2146230241817, 2.579607438477535e-21, 1394.75231930924, 383.70165106272509, 3551.423911800709, 861.29399264709446, 1713.0776744364009, 507.26000741734191, 2793.8479339605979, 861.50515315524115, 8.3077585621248307e-11, 380.54064898136039, 663.53506127178696, 1351.9362247012382, 1295.4903005469785, 1079.1412186037937, 0, 1.1504279591690496, 2518.1309105167365, 762.25544106079576, 764.44938869714156, 828.27286983418355, 19.639953916959584, 22.719425704010174, 643.82289375167284, 0.7330037268082138, 1541.0857446349212, 1184.8846995219126, 274.69733062400149, 378.39061638347522, 472.64408997831606, 331.68843342897804, 0, 0.15557901827356257, 2542.4465893848997, 519.39067467835082, 1044.9397248053269, 291.73597538782764, 6.5681016200315261, 9.9061016418652444e-06, 809.14734192308288, 221.50453308145651, 1059.2896497357838, 172.36107108260472, 0.0031925965997783626, 5379.3589223591825, 3260.9045541647133, 0, 0, 2563.7519036452863, 312.54544228111411, 2266.8797250508883, 2133.4891508359747, 1138.7490522326295, 1845.5732935382073, 1278.6249294684835, 477.52922324741013, 1942.8442651279022, 616.82745757333976, 521.58096409167024, 3297.9818187314017, 4636.4713220678823, 0, 1998.4848251485382, 2099.6495206104346, 2448.921407207783, 135.93356788981015, 2402.963513164455, 2898.8654523559749, 1508.7898641578436, 3249.6401696107851, 1734.9827278974715, 3231.8607317168467, 905.53203482135621, 585.71031055619778, 3082.192424191217, 176.10515431748178, 962.93930399064368, 2072.150543338922, 3159.9119054010175, 3663.7910135477819, 1578.5734216522305, 1559.2689497632355, 3260.7032434037005, 2566.1731860064529, 1867.2408324351079, 1853.6128997637586, 0, 3255.643146684291, 1401.9860478098149, 2932.6254638902187, 2824.0409815627959, 3767.0831931207076, 1930.0621206004016, 2047.8684537119502, 0, 2685.1452020349489, 1962.3944930170805, 1737.099335439206, 2793.9345024451191, 0, 0, 1775.815372544944, 707.63808829920606, 769.19267026513455, 1587.7493948589606, 3622.6424589213366, 296.90284784369459, 1810.9787868007929, 0.097406130314943856, 247.46263281250526, 164.97004445955278, 3390.2001206526425, 1.8678954629844555, 1965.2303854490294, 2514.6312176981878, 2668.4990282953172, 2544.488155560718, 1921.3025460204756, 1104.7776790284277, 1310.2253677370852, 1096.8653559233276, 1757.9547487290706, 1613.9456585850135, 3732.8149727427772, 609.98274816807725, 133.42735098271351, 697.89549412971019, 1289.5983398048902, 3768.2282342488288, 2302.1038980116678, 2336.8770573754146, 1992.1456807233926, 535.22255783825335, 1080.8661779130493, 755.6354129233988, 2189.6393641798281, 806.49902131136378, 2408.1072008553642, 1066.5351976235568, 330.48166511469066, 818.26689557880366, 1239.5786751198805, 5639.947122015993, 3267.5233551917763, 441.81532022811041, 1352.9107363562905, 863.82273664354875, 1787.4648119575463, 4226.9307381887065, 7388.3927344456633, 2547.3747723015285, 2457.1486038425232, 3031.2639959285657, 0.0026982297991108967, 885.03203349834462, 1657.3687687154561, 0.085729646793833478, 0, 0, 0, 0, 1044.1844438174387, 3355.7296620308693, 1478.129173158911, 1810.2973624488889, 3326.8421414757408, 196.74740239281414, 0.13169081129166305, 1865.2058990291187, 1392.0610508838511, 2404.0645045778501, 254.87793297294846, 1013.930966920334, 947.28074751254019, 2652.2410336832168, 3592.8250192113778, 1590.6258006653675, 1990.9952465936312, 910.11244540741643, 2159.8957150729602, 3530.7318323741392, 4041.5694564465425, 4744.3853579299011, 2822.8714730199617, 2737.818923048439, 2773.0634460974879, 2558.3395085989009, 2542.6714540588423, 1837.6142278733253, 3197.2177038785044, 1040.4157758774716, 0, 1473.0321633149581, 4434.9136245139716, 453.44330152421139, 1595.3455558576525, 1187.6422512085337, 1672.203092802343, 83.270164402944758, 2304.8435019272983, 3251.4572663785743, 3858.3508816743933, 3240.2700739000597, 1530.0186776314169, 1140.2962327210705, 279.63232523177425, 2955.57744637922, 2293.9968801918635, 1176.5019140753759, 432.01652542444481, 
    #     1072.7059857833208, 0, 4383.2432866422623, 3706.9056909825658, 1727.9073332862142, 3036.6962771926474, 2617.0100373759246, 457.34411492306145, 2518.2067739236818, 690.56018070989978, 464.46966249066674, 34.771858609141063, 2183.8725477473863, 1717.3526774596251, 1723.3717185495493, 7963.0082794668224, 1463.2960766634524, 0, 5551.7743475765446, 1977.9230065094252, 2420.6004461040338, 2999.3338820721542, 213.0597317534189, 1747.5462781147521, 1802.6927462194813, 2590.4675505403411, 3102.3572072777738, 1710.7268367028323, 1135.102591554084, 1174.9548295280972, 2544.1189045855394, 6863.5284035886925, 4353.946494654474, 1641.5475802137605, 7899.7924256421375, 1545.713986700234, 1395.5115959476286, 1516.0900015176621, 3041.4221496742798, 761.50659506554541, 863.16496600075811, 634.34847367410396, 1410.1507820289553, 4749.1809571415115, 547.0005721978506, 500.30718847642419, 1361.6874992178712, 2058.3730550251244, 4537.3271747467206, 4970.2245095942317, 1576.8445505394479, 0, 1568.1404192820785, 251.82085139364736, 0, 0, 0, 0, 2299.5992187492079, 3750.5174348960609, 3312.3989318118242, 2778.2317277986049, 3634.2735687750355, 6716.3183142033704, 0, 6456.9072873471869, 2362.3024031473769, 595.13338000428473, 1701.9488295897322, 1615.9712638465382, 2398.5390159519429, 1685.0994342231129, 2066.4369555866501, 3123.407831893292, 3170.8420318852804, 3365.0806606493129, 1779.9094387278928, 9928.6082877332683, 3580.3971914062836])
    # b = np.array([4501, 547, 2901, 8477, 790, 1582, 525, 1913, 21, 795, 4868, 271, 19, 2126, 808, 3746, 272, 2668, 5914, 1144, 1899, 13, 532, 1644, 534, 322, 23, 2155, 1078, 0, 2508, 1063, 2142, 1619, 1058, 2366, 2148, 3445, 1047, 291, 1576, 527, 1325, 1098, 624, 0, 4257, 284, 348, 784, 800, 266, 1595, 2177, 2133, 3252, 794, 1888, 1346, 1323, 4283, 16, 1198, 1879, 4, 1598, 4630, 3781, 0, 21, 1585, 1090, 514, 3609, 1889, 4011, 2411, 275, 1339, 275, 0, 1599, 3471, 1091, 544, 3139, 1105, 1098, 4023, 284, 6079, 1053, 0, 0, 521, 2119, 16, 1606, 874, 1896, 1853, 1397, 527, 1318, 269, 1352, 4363, 1717, 801, 280, 1647, 1081, 816, 817, 1097, 536, 257, 1867, 535, 811, 1615, 2657, 2180, 2381, 44, 861, 1597, 585, 2669, 811, 807, 540, 851, 340, 5127, 1876, 1627, 0, 0, 0, 1039, 274, 801, 23, 1589, 273, 0, 0, 0, 0, 2420, 274, 3477, 1073, 2, 16, 1564, 2404, 784, 1133, 1598, 105, 1939, 3, 0, 0, 0, 1, 1632, 1361, 1587, 1027, 8, 10, 2726, 2657, 1315, 816, 3704, 1915, 17, 261, 1582, 1085, 1151, 2160, 1075, 518, 2019, 1159, 315, 5502, 2838, 340, 1727, 2834, 1651, 0, 0, 315, 1760, 1644, 1085, 569, 3254, 3489, 4621, 2447, 5631, 4172, 8121, 4661, 17865, 1862, 4863, 315, 571, 270, 825, 1860, 1669, 3539, 1927, 0, 2455, 0, 1396, 866, 301, 0, 1634, 2184, 1596, 273, 305, 809, 1644, 0, 3552, 4394, 1227, 52, 369, 1330, 1044, 267, 1582, 844, 3147, 1552, 3852, 0, 4629, 1884, 1339, 1362, 2019, 659, 3247, 173, 1, 0, 0, 0, 0, 1739, 2124, 2309, 641, 1855, 2659, 1115, 16, 2209, 1636,
    #     336, 525, 1738, 3195, 0, 3609, 6439, 1418, 1667, 0, 2394, 0, 0, 0, 0, 0, 0, 0, 0, 1737, 3007, 2, 100, 4, 2201, 1121, 3137, 56, 2480, 1911, 1860, 1057, 2988, 5302, 2950, 2171, 1448, 4045, 626, 2433, 1678, 1591, 4239, 10, 14, 652, 2402, 1895, 1627, 6526, 3420, 2317, 363, 1, 6381, 354, 8, 7, 1056, 3151, 1465, 0, 2374, 1897, 553, 89, 1117, 530, 797, 1569, 1888, 4264, 5438, 3443, 8, 261, 2, 0, 8621, 790, 19, 1864, 272, 2134, 21, 275, 534, 1033, 5, 5396, 2, 5601, 0, 0, 1349, 590, 276, 599, 528, 1349, 282, 840, 1673, 1388, 2285, 671, 2403, 0, 1405, 91, 566, 2980, 1397, 11, 3691, 5056, 1136, 4258, 4, 3, 2222, 801, 1866, 112, 4277, 263, 1305, 10, 4794, 2160, 4149, 1604, 0, 3275, 291, 313, 1154, 2243, 1812, 629, 0, 101, 789, 283, 2411, 0, 0, 2667, 7, 551, 532, 2642, 35, 1868, 2104, 285, 1339, 4560, 1592, 1912, 2719, 9843, 822, 791, 780, 0, 1, 1261, 1371, 2729, 1083, 1139, 1622, 292, 3951, 1364, 576, 800, 280, 530, 0, 3, 802, 1378, 570, 357, 1056, 532, 303, 3894, 2043, 60, 3746, 2029, 5589, 11495, 354, 2934, 1048, 310, 258, 605, 2140, 0, 0, 0, 0, 1322, 1322, 280, 1153, 2049, 280, 542, 1902, 283, 1384, 7, 1898, 536, 611, 2992, 1664, 1874, 3764, 2811, 555, 1659, 3230, 2014, 3464, 1765, 1789, 2131, 562, 1264, 521, 0, 1548, 525, 1879, 301, 169, 2252, 292, 1453, 6664, 1159, 1776, 264, 27, 2176, 4659, 1940, 2, 331, 1688, 0, 10024, 4045, 2420, 2042, 554, 7, 811, 257, 592, 266, 1067, 2519, 2870, 1717, 2094, 0, 1224, 552, 1103, 3934, 287, 795, 1075, 555, 0, 1663, 921, 4, 4321, 2832, 263, 2131, 2267, 3416, 4505, 267, 787, 1109, 862, 2547, 1069, 1572, 275, 287, 1396, 2385, 6099, 3503, 4356, 0, 2008, 284, 0, 0, 0, 0, 4046, 1666, 1097, 0, 2186, 4275, 0, 15883, 4561, 0, 279, 530, 10, 5393, 3217, 7372, 6744, 1423, 2952, 12123, 4984])
    # d = {'90': 0,'75': 0,'50':0,'25':0,'0':0}

    # # print max(a)
    # # print max(b)
    # max_a = 20805.5831017
    # max_b = 17865
    # indice_a = []
    # indice_b = []
    # corresp = 0
    # for i in range(610):
    #     if (float(a[i]) / float(max_a)) >=0.2:
    #         indice_a.append(i)
    #     if (float(b[i]) / float(max_b)) >=0.5:
    #         indice_b.append(i)
        
    #     if i in indice_a and i in indice_b:
    #         corresp +=1

    # print indice_b          
    # for i in range(610):
    #     if (np.fabs(a[i] - b[i]) / max(a[i], b[i])) <=0.1:
    #         d['90'] +=1
    #     elif 0.1<(np.fabs(a[i] - b[i]) / max(a[i], b[i]))<=0.25:
    #         d['75'] +=1
    #     elif 0.25<(np.fabs(a[i] - b[i]) / max(a[i], b[i]))<=0.5:
    #         d['50'] +=1
    #     elif 0.5<(np.fabs(a[i] - b[i]) / max(a[i], b[i]))<=0.75:
    #         d['25'] +=1
    #     else:
    #         d['0'] +=1
    # for t in range(4,-1,-1):
    #     print t

    # a = [400,6,7,825,500,9,200]
    # for i in a:
    #     if i%100 == 0:
    #         print i
    # print 4 // 23
    
    

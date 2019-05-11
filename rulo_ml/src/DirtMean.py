#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.graph_plot import Plot
from rulo_base.markers import VizualMark
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
        self.rewrite_human_file(rewrite=False)
        self.dirt_cells = OrderedDict()

        for cells in self.cells:
            self.dirt_cells[str(cells)] = OrderedDict()
        
        for files in sorted(os.listdir(dirt_sensor_file + self.sensor + '/dirt_value/'))[-10:]:
            dirt_data = csvread(dirt_sensor_file + self.sensor + '/dirt_value/' + files)
            human_data = csvread(result_file + 'human_time_in_cells/' + files)
            for cells in self.dirt_cells:
                try:
                    self.dirt_cells[cells]['dirt_level'].append(int(dirt_data['dirt_level'][int(cells)]))
                    self.dirt_cells[cells]['wall_time'].append(float(dirt_data['wall_time'][int(cells)]))
                    self.dirt_cells[cells]['cleaning_duration'].append(float(dirt_data['cleaning_duration'][int(cells)]))
                    self.dirt_cells[cells]['human_frequency'].append(float(human_data['human_frequency'][int(cells)]))              
                except:
                    self.dirt_cells[cells]['dirt_level'] = [int(dirt_data['dirt_level'][int(cells)])]
                    self.dirt_cells[cells]['wall_time'] =[float(dirt_data['wall_time'][int(cells)])]
                    self.dirt_cells[cells]['human_frequency'] =[float(human_data['human_frequency'][int(cells)])]
                    self.dirt_cells[cells]['cleaning_duration'] =[float(dirt_data['cleaning_duration'][int(cells)])]
        # print self.dirt_cells['116']
        if save :
            for cells in self.dirt_cells:
                dataframe_human = DataFrame(data=self.dirt_cells[cells]['human_frequency'],
                                            index=self.dirt_cells[cells]['wall_time'], columns=['human_frequency'])
                dataframe_dirt = DataFrame(data=self.dirt_cells[cells]['dirt_level'],
                                        index=self.dirt_cells[cells]['wall_time'], columns=['dirt_level'])
                dataframe_cycle = DataFrame(data=self.dirt_cells[cells]['cleaning_duration'],
                                           index=self.dirt_cells[cells]['wall_time'], columns=['cleaning_duration'])
                dataframe = pd.merge(dataframe_dirt, dataframe_human,right_index=True, left_index=True)    
                dataframe = pd.merge(dataframe, dataframe_cycle,right_index=True, left_index=True)    

                path = self.makedir(result_file + 'human_data_arranged/'+self.sensor +'/')
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
            self.last_wall_time[str(elem)] = 1510026191.80 
    
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
        folder = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/human_data/high/data/'
        for files_num in range(966,1250):
            if os.path.isfile(folder + str(files_num)+'.csv'):
                # filename = raw_input('Enter_filename: ')            
                data = pd.read_csv(folder + str(files_num)+'.csv',index_col=[0])
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
                                dirt[keys] += data['dirt_level'].values[i]
                                times[keys].append(i)
                                break                                 
                    for keys in dirt:
                        if len(times[keys]) >0:
                            dirt[keys] = dirt[keys] / len(times[keys])                   
                    Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
                    saving = raw_input('Do you wanna save?: ')
                    if str(saving) == 'y':
                        skip_id = [raw_input('Enter skip id: ')]
                        csvwriter('/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull.csv',
                                    headers=['cells','step','k','skip_id'], 
                                    rows = [[files_num],[step],[k+1],[skip_id]])
                        break
                    else:
                        continue

                    # if files[:-4] == '116':
                    #     # self.get_weibull_distribution(files[:-4],dirt)
                    #     Plot().scatter_plot(dirt.keys(), dirt.values(), show=True) 
                    #     break
        
            else:
                pass
    
    def learn(self):
        folder = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/human_data/high/data/'
        
        file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull.csv'
        data = csvread(file)
        # while True:
        for p in range(618,len(data['cells'])):
            # p = int(raw_input('Enter p: '))
            cells = data['cells'][p]
            step = float(data['step'][p])
            k = int(data['k'][p])
            skip_id = data['skip_id'][p]
            # id_to_skip = []
            # for elem in skip_id.split(','):
            #     id_to_skip.append(elem)
            print cells
            print skip_id
                # break
            # break
                # id_to_skip.append(int(elem))
            while True:
                data_1 = pd.read_csv(folder + cells + '.csv', index_col=[0])
                times = OrderedDict()
                dirt = OrderedDict()
                for l in range(k):
                    times[step * float(l + 1)] = []
                    dirt[step * float(l + 1)] = 0.0
                for i in range(data_1.index.values.shape[0]):            
                    for keys in times:
                        indice = times.keys().index(keys)
                        if step *float(indice) <=data_1.index.values[i]<=keys:
                            dirt[keys] += data_1['dirt_level'].values[i]
                            times[keys].append(i)
                            break                                 
                for keys in dirt:
                    if len(times[keys]) >0:
                        dirt[keys] = dirt[keys] / len(times[keys])                   
                Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
                prompt_1 = raw_input('Do you wanna learn?')
                if str(prompt_1) =='y':
                    skip_id = []
                    prompt_2 = raw_input('Enter skip id: ')
                    for elem in str(prompt_2).split(','):
                        skip_id.append(int(elem))
                    # try:
                    self.get_weibull_distribution(cells, skip_id, dirt)
                    # except KeyboardInterrupt:
                    #     pass
                    break
                elif str(prompt_1) == 'c':
                    sys.exit(0)
                # elif 
                else:
                    break
                
    def get_weibull_distribution(self,cells,skip_id=[],*args):
        t = tf.placeholder(tf.float32)
        dirt_data= tf.placeholder(tf.float32)
        beta = tf.Variable(initial_value= 2.0)
        eta = tf.Variable(initial_value=1.0)
        A = tf.Variable(initial_value=3000.0)
        f1 = tf.divide(beta, tf.pow(eta, beta) )
        f2 = tf.pow(t, (beta -1))
        f3 = tf.exp(-(t/eta)**beta)
        f_t =  A * f1 * f2 * f3
       
        loss = dirt_data - f_t
        cost = tf.reduce_mean(tf.pow(loss,2))
        optimize_BE = tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
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

            try:
                for i in range(1000000):
                    opt1,opt2 , error,  var1, var2 ,var3=  sess.run([opt_A,opt_B, cost,A, beta,eta], {t: times_param, dirt_data: y })
                    
                    print cells,np.sqrt(error), var1,var2, var3
            except KeyboardInterrupt:
                # prompt_3 = raw_input('Do you wanna save the parameters? ')
                # if str(prompt_3) == 'y':
                #     prompt_4 = raw_input('Enter success or error')
                    csvwriter(
                        '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_learn_2.csv',
                        ['cells', 'error', 'A', 'beta', 'eta', 'mode','cycle'], [[cells], [np.sqrt(error)], [var1], [var2], [var3], ['s'],[cells_cycle]])
                                
                # else:
                #     pass
                    
        

            # A, beta, eta =1184.2 ,3.67853 ,0.505138
            # f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A for t in times_param]
            # plt.plot(times_param, y,'ro', times_param, f_t , 'go')
            # plt.show()

    def plot_estimation(self):
        est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_learn_2.csv'
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/human_data/high/data/'
        data_1 = pd.read_csv(est_file)
        a = np.array(data_1['cells'].values,dtype='str')
        # print data_1
        for files in os.listdir(dirt_fold):
            data_2 = pd.read_csv(dirt_fold + files,index_col=[0])
            # A, eta, beta = data_1[np.wheredata_1[files[:-4]].values]
            if files[:-4] !='966':            
                index =  np.where(int(files[:-4]) == data_1['cells'].values)[0][0]
                A, beta, eta, cycle= data_1.iloc[index]['A'], data_1.iloc[index]['beta'], data_1.iloc[index]['eta'], data_1.iloc[index]['cycle']
                dirt = data_2['dirt_level'].values
                duration = data_2.index.values
                x_val = []
                y_val = []
                if np.amax(duration) > cycle:
                    cycle_time =int(np.amax(duration) / cycle)
                    for m in range(cycle_time):
                        times_par = np.linspace(m*cycle, (m+1)*cycle, 10)
                        times_param = []
                        for elem in times_par:
                           
                            times_param.append(elem / ((m + 1) * cycle))
                        # print times_param
                        if m ==0:
                            f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A for t in times_param]                        
                        for i in range(len(f_t)):
                            x_val.append(times_par[i])
                            y_val.append(f_t[i])
                    plt.plot(duration, dirt, 'ro')
                    plt.plot(x_val, y_val)
                    plt.xlabel('Cleaning_cycle')
                    plt.ylabel('Dirt')
                    plt.title('High Dirt estimation for cells ' + files[:-4])
                    plt.savefig(
                        '/home/mtb/Documents/data/dirt_extraction_2/result/estimation_fig/' + files[:-4] +'.png')
                    plt.close()
                    
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
                if i ==116:
                    print float(data_2['dirt_level'].values[i])
                mean_day[str(i)]['mean'].append(float(data_2['dirt_level'].values[i]))
                mean_day[str(i)]['ratio'].append(float(data_2['dirt_level'].values[i])/float(np.max(data_2['dirt_level'])))
        
        for keys in mean_day:
            if keys in mean_result:
                mean_by_cleaning[keys]['mean']= np.mean(mean_day[keys]['mean'])
                mean_by_cleaning[keys]['ratio']= np.mean(mean_day[keys]['ratio'])
        result=OrderedDict()

        # print mean_by_cleaning['116']
        # print mean_result['116']

        # for keys in mean_by_cleaning:
        #     result[keys]= OrderedDict()     
        #     result[keys]['estimation'] = mean_result[keys]
        #     result[keys]['average_by_day'] = mean_by_cleaning[keys]
        
        # print result['116']

if __name__ == '__main__':
    # HumanInfluence('high').get_cells_param(save=True)
    # HumanInfluence(sensor='low').plot_dirt_cells()
    # HumanInfluence(sensor='high').get_cells_param(save=True)
    # a = DataFrame(data=[1, 2, 3], 
    #             index=['b', 'c','a'])
    # b = DataFrame(data=[3, 2, 3],
    #            index=['b', 'c', 'a'])
    
    # c = pd.merge(a,b,right_index=True, left_index=True)'
    # c.columns = ['hum', 'dkjhk']
    # print c
    # HumanInfluence().get_params()
    # HumanInfluence().learn()
    HumanInfluence().get_mean_performance()

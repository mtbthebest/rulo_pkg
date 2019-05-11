#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from collections import OrderedDict
from time import sleep
dirt_sensor_file = '/home/mtb/Documents/data/dirt_extraction_2/data/'
result_file = '/home/mtb/Documents/data/dirt_extraction_2/result/air_influence/'

class AirInfluence:
    def __init__(self,sensor='high'):
        self.sensor= sensor
        self.cells = ['838','839','840','841','842','865','866','867','868','869']
        print 'Getting data'
        self.get_data()
        self.marker = 'r' if self.sensor == 'high' else 'g'
        
        
    def get_data(self):
        '''
        Reorder data in function of cleaning timestep, dirt level , and cleaning duration
        '''
        file_data = self.get_cells_dirt_and_wall_time()
        # print file_data['838']
        self.dirt_cell = self.reorder_cells_cleaning_time(file_data)    
        # print self.dirt_cell['838']
    
    def get_cells_dirt_and_wall_time(self):
        '''
        Return a dictionary of cells dirt and cleaning time
        '''
        path = dirt_sensor_file + self.sensor + '/dirt_value/'
        res = OrderedDict()        
        for cell in self.cells:
                res[cell] = OrderedDict()
                res[cell]['dirt_level'] = []
                res[cell]['wall_time'] = []
                res[cell]['cleaning_duration'] = []
                for files in sorted(os.listdir(path)):
                    # print files
                    data = csvread(path + str(files))
                    res[cell]['dirt_level'].append(int(data['dirt_level'][int(cell)]))
                    res[cell]['wall_time'].append(float(data['wall_time'][int(cell)]))
                    res[cell]['cleaning_duration'].append(float(data['cleaning_duration'][int(cell)]))
        return res
    
    def reorder_cells_cleaning_time(self, file_data):
        '''
        Return a dictionary of cells dirt and cleaning time with the non cleaned cells deleted from the data to process
        '''
        dirt_cell = OrderedDict()
        for cell in file_data:
            dirt_cell[cell] = OrderedDict()
            dirt_cell[cell]['dirt_level'] = []
            dirt_cell[cell]['wall_time'] = []
            dirt_cell[cell]['cleaning_duration'] = []
            for j in range(len(file_data[cell]['wall_time'])):            
                if not file_data[cell]['wall_time'][j] == 0.0:
                    dirt_cell[cell]['dirt_level'].append(file_data[cell]['dirt_level'][j])
                    dirt_cell[cell]['wall_time'].append(file_data[cell]['wall_time'][j])
                    dirt_cell[cell]['cleaning_duration'].append(file_data[cell]['cleaning_duration'][j])
                    # print cell, j, file_data[cell]['dirt_level'][j]

        return dirt_cell 
            
    def get_dirt_increase_in_time(self, dataframe, cell,path, save=False):
        '''
        Plot the increase of dirt during time 
        '''
        index = dataframe.index.values
        data = dataframe.values
        dirt_increase , time_increase = [data[0][0]], [index[0]]
        self.dirt_num.append(data[0][0])
        self.time.append(index[0])
        for i in range(1,index.shape[0]):
            time_increase.append(np.sum(index[:i+1]))
            dirt_increase.append(np.sum(data[:i+1]))
        for l in range(len(time_increase)):
            self.time.append(time_increase[l])
            self.dirt_num.append(dirt_increase[l])
        
        
        if save:
            csvwriter(path+'cells/' + cell+'.csv',
                    headers=['duration', 'dirt_level'], rows=[time_increase, dirt_increase])
            Plot().scatter_plot(x_value=time_increase, y_value=dirt_increase, marker=self.marker + 'o', 
            title='Dirt increase detected by the ' + self.sensor + ' level sensor in the cell ' + cell,
            labels =['Cleaning Cycle', 'Dirt Level'],save_path=path + 'plot/'+cell+'.png')
    
    def process_dirt_increase(self, scaling=False,save=False):    
        self.time = []
        self.dirt_num = []         
        for cell in self.dirt_cell:
            duration = [0.0]            
            for j in range(len(self.dirt_cell[cell]['wall_time'])-1):
                duration.append(self.dirt_cell[cell]['wall_time'][j+1]- self.dirt_cell[cell]['wall_time'][j])
    
            if not scaling:
                dataframe = DataFrame(data=np.array(self.dirt_cell[cell]['dirt_level']), index = duration, columns=[cell])
                path = result_file + 'dirt_increase' +'/no_scaling/'+self.sensor +'/'                
            else:
                dataframe = DataFrame(data=np.array(self.dirt_cell[cell]['dirt_level'],dtype='float')/ np.array(self.dirt_cell[cell]['cleaning_duration'],dtype='float'), 
                index = duration, columns=[cell])  
                path = result_file + 'dirt_increase' +'/scaling/'+self.sensor +'/'     
            if not os.path.isdir(path):
                os.makedirs(path+'cells/')
                os.makedirs(path+'plot/')       
                
            self.get_dirt_increase_in_time(dataframe, cell,path,save)

        Plot().scatter_plot(x_value=self.time, y_value=self.dirt_num, marker=self.marker + 'o', title='Dirt increase detected by the ' +self.sensor+ ' level sensor in the cells ',
                            labels=['Cleaning Cycle', 'Dirt Level'], save_path=path + 'plot/' + self.sensor+ '_level.png')

    def process_dirt_variation(self, scaling = False,save=False):
        x_val = []
        y_val = []
        for cell in self.dirt_cell:
            duration = [0.0]
            self.time = []
            self.dirt_num = []
            for j in range(len(self.dirt_cell[cell]['wall_time']) - 1):
                duration.append(self.dirt_cell[cell]['wall_time'][j + 1] - self.dirt_cell[cell]['wall_time'][j])
            for l in range(len(duration)):
                self.time.append(duration[l])
                x_val.append(duration[l])
                if scaling :
                    value = float(self.dirt_cell[cell]['dirt_level'][l]) / float(self.dirt_cell[cell]['cleaning_duration'][l])                    
                else:
                    value = float(self.dirt_cell[cell]['dirt_level'][l])
                self.dirt_num.append(value)
                y_val.append(value)
            if scaling:
                path = result_file + 'dirt_variations' +'/scaling/'+self.sensor +'/'
            else:
                path = result_file + 'dirt_variations' +'/no_scaling/'+self.sensor +'/'
            if not os.path.isdir(path):                
                os.makedirs(path+'cells/')
                os.makedirs(path+'plot/')
            if save:
                csvwriter(path + 'cells/' + cell + '.csv',
                      headers=['duration', 'dirt_level'], rows=[self.time, self.dirt_num])
                Plot().scatter_plot(x_value=self.time, y_value=self.dirt_num, marker=self.marker + 'o', 
                                title='Dirt sucked by the ' + self.sensor + ' level sensor in the cell ' + cell,
                                labels=['Cleaning Cycle', 'Dirt Level'], save_path=path + 'plot/' + cell + '.png')
        if save:
            Plot().scatter_plot(x_value=x_val, y_value=y_val, marker=self.marker + 'o', 
                           title='Dirt sucked by the ' + self.sensor + ' level sensor in the cells ',
                            labels=['Cleaning Cycle', 'Dirt Level'], save_path=path + 'plot/' + self.sensor + '_level.png')

    def get_cells_dirt_sucked_time(self, plot_and_save=False, train=False):
        path_dirt = dirt_sensor_file + self.sensor + '/dirt_sucked_by_cells/'
        path_time = dirt_sensor_file + self.sensor + '/time/'
        # path_dirt = dirt_sensor_file + self.sensor + '/dirt_sucked_by_cells/' + filename
        # path_time = dirt_sensor_file + self.sensor + '/time/' + filename
        res = DataFrame()
        self.step = OrderedDict()

        for filename in sorted(os.listdir(path_dirt))[3:10]:         
            # print filename
            res = DataFrame({'time':np.load(path_time + filename ),'dirt':np.load(path_dirt +filename)},index=range(1296))         
            for cells in self.cells:
                if len(res.loc[int(cells),'dirt']) ==1:
                    pass
                else:
                    dirt = []
                    if plot_and_save:
                        if not os.path.isdir(result_file + 'dirt_sucking_time/' + self.sensor +'/' + cells +'/' ):
                            os.makedirs(result_file + 'dirt_sucking_time/' + self.sensor +'/' + cells +'/var/')
                            os.makedirs(result_file + 'dirt_sucking_time/' + self.sensor +'/' + cells +'/increase/')                        
                            dirt = []
                            for i in range(len(res.loc[int(cells),'dirt'])):
                                dirt.append(sum(res.loc[int(cells),'dirt'][:i+1]))
                            # print len(dirt), len(res.loc[int(cells),'dirt'])
                            Plot().scatter_plot(x_value=res.loc[int(cells),'time'], y_value =res.loc[int(cells),'dirt'],marker=self.marker + 'o',
                                title = files[9:14],save_path= result_file + 'dirt_sucking_time/' + self.sensor +'/' + cells +'/var/' + files[9:14]+'.png')
                            Plot().scatter_plot(x_value=res.loc[int(cells),'time'], y_value =dirt,marker=self.marker + 'o',
                                title = files[9:14],save_path= result_file + 'dirt_sucking_time/' + self.sensor +'/' + cells +'/increase/' + files[9:14]+'.png')
                    if train:
                        if not dirt:
                            for i in range(len(res.loc[int(cells), 'dirt'])):
                                dirt.append(sum(res.loc[int(cells), 'dirt'][:i + 1]))
                        print 'Training ' + cells + ' ' + filename
                        # print dirt, res.loc[int(cells), 'time']
                        # try:
          
                        if not os.path.isfile(result_file + 'dirt_sucking_time/' + self.sensor + '/param/' + filename[:-4] + '/'+cells +'.csv'):                                
                            error, a,b = self.train(cells,res.loc[int(cells),'time'],dirt,mode='exp')
                            if not os.path.isdir(result_file + 'dirt_sucking_time/' + self.sensor + '/param/' + filename[:-4]):
                                        os.makedirs(result_file + 'dirt_sucking_time/' +
                                                    self.sensor + '/param/' + filename[:-4])
                            csvwriter(result_file + 'dirt_sucking_time/' + self.sensor + '/param/' + filename[:-4] + '/'+cells +'.csv', 
                                            headers=['cells', 'error', 'a', 'b','filename'],
                                            rows = [[cells],[error], [a],[b],[filename[:-4]+'.csv']])
                            
    def train(self,cells,features,labels,mode='linear'):
        if mode == 'linear':
            i,j = np.random.choice(range(len(labels))),np.random.choice(range(len(labels)))
            # print features, labels
            # return True
            
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)
            a = tf.Variable(initial_value= tf.divide(tf.abs(float(labels[i]) - float(labels[j])), tf.abs(float(features[i])-float(features[j]))))
            b = tf.Variable(initial_value= float(labels[i]) - tf.divide(tf.abs(float(labels[i]) - float(labels[j])), tf.abs(float(features[i])-float(features[j]))) * float(features[i]))

            y_ = tf.add(tf.multiply(a,x) , b)
            loss = y - y_
            cost = tf.reduce_mean(tf.pow(loss,2))
            optimize = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                diff = 10000
                step =0
                try:
                    for i in range(2500000):
                        grad, error, var1, var2 = sess.run(
                            [optimize, cost, a, b], {x: features, y: labels})
                       
                        if i %500000 ==0:
                            print cells, np.sqrt(error)
                            try: 
                                new_error = np.sqrt(error)                               
                                if new_error == last_error:
                                    break
                            except:
                                pass
                        last_error = np.sqrt(error)

                except KeyboardInterrupt:
                    pass
                return np.sqrt(error), var1, var2
        
        if mode  == 'exp':
          
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)
            m_t = tf.Variable(initial_value= float(max(labels)))
            tau= tf.Variable(initial_value= float(max(features)) + 100.0)

            y_ = m_t *(1-tf.exp(-x/tau))
            loss = y - y_
            cost = tf.reduce_mean(tf.pow(loss, 2))
            optimize = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    for i in range(2500000):
                        grad, error, var1, var2 = sess.run(
                            [optimize, cost, m_t, tau], {x: features, y: labels})

                except KeyboardInterrupt:
                    pass
                return np.sqrt(error), var1, var2
            
    def express_param(self):
        res = OrderedDict()
        param_dict = OrderedDict()
        cells_path = result_file+'dirt_variations/no_scaling/' +self.sensor+'/cells/'
        param_path = result_file + '/dirt_sucking_time/' + self.sensor + '/param/'        
        for cells_file in sorted(os.listdir(cells_path)):
                data = csvread(cells_path + cells_file)
                duration = map(float, data['duration']  )
                res[cells_file[:-4]] = duration
                param_dict[cells_file[:-4]] = OrderedDict()  
                param_dict[cells_file[:-4]]['a'] = []
                param_dict[cells_file[:-4]]['b'] = []
                param_dict[cells_file[:-4]]['duration'] = duration
        
        for folders in sorted(os.listdir(param_path)):
            for files in sorted(os.listdir(param_path + folders +'/')):
                data = csvread(param_path + folders + '/' + files)
                a = map(float,data['a'])
                b = map(float, data['b'])
                param_dict[files[:-4]]['a'].append(a[0])
                param_dict[files[:-4]]['b'].append(b[0])
       
        for keys in param_dict:
            data = OrderedDict([('a', []), ('duration', [])])
            for i in range(len(param_dict[keys]['a'])):
                if not np.isnan(param_dict[keys]['a'][i]):
                    data['a'].append(param_dict[keys]['a'][i])
                    data['duration'].append(param_dict[keys]['duration'][i])
        # print data['duration']
            Plot().scatter_plot(data['duration'],data['a'], show=True)
        # x = range(10000)
        # z = [1 - np.exp(-float(x[i]) / 1000.0)
        #      for i in range(len(x))]
        # y=[1- np.exp(-(float(x[i])-2000.0)/ 1000.0) for i in range(len(x))]
        # plt.plot(x,y,x,z)
        # plt.show()      
        

    def get_params(self):
        folder = '/home/mtb/Documents/data/dirt_extraction_2/result/air_influence/dirt_variations/no_scaling/high/cells/'
        step = 100000.0
        for files in sorted(os.listdir(folder)):
            print files
            data = pd.read_csv(folder + files)
            # print data['duration']
            times = OrderedDict()
            dirt = OrderedDict()
            for i in range(6):
                times[step * float(i+1)] = []
                dirt[step * float(i + 1)] = 0.0

            for i in range(data['duration'].shape[0]):            
                for keys in times:
                    indice = times.keys().index(keys)
                    if step *float(indice) <=data['duration'][i]<=keys:
                        dirt[keys] += data['dirt_level'][i]
                        times[keys].append(i)
                        break

            for keys in dirt:
                if len(times[keys]) >0:
                    dirt[keys] = dirt[keys] / len(times[keys])
                
            # Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
            if files[:-4] == '838':
                # self.get_weibull_distribution(files[:-4],dirt)
                Plot().scatter_plot(dirt.keys(), dirt.values(), show=True)
            # self.get_factor(files[:-4], dirt)
            # self.verify_param(files[:-4], dirt)
            # print dirt
                break
    
    
    def get_weibull_distribution(self,cells,*args):
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
        optimize_A = tf.train.GradientDescentOptimizer(learning_rate =0.1)

        grads_A = optimize_A.compute_gradients(cost,[A])
        grads_BE = optimize_BE.compute_gradients(cost, [beta, eta])
        opt_A = optimize_A.apply_gradients(grads_A)
        opt_B = optimize_BE.apply_gradients(grads_BE)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # times_param = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys())) ]
            # y  = [args[0].values()[i] for i in range(len(args[0].values()))]
            #if i!=5

            times_param = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys()))  
                       if i !=5  ]
            y  = [args[0].values()[i] for i in range(len(args[0].values())) 
                   if i !=5 ]

            print y
            print times_param


            # for i in range(100000):
            #     opt1,opt2 , error,  var1, var2 ,var3=  sess.run([opt_A,opt_B, cost,A, beta,eta], {t: times_param, dirt_data: y })
            #     print np.sqrt(error), var1,var2, var3




            # A, beta, eta = 1239.98,3.21712,0.504754
            # f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) * A for t in times_param]
            # plt.plot(times_param, y,'ro', times_param, f_t , 'go')
            # plt.show()



















        
    # def get_weibull_distribution(self,cells,*args):
    #     t = tf.placeholder(tf.float32)
    #     Q_t = tf.placeholder(tf.float32)
    #     beta = tf.Variable(initial_value= 0.5)
    #     eta = tf.Variable(initial_value=1.0)
    #     y_op = tf.log(tf.log(tf.divide(1,1-Q_t))) 
    #     y_apx = beta * tf.log(t) - beta * tf.log(eta)
    #     loss = y_op - y_apx
    #     cost = tf.reduce_mean(tf.pow(loss,2))
    #     optimize = tf.train.GradientDescentOptimizer(learning_rate=0.0010).minimize(cost)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         times_param = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys())) if i!=2]
    #         # dirt = [args[0].values()[i] /8500.0 for i in range(len(args[0].values()))]
    #         y  =dirt = [args[0].values()[i] for i in range(len(args[0].values())) if i!=2]
    #         dirt = [args[0].values()[i]/ 8000.0 for i in range(len(args[0].values())) if i!=2]
    #         dirt_sum = [sum(dirt[:i]) for i in range(1,len(dirt)+1)]
    #         print times_param
    #         print dirt
    #         print dirt_sum

    #         # for i in range(1000000):
    #         #     opt , error, var1 , var2 =  sess.run([optimize, cost, beta,eta], {t: times_param, Q_t: dirt_sum})
    #         #     print np.sqrt(error), var1,var2

    #         beta, eta = 1.60998 , 0.459749 
    #         f_t = [(beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta) *1900.0 for t in times_param]
    #         print f_t            
    #         plt.plot(times_param, y,'ro', times_param, f_t , 'go')
    #         plt.show()
    
    def get_factor(self, cells,*args):
        
        dirt_data = tf.placeholder(tf.float32)
        dirt_scale  = tf.placeholder(tf.float32)
        factor = tf.Variable(initial_value=10000.0)
        dirt_est = tf.multiply(factor , dirt_scale)
        loss = dirt_data - dirt_est
        cost = tf.reduce_mean(tf.pow(loss,2))
        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.010).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            f_t = []
            times_param = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys()))]
            beta,eta = 1.46211, 0.536181
            for t in times_param:
                f_t.append((beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta))
            dirt = [args[0].values()[i] for i in range(len(args[0].values()))]
   
            for i in range(100000):
                opt, error, var = sess.run([optimize, cost, factor], {dirt_data: dirt, dirt_scale: f_t})
                print np.sqrt(error), var
            # print dirt

    def verify_param(self,cells, *args):
        beta,eta = 1.46211 ,0.9#0.536181
        times = [args[0].keys()[i] / max(args[0].keys()) for i in range(len(args[0].keys()))]
        dirt = [args[0].values()[i] for i in range(len(args[0].values()))]
        f_t = []
        for t in times:
            f_t.append((beta / eta ** beta) * (t **(beta - 1)) * np.exp(-(t/eta)**beta)*4000.0)
        
        plt.plot(times, dirt,'ro', times, f_t, 'bo')
        plt.show()
    
  
    def plot_estimation(self):
        est_file = '/home/mtb/Documents/data/dirt_extraction_2/result/weibull/human/high/weibull_learn_2.csv'
        dirt_fold = '/home/mtb/Documents/data/dirt_extraction_2/result/air_influence/dirt_variations/no_scaling/high/cells/'
        data_1 = pd.read_csv(est_file)
        a = np.array(data_1['cells'].values,dtype='str')
        # print data_1
        for files in os.listdir(dirt_fold):
            data_2 = pd.read_csv(dirt_fold + files)
            print data_2
          
            if files[:-4] !='966':            
                index =  np.where(int(files[:-4]) == data_1['cells'].values)[0][0]
                A, beta, eta, cycle= data_1.iloc[index]['A'], data_1.iloc[index]['beta'], data_1.iloc[index]['eta'], data_1.iloc[index]['cycle']
                dirt = data_2['dirt_level'].values
                duration = data_2['duration'].values
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
                    # plt.show()
                    plt.savefig(
                        '/home/mtb/Documents/data/dirt_extraction_2/result/estimation_fig/' + files[:-4] +'.png')
                    plt.close()
    
    def get_mean_performance(self):
        pass

            
            
        
if __name__ == '__main__':
    # AirInfluence().get_params()
    AirInfluence().plot_estimation()
    # AirInfluence().verify_param()
    # AirInfluence(sensor='low').process_dirt_variation()
    # AirInfluence(sensor='low').process_dirt_increase(scaling = True,save=True)
    # AirInfluence(sensor='high').express_param('air')

    # AirInfluence(sensor='high').get_cells_dirt_sucked_time(train=True)
    # AirInfluence(sensor='high').express_param()
    

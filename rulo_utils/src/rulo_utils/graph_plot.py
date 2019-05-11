#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


class Plot:
        def __init__(self):
                pass      
        def scatter_plot(self,x_value= None, y_value = None, marker='ro',axis_max=[], title=None,labels =[], save_path=None, show=False):   
                # fig  = plt.figure()
                plt.plot(x_value, y_value, marker,markersize =3)
                if axis_max:
                    plt.axis([0, axis_max[0],0,axis_max[1]])
                if labels:
                    try:
                        plt.xlabel(labels[0])
                        plt.ylabel(labels[1])
                    except:
                        pass
                if title:
                        plt.title(title,loc='center')                
                if save_path:
                        plt.savefig(save_path)
                if show:
                        plt.show()
                plt.close()
        def plot_bar(self, x =[], y=[], x_labels= [], y_labels =[], width=1, x_axis ='', y_axis ='', labels_position = 'vertical', align='center'):
                plt.xticks(x, x_labels, rotation=labels_position)
                plt.bar(x, y, width=width, align= align)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis) 
                # plt.ylim(ymin =0.0, ymax=1.0)               
                plt.show()
        def plot_and_save(self, fig_type='line', marker ='ro',x =[], y=[],  width=1, color = 'red',save_path=''):
                if fig_type =='line':
                        fig = plt.figure()
                        plt.plot(x, y)
                        plt.savefig(save_path)
                        plt.close(fig)
                elif fig_type =='bar':
                        fig =  plt.figure()
                        plt.bar(x, y, width=width,color=color)
                        plt.savefig(save_path)
                        plt.close(fig)
                elif fig_type == 'points':
                        fig = plt.figure()
                        plt.plot(x, y ,marker)
                        plt.axis([0, max(x), 0, max(y)])
                        plt.savefig(save_path)
                        plt.close(fig)
                

                        
                        
                        


# if __name__ == '__main__':
#         class_num = 10
#         x = []
#         for k in range(class_num):
#             x.append((k+1)*10)
#         y = [889, 0, 0, 0, 0, 0, 0, 0, 0, 407]
#         x_labels = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5',
#                     'class_6', 'class_7', 'class_8', 'class_9', 'class_10']
#         Plot().plot_bar(x, y,x_labels, width=5)

#         c = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5',
#              'class_6', 'class_7', 'class_8', 'class_9', 'class_10']
#         x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#         y = [889, 0, 0, 0, 0, 0, 0, 0, 0, 407]
#         fig, ax = plt.subplots()
#         ax.set_xticks(x)
#         ax.set_xticklabels(c)
#         plt.bar(x, y,width = 1, align ='center')
#         plt.xlabel('x')
#         plt.ylabel('y')                
#         plt.show()



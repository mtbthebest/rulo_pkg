#!/usr/bin/env	python
import rospy 
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib import colors
from collections import OrderedDict
from rulo_base.markers import VizualMark, TextMarker
import pandas as pd
from pandas import DataFrame, Series
from Memory import Memory 
from rulo_utils.csvwriter import csvwriter

# data = np.random.rand(10, 10) * 20
# print data
# #create discrete colormap
# cmap = colors.ListedColormap(['red','green','blue'])
# bounds = [0,0,0]
# norm = colors.BoundaryNorm(bounds, cmap.N)

# fig, ax = plt.subplots()
# ax.imshow(data, cmap=cmap, norm=norm)

# # draw gridlines
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
# ax.set_xticks(np.arange(0.0, 8.0, 1));
# ax.set_yticks(np.arange(0.0, 10.0, 1));

# plt.show()

# import pylab
# from pylab import *
# cdict = {'red': ((0.0, 0.0, 0.0),
#                  (0.5, 1.0, 0.7),
#                  (1.0, 1.0, 1.0))}#,
#         #  'green': ((0.0, 0.0, 0.0),
#         #            (0.5, 1.0, 0.0),
#         #            (1.0, 1.0, 1.0)),
#         #  'blue': ((0.0, 0.0, 0.0),
#         #           (0.5, 1.0, 0.0),
#         #           (1.0, 0.5, 1.0))}
# my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
# print rand(10,10)
# pcolor(rand(10,10),cmap=my_cmap)
# colorbar()
# pylab.show() 

# import numpy as np
# import matplotlib.pyplot as plt


# # Have colormaps separated into categories:
# # http://matplotlib.org/examples/color/colormaps_reference.html
# cmaps = [('Perceptually Uniform Sequential', [
#             'viridis', 'plasma', 'inferno', 'magma']),
#          ('Sequential', [
#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#          ('Sequential (2)', [
#             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#             'hot', 'afmhot', 'gist_heat', 'copper']),
#          ('Diverging', [
#             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#          ('Qualitative', [
#             'Pastel1', 'Pastel2', 'Paired', 'Accent',
#             'Dark2', 'Set1', 'Set2', 'Set3',
#             'tab10', 'tab20', 'tab20b', 'tab20c']),
#          ('Miscellaneous', [
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))


# def plot_color_gradients(cmap_category, cmap_list, nrows):
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()


# for cmap_category, cmap_list in cmaps:
#     plot_color_gradients(cmap_category, cmap_list, nrows)

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Make a 9x9 grid...
# nrows, ncols = 9,9
# image = np.zeros(nrows*ncols)

# # Set every other cell to a random number (this would be your data)
# image[::2] = np.random.random(nrows*ncols //2 + 1)

# # Reshape things into a 9x9 grid.
# image = image.reshape((nrows, ncols))

# row_labels = range(nrows)
# col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
# plt.matshow(image)
# plt.xticks(range(ncols), col_labels)
# plt.yticks(range(nrows), row_labels)
# plt.show()




# points = [
#     (0, 10),
#     (10, 20),
#     (20, 40),
#     (60, 100),
# ]

# x = list(map(lambda x: x[0], points))
# y = list(map(lambda x: x[1], points))

# plt.rc('grid', linestyle="-", color='black')
# plt.scatter(x, y)
# plt.grid(True,markerfacecolor='red')

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# # Major ticks every 20, minor ticks every 5
# major_ticks = np.arange(-2.25, 4.75, 0.5)
# minor_ticks = np.arange(-4.0, 8.25, 0.5)

# ax.set_xticks(major_ticks)
# # ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(minor_ticks)
# # ax.set_yticks(minor_ticks, minor=True)

# # And a corresponding grid
# ax.grid(which='both')

# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
# ax.fill_between(-2.25, -4.0, -3.75, facecolor='green', interpolate=True)
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# pose = np.load('/home/mtb/array.npy')
# path = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/'
# dirt = np.load(path + 'dirt.npy')
# indices = np.load(path + 'indices.npy')
# dirt = dirt / 10000.0
# all_dirt = []

# for i in range(1296):
#     if i in indices:
#         color = 255.0 - dirt[0] * 245.0
#         if color <0.0:
#             color = 0.0
#         else:
#             res = color /255.0

#         all_dirt.append(res)
#         dirt = np.delete(dirt, [0], axis=0)
#     else:
#         all_dirt.append(1.0)



# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, aspect='equal')
# for p in [patches.Rectangle((pose[i][0], pose[i][1]), 0.25, 0.25,facecolor=(1.0, all_dirt[i],  all_dirt[i], 1.0)) for i in range(1296)]:
#     ax3.add_patch(p)
# # d = patches.Rectangle((3.0, 7.5), 0.25, 0.25, facecolor=(0.0,0.0,0.0,1.0)) 
# # ax3.add_patch(d)
# plt.axis([-2.25, 4.5, -4.0, 8.0])
# plt.axis('off')
# # plt.show()
# fig3.savefig('/home/mtb/dirt_map.png', dpi=90, bbox_inches='tight')


from scipy import misc

face = misc.face()
face = misc.imread('/home/mtb/dirt_map.png')
print face[150,150]
# plt.imshow(face)
# plt.show()



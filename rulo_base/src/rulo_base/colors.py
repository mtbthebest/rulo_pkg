#!/usr/bin/env python 
import rospy
from std_msgs.msg import  ColorRGBA

color_name =['Red', 'Green', 'Blue', 'Yellow', 'Aqua', 'Gray', 'Pink', 'White', 'Black','Orange' , 'Maroon','Navy' ,'Default']
color_rgba =[(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(0,1,1,1),(0.5,0.5,0.5,1),(1,0,1,1), (1,1,1,1), (0,0,0,1),(1,0.55,0,1),(0.5,0,0,1), (0,0,0.5,1),(1,0,0,1)]
color_dict = dict( zip(color_name, color_rgba))

def get_color(color = 'Default', convert= False, texture=None):
    
    if not convert:
        return ColorRGBA(*color_dict[color])
    else:    
        if texture == None:
            return ColorRGBA(*(float(color) / 255.0, float(color) / 255.0, float(color) / 255.0, 1.0))
        if texture == 'Red':
            return ColorRGBA(*(1.0, float(color) / 255.0, float(color) / 255.0, 1.0))
            


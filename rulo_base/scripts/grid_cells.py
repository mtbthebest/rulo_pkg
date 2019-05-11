#!/usr/bin/env python 
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import GridCells
from std_msgs.msg import Header
from geometry_msgs.msg import Point

class GridCell:
    def __init__(self):
        rospy.init_node('grid')
        self.rate = 10
        self.r = rospy.Rate(self.rate)
        self.grid_cell_pub = rospy.Publisher("/grid_cell", GridCells, queue_size=10)
        self.Grid = GridCells()
        self.Grid.header.frame_id = '/map'
    
    def creater(self, height =0.5, width=0.5, center=[]):
        self.Grid.cell_height =  height
        self.Grid.cell_width = width


        for centers in center:
            self.Grid.cells.append(Point(*centers))
        while not rospy.is_shutdown():                
                self.grid_cell_pub.publish(self.Grid)
                self.r.sleep()

if __name__ == '__main__':
    GridCell() .creater(height=0.25, width=0.25, center=[[0,0,0],[2,2,0]])
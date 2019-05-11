#!/usr/bin/env python
import os
import json

class JSON:
    def __init__(self,filename):
        self.filename = filename
        
        
    def creater(self):
        if not os.path.isfile(self.filename):
            file =  open(self.filename, 'wb')
            print 'Created'
            return True
        else:
            print 'Already created'
            return False
    def writer(self,data):        
        with open(self.filename, 'ab') as file:
             json.dump(data,file)
    def loader(self):
            data = []
            with open(self.filename, 'rb') as file :
                for line in file:
                    data.append(json.loads(line))
         
           

if __name__ == '__main__':
    data = {'a':1, 'b':2}
    JSON('./portloppofjjf') .writer(data)
    
    car = {'c':3, 'd':4}
    JSON('./portloppofjjf') .writer(car)    
    print JSON('./portloppofjjf').loader()
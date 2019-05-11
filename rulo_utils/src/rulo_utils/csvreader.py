#!/usr/bin/env python
import csv
from collections import OrderedDict
from rulo_utils.csvwriter import csvwriter

def csvread(filename):
    # print filename
    data = OrderedDict()
    with  open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)
        for header in headers:
            data[header] = list()
        j = 0
        for row in csvreader:
            # j +=1
            # print j
            # # if j <5:
            # if not j == 282954:
            #   csvwriter('/home/mtb/Documents/data/test2017-11-29_10h20_4.csv',
            #               headers, [[row[i]] for i in range(len(row)) ])
                # print [[row[i]] for i in range(len(row))][10]
                # print headers
                
            for i in range(len(row)):               
                try:
                   data[data.keys()[i]].append(row[i])
                except:
                    pass
    csvfile.close()
              
                    
        
    return data



# if __name__ == '__main__':
#     main()
    

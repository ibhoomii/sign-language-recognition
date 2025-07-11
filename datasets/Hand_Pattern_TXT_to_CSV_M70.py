#After Stage 2, move all data text files into one folder. Then Follow below steps
#1) Copy paste this script into the same folder
#2) For the first time run this script directly, this will read the dataA.txt file from the earlier stage and then convert it into the CSV file named "datasets_A.csv"
#3) Change edit one & edit two lines in this code for the respective letter and re-run.
#4) Perform these steps for all the letters


import pandas as pd

data = pd.read_csv("dataZ.txt",delimiter = ',')  # edit one
data.columns = ['x0','y0','x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6','x7','y7',
                'x8','y8','x9','y9','x10','y10','x11','y11','x12','y12','x13','y13','x14','y14',
                'x15','y15','x16','y16','x17','y17','x18','y18','x19','y19','x20','y20',
                '0_1','0_2','0_3','0_4','0_5','0_6','0_7','0_8','0_9','0_10','0_11','0_12','0_13','0_14','0_15','0_16','0_17','0_18','0_19','0_20','Lable','Location']

data.to_csv('datasets_Z.csv', index = None) # edit two


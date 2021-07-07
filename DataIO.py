import numpy as np
from numpy.lib.shape_base import split

def DataIO():
    originFile = open("para_cameras_1004_same.txt")
    RMatirxFile = open("rotation_matrix_file_12", 'w')
    worldPositionFile = open("world_position_file_12", 'w')
    originDataLines = originFile.readlines()
    for id in range(12):
        line1 = originDataLines[id*5+3]
        line2 = originDataLines[id*5+4]
        splitLine1 = line1.split( ) #R
        for index in range(8):
            RMatirxFile.write("%.15f, "%float(splitLine1[index+1]))
        RMatirxFile.write("%.15f\n"%float(splitLine1[8+1]))
        splitLine2 = line2.split( )
        for index in range(2):
            worldPositionFile.write("%.15f\t"%float(splitLine2[index+1]))
        worldPositionFile.write("%.15f\n"%float(splitLine2[2+1]))

if __name__ == '__main__':
    DataIO()

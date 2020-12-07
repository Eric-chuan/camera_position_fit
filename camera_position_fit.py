import numpy as np
from numpy import mean
import math
import random
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit

from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

from fitEllipse import fit_ellipse

graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

def rotationMatrixToEulerAngles(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    #assert(n < 1e-6)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def objective1(x, a, b):
    return (a * x + b)

def objective2(x, a, b, c):
	return (a * x + b * x**2 + c)

def objective3(x, a, b, c, d):
	return (a * x + b * x**2 + c * x**3 + d)

def objective4(x, a, b, c, d, e):
	return (a * x + b * x**2 + c * x**3 + d * x**4 + e)

def objective5(x, a, b, c, d, e, f):
	return (a * x + b * x**2 + c * x**3 + d * x**4 + e * x**5 + f)

def objective6(x, a, b, c, d, e, f, g):
	return (a * x + b * x**2 + c * x**3 + d * x**4 + e * x**5 + f * x**6 + g)



def yz_plot(data, objective):
    y = data[:, 0, 1]
    z = data[:, 0, 2]
    popt, _ = curve_fit(objective, y, z)
    return popt

def yx_plot(data, objective):
    y = data[:, 0, 1]
    x = data[:, 0, 0]
    popt, _ = curve_fit(objective, y, x)
    return popt

def ellipse_fit(data):
    X = data[:, 0, 0]
    Y = data[:, 0, 1]
    Z = np.array(list(zip(X, Y)))
    reg = LsqEllipse().fit(Z)
    return reg

def ellipse_fit2(data):
    X = data[:, 0, 0]
    Y = data[:, 0, 1]
    reg = fit_ellipse(X, Y)
    return reg


def y_eulerx_fit(data1, data2):
    y = data1[:, 0, 1]
    euler_x = data2[:, 0, 0]
    popt, _ = curve_fit(objective6, y, euler_x)
    return popt

def y_eulery_fit(data1, data2):
    y = data1[:, 0, 1]
    euler_y = data2[:, 0, 1]
    popt, _ = curve_fit(objective6, y, euler_y)
    return popt

def y_eulerz_fit(data1, data2):
    y = data1[:, 0, 1]
    euler_z = data2[:, 0, 2]
    popt, _ = curve_fit(objective6, y, euler_z)
    return popt

if __name__ == '__main__':
    #read rotation matrix data
    rotation_data = np.ones((30,3,3))
    rotation_file = open("rotation_matrix_file")
    lines = rotation_file.readlines()
    index = 0
    for line in lines:
        line = line.strip()
        formLine = line.split(',')
        for i in range(0,3):
            for j in range(0,3):
                rotation_data[index, i, j] = formLine[3 * i + j]
        index += 1

    #read world_position data
    result = np.ones((30,1,3))
    world_position_data = np.ones((30,1,3))
    world_position_file = open("world_position_file")
    world_position_file_lines = world_position_file.readlines()
    index = 0
    for line in world_position_file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split('\t')
        world_position_data[index,:] = formLine[0:3]
        tmp = np.dot(np.linalg.inv(rotation_data[index]), (world_position_data[index].T))
        result[index,:] = tmp.T
        index += 1

    #get euler for each rotation matrix
    euler = np.ones((30,1,3))
    for i in range(0,30):
        euler1 = rotationMatrixToEulerAngles(rotation_data[i])
        euler[i,:] = euler1


    y = np.arange(-45, 25, 0.1)
    point_x = world_position_data[:,0,0]
    point_y = world_position_data[:,0,1]
    point_z = world_position_data[:,0,2]

#################   Y-X poly fit   #################
    popt_x = yx_plot(world_position_data, objective6)
    a_x, b_x, c_x, d_x, e_x, f_x, g_x = popt_x
    fit_x = a_x * y + b_x * y**2 + c_x * y**3 + d_x * y**4 + e_x * y**5 + f_x * y**6 + g_x

#################   Y-X ellipses fit   #################
    # reg2 = ellipse_fit2(world_position_data)
    # a, b, center0, center1, phi2 = reg2

    # A1 = a * np.cos(phi2)
    # B1 = b * np.sin(phi2)
    # C1 = a * np.sin(phi2)
    # D1 = b * np.cos(phi2)
    # theta1 = np.arccos(C1/np.sqrt(C1 ** 2 + D1 ** 2))

    # fit_point_x = np.zeros(30)
    # for i in range(0, 30):
    #     a = point_y[i] - center1
    #     t = np.arccos(a / np.sqrt(C1 ** 2 + D1 **2)) - theta1
    #     fit_point_x[i] = center0 - (A1 * np.cos(t) - B1 * np.sin(t))

    # t = np.arccos((y - center1) / np.sqrt(C1 ** 2 + D1 **2)) - theta1
    # fit_x = center0 - (A1 * np.cos(t) - B1 * np.sin(t))

#################   Y-Z poly fit   #################
    popt_z = yz_plot(world_position_data, objective2)
    a_z, b_z, c_z = popt_z
    fit_z = a_z * y + b_z * y**2 + c_z

#################   Y-Eulerx poly fit   #################
    popt_eulerx = y_eulerx_fit(world_position_data, euler)
    a_ex, b_ex, c_ex, d_ex, e_ex, f_ex, g_ex= popt_eulerx
    fit_eulerx = a_ex * y + b_ex * y**2 + c_ex * y**3 + d_ex * y**4 + e_ex * y**5 + f_ex * y**6 + g_ex

#################   Y-Eulery poly fit   #################
    popt_eulery = y_eulery_fit(world_position_data, euler)
    a_ey, b_ey, c_ey, d_ey, e_ey, f_ey, g_ey= popt_eulery
    fit_eulery = a_ey * y + b_ey * y**2 + c_ey * y**3 + d_ey * y**4 + e_ey * y**5 + f_ey * y**6 + g_ey

#################   Y-Eulerz poly fit   #################
    popt_eulerz = y_eulerz_fit(world_position_data, euler)
    a_ez, b_ez, c_ez, d_ez, e_ez, f_ez, g_ez= popt_eulerz
    fit_eulerz = a_ez * y + b_ez * y**2 + c_ez * y**3 + d_ez * y**4 + e_ez * y**5 + f_ez * y**6 + g_ez

##################  test point ##########################
    test_y = random.uniform(-46, 23)

    test_x = a_x * test_y + b_x * test_y**2 + c_x * test_y**3 + d_x * test_y**4 + e_x * test_y**5 + f_x * test_y**6 + g_x
    test_z = a_z * test_y + b_z * test_y**2 + c_z

    test_eulerx = a_ex *test_y + b_ex * test_y**2 + c_ex * test_y**3 + d_ex * test_y**4 + e_ex * test_y**5 + f_ex * test_y**6 + g_ex
    test_eulery = a_ey * test_y + b_ey * test_y**2 + c_ey * test_y**3 + d_ey * test_y**4 + e_ey * test_y**5 + f_ey * test_y**6 + g_ey
    test_eulerz = a_ez * test_y + b_ez * test_y**2 + c_ez * test_y**3 + d_ez * test_y**4 + e_ez * test_y**5 + f_ez * test_y**6 + g_ez

    print(test_x, test_y, test_z, test_eulerx, test_eulery, test_eulerz)
    test_world_position = np.array([test_x, test_y, test_z])
    test_euler = np.array([test_eulerx, test_eulery, test_eulerz])
    test_rotation_matrix = eulerAnglesToRotationMatrix(test_euler)
    print(test_rotation_matrix)

    with open('rotation_matrix_result', 'w') as f: # 默认模式为‘r’，只读模式
        for i in range(0,3):
            for j in range(0,3):
                f.write("%5s%i\t%s\t%.15f\n"%("Vcam_krt_R_", i*3 + j, '=', test_rotation_matrix[i][j]))
        f.write('\n')
        for i in range(0,3):
            f.write("%15s%i\t%s\t%.15f\n"%("Vcam_krt_WorldPosition_", i, '=',  test_world_position[i]))
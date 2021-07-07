import numpy as np
from numpy import mean
import math
import random
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit

from fitEllipse import fit_ellipse

graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

def draw3D(X, Y, Z, f):
    plt.grid(True)
    plt.title("camera 3D position")
    axes = Axes3D(f)
    axes.scatter(X, Y, Z, c='g')
    # axes.scatter(fit_point_x, point_y, fit_point_z, c='purple')
    axes.set_title('Scatter Plot (click-drag with mouse)')
    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')
    axes.set_zlabel('Z Data')
    # plt.show()

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

def xy_plot(data, objective):
    x = data[:, 0, 0]
    y = data[:, 0, 1]
    popt, _ = curve_fit(objective, x, y)
    return popt

def xz_plot(data, objective):
    x = data[:, 0, 0]
    z = data[:, 0, 2]
    popt, _ = curve_fit(objective, x, z)
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


def x_eulerx_fit(data1, data2):
    x = data1[:, 0, 0]
    euler_x = data2[:, 0, 0]
    popt, _ = curve_fit(objective6, x, euler_x)
    #eulerx = a1 * y + b1 * y**2 + c1 * y**3 + d1
    return popt

def x_eulery_fit(data1, data2):
    x = data1[:, 0, 0]
    euler_y = data2[:, 0, 1]
    popt, _ = curve_fit(objective6, x, euler_y)
    #eulerya1 * y + b1 * y**2 + c1
    return popt

def x_eulerz_fit(data1, data2):
    x = data1[:, 0, 0]
    euler_z = data2[:, 0, 2]
    popt, _ = curve_fit(objective6, x, euler_z)
    #eulerx = a1 * y + b1 * y**2 + c1 * y**3 + d1
    return popt
if __name__ == '__main__':
    #read rotation matrix data
    rotation_data = np.ones((12,3,3))
    rotation_file = open("rotation_matrix_file_12")
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
    result = np.ones((12,1,3))
    world_position_data = np.ones((12,1,3))
    world_position_file = open("world_position_file_12")
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
    euler = np.ones((12,1,3))
    for i in range(0,12):
        euler1 = rotationMatrixToEulerAngles(rotation_data[i])
        euler[i,:] = euler1


    x = np.arange(-6, 8, 0.1)
    point_x = world_position_data[:,0,0]
    point_y = world_position_data[:,0,1]
    point_z = world_position_data[:,0,2]
    fitParaFile = open("fit_para.txt", 'w')
    f1 = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=100)
    draw3D(point_x, point_y, point_z, f1)
#################   X-Y poly fit   #################
    popt_y = xy_plot(world_position_data, objective2)
    a_y, b_y, c_y = popt_y
    fit_y = a_y * x + b_y * x**2 + c_y
    fitParaFile.write("y=%.15f*x + %.15f*x^2 + %.15f\n"%(a_y, b_y, c_y))

    fig = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    plt.title('X-Y fit')
    plt.scatter(point_x, world_position_data[:,0,1], s=16., c='b') #三次
    plt.plot(x, fit_y, c='g') #三次
    # plt.show()

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

#################   X-Z poly fit   #################
    popt_z = xz_plot(world_position_data, objective2)
    a_z, b_z, c_z = popt_z
    #print(a_z, b_z, c_z)
    fit_z = a_z * x + b_z * x**2 + c_z
    fitParaFile.write("z=%.15f*x + %.15f*x^2 + %.15f\n"%(a_z, b_z, c_z))
    fig = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    plt.title('X-Z fit')
    plt.scatter(point_x, world_position_data[:,0,2], s=16., c='b') #三次
    plt.plot(x, fit_z, c='g') #三次
    # plt.show()

#################   X-Eulerx poly fit   #################
    popt_eulerx = x_eulerx_fit(world_position_data, euler)
    a_ex, b_ex, c_ex, d_ex, e_ex, f_ex, g_ex= popt_eulerx
    #print(a_ex, b_ex, c_ex, d_ex, e_ex, f_ex, g_ex)
    fit_point_eulerx = a_ex * point_x + b_ex * point_x**2 + c_ex * point_x**3 + d_ex * point_x**4 + e_ex * point_x**5 + f_ex * point_x**6 + g_ex
    fit_eulerx = a_ex * x + b_ex * x**2 + c_ex *x**3 + d_ex * x**4 + e_ex * x**5 + f_ex * x**6 + g_ex
    fitParaFile.write("Eulerx=%.15f*x + %.15f*x^2 + %.15f*x^3 + %.15f*x^4 + %.15f*x^5 + %.15f*x^6 + %.15f\n"%(a_ex, b_ex, c_ex, d_ex, e_ex, f_ex, g_ex))
    fig = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    plt.title('X-EulerX fit')
    plt.scatter(point_x, fit_point_eulerx, s=16., c='b') #三次
    plt.plot(x, fit_eulerx, c='g') #三次
    # plt.show()
#################   X-Eulery poly fit   #################
    popt_eulery = x_eulery_fit(world_position_data, euler)
    a_ey, b_ey, c_ey, d_ey, e_ey, f_ey, g_ey= popt_eulery
    fit_point_eulery = a_ey * point_x + b_ey * point_x**2 + c_ey * point_x**3 + d_ey * point_x**4 + e_ey * point_x**5 + f_ey * point_x**6 + g_ey
    fit_eulery = a_ey * x + b_ey * x**2 + c_ey * x**3 + d_ey * x**4 + e_ey * x**5 + f_ey * x**6 + g_ey
    fitParaFile.write("Eulery=%.15f*x + %.15f*x^2 + %.15f*x^3 + %.15f*x^4 + %.15f*x^5 + %.15f*x^6 + %.15f\n"%(a_ey, b_ey, c_ey, d_ey, e_ey, f_ey, g_ey))
    fig = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    plt.title('X-EulerY fit')
    plt.scatter(point_x, fit_point_eulery, s=16., c='b') #三次
    plt.plot(x, fit_eulery, c='g') #三次
    # plt.show()
#################   X-Eulerz poly fit   #################
    popt_eulerz = x_eulerz_fit(world_position_data, euler)
    a_ez, b_ez, c_ez, d_ez, e_ez, f_ez, g_ez= popt_eulerz
    fit_point_eulerz = a_ez * point_x + b_ez * point_x**2 + c_ez * point_x**3 + d_ez * point_x**4 + e_ez * point_x**5 + f_ez * point_x**6 + g_ez
    fit_eulerz = a_ez * x + b_ez * x**2 + c_ez * x**3 + d_ez * x**4 + e_ez * x**5 + f_ez * x**6 + g_ez
    fitParaFile.write("Eulerz=%.15f*x + %.15f*x^2 + %.15f*x^3 + %.15f*x^4 + %.15f*x^5 + %.15f*x^6 + %.15f\n"%( a_ez, b_ez, c_ez, d_ez, e_ez, f_ez, g_ez))
    # print(a_ez, b_ez, c_ez, d_ez, e_ez, f_ez, g_ez)
    fig = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    plt.title('X-EulerZ fit')
    plt.scatter(point_x, fit_point_eulerz, s=16., c='b') #三次
    plt.plot(x, fit_eulerz, c='g') #三次
    plt.show()

##################  test point ##########################
    test_x = random.uniform(-46, 23)

    test_y = a_y * test_x + b_y * test_x**2 + c_y
    test_z = a_z * test_x + b_z * test_x**2 + c_z

    test_eulerx = a_ex *test_x + b_ex * test_x**2 + c_ex * test_x**3 + d_ex * test_x**4 + e_ex * test_x**5 + f_ex * test_x**6 + g_ex
    test_eulery = a_ey * test_x + b_ey * test_x**2 + c_ey * test_x**3 + d_ey * test_x**4 + e_ey * test_x**5 + f_ey * test_x**6 + g_ey
    test_eulerz = a_ez * test_x + b_ez * test_x**2 + c_ez * test_x**3 + d_ez * test_x**4 + e_ez * test_x**5 + f_ez * test_x**6 + g_ez

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
import numpy as np
from ellipse import LsqEllipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

if __name__ == '__main__':
    # avalible in the `example.py` script in this repo
    world_position_data = np.ones((30,1,3))
    world_position_file = open("world_position_file")
    world_position_file_lines = world_position_file.readlines()
    index = 0

    for line in world_position_file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split('\t')
        world_position_data[index,:] = formLine[0:3]
        #ax.scatter(float(formLine[0]), float(formLine[1]), float(formLine[2]), c='g')
        #ax1.scatter(float(formLine[0]), float(formLine[1]), float(formLine[2]), c='g')
        index += 1
    X1 = world_position_data[:, 0, 0]
    X2 = world_position_data[:, 0, 1]
    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()

    A0 = width * np.cos(phi)
    B0 = height * np.sin(phi)
    C0 = width * np.sin(phi)
    D0 = height * np.cos(phi)
    theta = np.arccos(C0/np.sqrt(C0 ** 2 + D0 **2))

    print(A0, B0, C0, D0, theta)

    x_fit = np.zeros(30)
    for i in range(0, 30):
        a = X2[i] - center[1]
        t = np.arccos(a / np.sqrt(C0 ** 2 + D0 **2)) - theta
        x_fit[i] = center[0] - (A0 * np.cos(t) - B0 * np.sin(t))




    print(f'center: {center[0]:.3f}, {center[1]:.3f}')
    print(f'width: {width:.3f}')
    print(f'height: {height:.3f}')
    print(f'phi: {phi:.3f}')

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.axis('equal')
    ax.plot(X1, X2, 'ro', zorder=1)
    ax.scatter(x_fit, X2, c='g')
    ellipse = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    plt.legend()
    plt.show()




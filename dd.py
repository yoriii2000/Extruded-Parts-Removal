import numpy as np
import math
import scipy.linalg
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import numpy

import transformat as trans
import open3d as o3d


# data = np.array([[44, 18, 18],
#               [43, 18, 16],
#               [42, 19, 19]])
#
# # regular grid covering the domain of the data
# mn = np.min(data, axis=0)  # Set minimum in X, Y, Z
# mx = np.max(data, axis=0)
# A = np.c_[data[:, [0]], data[:, [1]]]
# B_1 = data[:, 0]
# B_2 = data[:, 1]
# B = data[:, :2]
#
# a = [[4,8],[12,2]]
# b = [2,2]
# c = scipy.linalg.lstsq(a, b)
#
# n = data[:, 0:2]

def ReadXyzFile(filename):
    print('File Path:', filename)
    f = open(filename, "r")
    lines = f.readlines()
    print('No of Points [XYZ]:', len(lines))
    PointList = []

    for x in range(0, len(lines)):
        RawData = lines[x].strip().split() #[x y z] from File
        PointList.append([float(RawData[0]), float(RawData[1]), float(RawData[2])])

    return PointList

def SaveFile(Pcd_File_Name, PCDList):
    np.savetxt('{}'.format(Pcd_File_Name), PCDList, delimiter=' ')
    print('Saved File: [{}].'.format(Pcd_File_Name))

NameOfSourcePointFile = 'Model.xyz'

allpoint= ReadXyzFile(NameOfSourcePointFile)
basexy = []
nxyz = []

for allpoint_no in range(0, len(allpoint)):
    # if (EdgePoint[EdgePoint_no][2]>1.5 and EdgePoint[EdgePoint_no][2]<39):
    if (allpoint[allpoint_no][2] == 0):
        basexy.append(allpoint[allpoint_no])

data = np.asarray(basexy)
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)

nx = (mx[0] - mn[0]) / 2
ny = (mx[1] - mn[1]) / 2
nxyz.append([nx, ny, 0])

# print(basexy)
print('nxyz = ', nxyz)
SaveFile('nxyz.xyz', nxyz)
Uni = 5
Ux = [nx + Uni, ny, 0]
Uy = [nx, ny + Uni, 0]
Uz = [nx, ny, 0 + Uni]
basexyz = np.vstack([Ux, Uy, Uz, nxyz])
basexyz = basexyz.tolist()

for PcdNo in range(0, len(basexyz)):
    basexyz[PcdNo].append(255)
    basexyz[PcdNo].append(255)
    basexyz[PcdNo].append(255)
print('basexyz', basexyz)
SaveFile('basexyz.xyz', basexyz)

oripoint = [0, 0, 0]
Uox = [0 + Uni, 0, 0]
Uoy = [0, 0 + Uni, 0]
Uoz = [0, 0, 0 + Uni]
Oxyz = np.vstack([Uox, Uoy, Uoz, oripoint])
Oxyz = Oxyz.tolist()

for PcdNo in range(0, len(basexyz)):
    Oxyz[PcdNo].append(255)
    Oxyz[PcdNo].append(255)
    Oxyz[PcdNo].append(255)
print('Oxyz', Oxyz)
SaveFile('Oxyz.xyz', Oxyz)

target = o3d.io.read_point_cloud("Oxyz.xyz")
source = o3d.io.read_point_cloud("basexyz.xyz")
text = 'tran.xyz'
trans.demo_manual_registration(source, target, text)

pcd_1 = o3d.io.read_point_cloud('Model.xyz')
toworld1_2 = np.genfromtxt('tran.xyz', dtype=None, comments='#', delimiter=' ')
pcd_1 = pcd_1.transform(toworld1_2)
o3d.io.write_point_cloud("Model.xyz", pcd_1)

x = []
y = []
z = []
X = []
Y = []
total_rad = 10
z_factor = 3
noise = 0.1
num_true_pts = 200
s_true = np.linspace(0, total_rad, num_true_pts)
x_true = np.cos(s_true)
y_true = np.sin(s_true)
z_true = s_true / z_factor
num_sample_pts = 80
s_sample = np.linspace(0, total_rad, num_sample_pts)
# x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
# y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
# z_sample = s_sample / z_factor + noise * np.random.randn(num_sample_pts)

for PointNo in range(0, len(traj)):
    x.append(traj[PointNo][0])
    y.append(traj[PointNo][1])
    z.append(traj[PointNo][2])

x_sample = x
y_sample = y
z_sample = z

tck, u = interpolate.splprep([x_sample, y_sample, z_sample], s=10000)

x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)

u_fine = np.linspace(0, 1, num_true_pts)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig2 = plt.figure()
ax3d = fig2.add_subplot(111, projection='3d')
ax3d.plot(x_true, y_true, z_true, 'b')  # blue line
ax3d.plot(x_sample, y_sample, z_sample, 'r*')  # red point
ax3d.plot(x_knots, y_knots, z_knots, 'go')  # green dot
ax3d.plot(x_fine, y_fine, z_fine, 'g')  # green line
ax3d.set_xlim3d(xmin=35, xmax=45)
ax3d.set_ylim3d(ymin=10, ymax=25)
ax3d.set_zlim3d(zmin=0, zmax=50)

# 3D example
# fig2.show()
# plt.show()
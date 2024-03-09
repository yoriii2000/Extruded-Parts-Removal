import math
import scipy.linalg
import numpy as np
import open3d as o3d

import transformat as trans
import Estnor as est
from numpy.linalg import inv


def ReadXyzFile(filename):
    print('File Path:', filename)
    f = open(filename, "r")
    lines = f.readlines()
    print('No of Points [XYZ]:', len(lines))
    PointList = []

    for x in range(0, len(lines)):
        RawData = lines[x].strip().split() #[x y z] from File
        # print('r = ', len(RawData))
        if len(RawData) > 3:
            PointList.append([float(RawData[0]), float(RawData[1]), float(RawData[2]), float(RawData[3]), float(RawData[4]),float(RawData[5])])
        else:
            PointList.append([float(RawData[0]), float(RawData[1]), float(RawData[2])])

    return PointList

def SaveFile(Pcd_File_Name, PCDList):

    np.savetxt('{}'.format(Pcd_File_Name), PCDList, delimiter=' ')
    print('Saved File: [{}].'.format(Pcd_File_Name))

def Sq2(value):

    return value*value

def Nuvw(PointBase, Point):

    # SaveFile('PointBase.xyz', PointBase)

    for i in range(1, len(Point)):
        # print(Point[i])
        # Distance
        d = math.sqrt(Sq2(Point[i][0] - PointBase[0]) + Sq2(Point[i][1] - PointBase[1]) + Sq2(Point[i][2] - PointBase[2]))

        # Get the vector reference which has smallest distance from base point
        if i == 1:  # initial
            d_min = d
            Point_ref = [Point[i][0], Point[i][1], Point[i][2]]
        else:
            if d < d_min:
                d_min = d
                Point_ref = [Point[i][0], Point[i][1], Point[i][2]]

    # print('\nPoint_ref and Point_base')
    # print(Point_ref)
    # print(PointBase)

    # Get the vector base and point reference (A)
    A = np.subtract(Point_ref, PointBase)

    ChosenPoint = []
    ChosenPointDot = []
    for PointNo in range(1, len(Point)):
        # Get the vector base and another point
        B = np.subtract(Point[PointNo], PointBase)
        # print('\nDot product to point:', Point[PointNo])
        AB_dot = np.dot(A, B)
        # print('dotprod:', AB_dot)

        # Sort the dot prod from smallest to largest:

        DotProdIsSmaller_sign = False
        # Initial value for chosen point
        if PointNo == 0:
            ChosenPointDot.append(AB_dot)  # Store the dotprod score
            ChosenPoint.append(Point[PointNo])  # Store the point

        else:
            # Compare the current dotprod with the latest dotprod
            for ChosenPointDot_no in range(0, len(ChosenPointDot)):
                if AB_dot < ChosenPointDot[ChosenPointDot_no]:
                    ChosenPointDot.insert(ChosenPointDot_no, AB_dot)
                    ChosenPoint.insert(ChosenPointDot_no, Point[PointNo])
                    DotProdIsSmaller_sign = True
                    break

            if DotProdIsSmaller_sign == False:
                ChosenPointDot.append(AB_dot)
                ChosenPoint.append(Point[PointNo])

        # print('\n Chosen point dot')
        # print(ChosenPointDot)

    # Just take the 3 smallest dot prod value
    ChosenPoint = ChosenPoint[:3]
    # print('\nChosen points:')
    # print(ChosenPoint)

    # Filter the chosen point which has smallest distance (max 2 points)
    FinalChosenPointD_list = []
    FinalChosenPoint = []

    for i in range(0, len(ChosenPoint)):
        # Distance
        d = math.sqrt(Sq2(ChosenPoint[i][0] - PointBase[0]) + Sq2(ChosenPoint[i][1] - PointBase[1]) + Sq2(ChosenPoint[i][2] - PointBase[2]))

        DistIsSmaller_Sign = False

        if i == 0:  # initial
            FinalChosenPointD_list.append(d)
            FinalChosenPoint.append([ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])

        else:
            for FinalChosenPoint_no in range(0, len(FinalChosenPoint)):
                if d < FinalChosenPointD_list[FinalChosenPoint_no]:
                    FinalChosenPointD_list.insert(FinalChosenPoint_no, d)
                    FinalChosenPoint.insert(FinalChosenPoint_no,[ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])
                    DistIsSmaller_Sign = True
                    break

            if DistIsSmaller_Sign == False:
                FinalChosenPointD_list.append(d)
                FinalChosenPoint.append([ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])

    FinalChosenPoint[2] = Point_ref

    # print('\n Final Chosen point')
    # print('xyz points = ', FinalChosenPoint)
    # SaveFile('FinalChosenPoint.xyz', FinalChosenPoint)

    # print('Base Point')
    # print('原點 = ', PointBase)

    return FinalChosenPoint

# NameOfEdgeFilenor = 'Result/SF_SampledPoints_Cluster_0nor.xyz'
NameOfEdgeFile = 'Result/SF_SampledPoints_Cluster_0.xyz'
print('fafa = ',type(NameOfEdgeFile))
OBBsffile = 'Process/OBBpoint_SF0.xyz'
# filename = 'Result/SF_SampledPoints_Cluster_0.xyz'

SamplePoint1 = ReadXyzFile('{}'.format(NameOfEdgeFile))
SamplePoint = est.Estnor(file=NameOfEdgeFile)

# SamplePoint = ReadXyzFile('{}'.format(NameOfEdgeFilenor))
OBBsfPoint = ReadXyzFile('{}'.format(OBBsffile))

X = []
Y = []
Z = []
Xnor = []
Ynor = []
Znor = []

XSum = 0
YSum = 0
ZSum = 0
Xnorsum = 0
Ynorsum = 0
Znorsum = 0

XMean = []
YMean = []
ZMean = []
XnorMean = []
YnorMean = []
ZnorMean = []


for PointNo in range(0, len(SamplePoint)):
    X.append(SamplePoint[PointNo][0])
    Y.append(SamplePoint[PointNo][1])
    Z.append(SamplePoint[PointNo][2])
    Xnor.append(SamplePoint[PointNo][3])
    Ynor.append(SamplePoint[PointNo][4])
    Znor.append(SamplePoint[PointNo][5])

    XSum = XSum + SamplePoint[PointNo][0]
    YSum = YSum + SamplePoint[PointNo][1]
    ZSum = ZSum + SamplePoint[PointNo][2]
    Xnorsum = Xnorsum + SamplePoint[PointNo][3]
    Ynorsum = Ynorsum + SamplePoint[PointNo][4]
    Znorsum = Znorsum + SamplePoint[PointNo][5]

XMean = XSum / len(SamplePoint)
YMean = YSum / len(SamplePoint)
ZMean = ZSum / len(SamplePoint)
XnorMean = Xnorsum / len(SamplePoint)
YnorMean = Ynorsum / len(SamplePoint)
ZnorMean = Znorsum / len(SamplePoint)


print('XnorMean = ', XnorMean)
print(XnorMean)
CP = []
CPnor = []
CP.append([XMean, YMean, ZMean])
CPnor.append([XnorMean, YnorMean, ZnorMean])

# SaveFile('central point.xyz', CP)
# SaveFile('central point nor.xyz', CPnor)

# print('central point = ', CP)
print('central point nor= ', CPnor)


OBBx = []
OBBy = []
OBBz = []

for i in range (0, len(OBBsfPoint)):
    OBBx.append(SamplePoint[PointNo][0])
    OBBy.append(SamplePoint[PointNo][1])
    OBBz.append(SamplePoint[PointNo][2])

uvwp = Nuvw(OBBsfPoint[0], OBBsfPoint)
print('OBBsfPoint = ', OBBsfPoint)

v1 = np.subtract(uvwp[0], OBBsfPoint[0])
# print('uvwp = ', uvwp[0])
# print('OBBsfPoint[0] = ', OBBsfPoint[0])
# print('v1 = ', v1)

# OBB上新的XYZ軸的向量
Nz = CPnor
Nz = np.asarray(CPnor)
print('sdasda = ',type(Nz) )

Ny = np.cross(Nz, v1)
Nx = np.cross(Ny, Nz)
print('Nx = ', Nx)
print('Ny = ', Ny)
print('Nz = ', Nz)

x_p = Nx.flatten()
y_p = Ny.flatten()
z_p = Nz.flatten()
print('x_p = ', x_p)
print('y_p = ', y_p)
print('z_p = ', z_p)

dx = math.sqrt(Sq2(x_p[0] - 0) + Sq2(x_p[1] - 0) + Sq2(x_p[2]) - 0)
dy = math.sqrt(Sq2(y_p[0] - 0) + Sq2(y_p[1] - 0) + Sq2(y_p[2]) - 0)
dz = math.sqrt(Sq2(z_p[0] - 0) + Sq2(z_p[1] - 0) + Sq2(z_p[2]) - 0)
x_p = x_p/dx
y_p = y_p/dy
z_p = z_p/dz

# OBB上新的XYZ軸，用點表示
Nx_axiel = OBBsfPoint[0] + x_p
Ny_axiel = OBBsfPoint[0] + y_p
Nz_axile = OBBsfPoint[0] + z_p
Nxyz_axiel = np.vstack([Nx_axiel, Ny_axiel, Nz_axile, OBBsfPoint[0]])
# SaveFile('NxyzAxiel.xyz', Nxyz_axiel)

# 平移到中心點
CP1 = np.asarray(CP)
CP1 = CP1.flatten()
# CP1 = [round(i,4) for i in CP1]
# print('CP1 = ',CP1)

shift_v = CP1 - OBBsfPoint[0]
CP1 = [round(i,4) for i in CP1]

x_c = Nx_axiel + shift_v
y_c = Ny_axiel + shift_v
z_c = Nz_axile + shift_v
x_c = [round(i,4) for i in x_c]
y_c = [round(i,4) for i in y_c]
z_c = [round(i,4) for i in z_c]

XYZC_axiel = np.vstack([x_c, y_c, z_c, CP1])
XYZC_axiel = XYZC_axiel.tolist()
for PcdNo in range(0, len(XYZC_axiel)):
    XYZC_axiel[PcdNo].append(255)
    XYZC_axiel[PcdNo].append(255)
    XYZC_axiel[PcdNo].append(255)
print('XYZC_axiel = ', XYZC_axiel)
SaveFile('XYZC_axiel.xyz', XYZC_axiel)

# # 原始坐標軸
# oripoint = [0, 0, 0]
# v1 = np.subtract(uvwp[0], OBBsfPoint[0])
# x_o = [0 + dx, 0, 0]
# y_o = [0, 0 + dy, 0]
# z_o = [0, 0, 0 + dz]
# x_o = [round(i,4) for i in x_o]
# y_o = [round(i,4) for i in y_o]
# z_o = [round(i,4) for i in z_o]
#
# XYZO_axiel = np.vstack([x_o, y_o, z_o, oripoint])
# XYZO_axiel = XYZO_axiel.tolist()
# for PcdNo in range(0, len(XYZO_axiel)):
#     XYZO_axiel[PcdNo].append(255)
#     XYZO_axiel[PcdNo].append(255)
#     XYZO_axiel[PcdNo].append(255)
#
# print('XYZO_axiel = ', XYZO_axiel)
# SaveFile('XYZO_axiel.xyz', XYZO_axiel)


for PcdNo in range(0, len(SamplePoint1)):
    SamplePoint1[PcdNo].append(255)
    SamplePoint1[PcdNo].append(255)
    SamplePoint1[PcdNo].append(255)
SaveFile('SamplePoint1.xyz', SamplePoint1)

oTc = np.vstack([x_p, y_p, z_p])
oTc = oTc.T
t = np.array([[CP1[0], CP1[1], CP1[2]]])
oTc = np.c_[oTc, t.T]
e = [[0, 0, 0, 1]]
oTc = np.r_[oTc, e]
cTo = inv(oTc)
SaveFile('cTo.xyz', cTo)

# sourceF = 'XYZC_axiel.xyz'
# targetF = 'XYZO_axiel.xyz'
# source = ReadXyzFile('{}'.format(sourceF))
# target = ReadXyzFile('{}'.format(targetF))
# source = np.asarray(source)
# target = np.asarray(target)


# target = o3d.io.read_point_cloud("XYZO_axiel.xyz")
# source = o3d.io.read_point_cloud("XYZC_axiel.xyz")
# print('dsfsf =',type(source))
# text = 'transformation.xyz'
# trans.demo_manual_registration(source, target, text)
#
# pcd_1 = o3d.io.read_point_cloud('SamplePoint1.xyz')
# toworld1_2 = np.genfromtxt('transformation.xyz', dtype=None, comments='#', delimiter=' ')
# pcd_1 = pcd_1.transform(toworld1_2)
# tra_inv = np.linalg.inv(toworld1_2)
# o3d.io.write_point_cloud("SamplePoint1Aftertrans.xyz", pcd_1)


pcd_1 = o3d.io.read_point_cloud('SamplePoint1.xyz')
tra = np.genfromtxt('cTo.xyz', dtype=None, comments='#', delimiter=' ')
pcd_1 = pcd_1.transform(tra)
tra_inv = np.linalg.inv(tra)
o3d.io.write_point_cloud("SamplePoint1Aftertrans.xyz", pcd_1)


SP = 'SamplePoint1Aftertrans.xyz'
SPfile = ReadXyzFile('{}'.format(SP))

GridPoint_dist = 0.4
data = np.asarray(SPfile)

mn = np.min(data, axis=0)  # Set minimum in X, Y, Z
mx = np.max(data, axis=0)

No_of_GridpointsX = math.ceil((mx[0] - mn[0]) / GridPoint_dist)  # 20
No_of_GridpointsY = math.ceil((mx[1] - mn[1]) / GridPoint_dist)

print('No of grid X', No_of_GridpointsX)
print('No of grid Y', No_of_GridpointsY)


X, Y = np.meshgrid(np.linspace(mn[0], mx[0], No_of_GridpointsX), np.linspace(mn[1], mx[1], No_of_GridpointsY))
XX = X.flatten()
YY = Y.flatten()

A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2, data[:, :2] ** 3]
C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 3, YY ** 3], C).reshape(X.shape)
ZZ = Z.flatten()

Grid = []
for pointNo in range(0, len(XX)):
    Grid.append([XX[pointNo], YY[pointNo], ZZ[pointNo]])

SaveFile('grid.xyz', Grid)
grid_before = o3d.io.read_point_cloud('grid.xyz')
grid_after = grid_before.transform(tra_inv)
o3d.io.write_point_cloud("grid_after.xyz", grid_after)

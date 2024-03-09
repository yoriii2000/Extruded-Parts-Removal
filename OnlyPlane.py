import numpy as np
import math
import scipy.linalg
import open3d as o3d
import shutil as shutil
import os as os
import random

from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

import NxyzSurfacfit as Nsurf
import worldxyzChange as WC

from scipy.optimize import curve_fit

def ReadXYZNormalFile(filename):
    print('File Path:', filename)
    f = open(filename, "r")
    lines = f.readlines()
    print('No of Points [XYZ]:', len(lines))
    PointList = []
    PointIdx = []
    NormalList = []

    for x in range(0, len(lines)):
        RawData = lines[x].strip().split(" ") #[x y z] from File

        # print(RawData)
        # print(float(RawData[0]))
        PointList.append([float(RawData[0]), float(RawData[1]), float(RawData[2])])

        NormalMagn = math.sqrt(pow(float(RawData[3]), 2) + pow(float(RawData[4]), 2) + pow(float(RawData[5]), 2))
        if NormalMagn == 0:
            NormalMagn = 0.00001
        NormalList.append([np.float(RawData[3]) / NormalMagn, np.float(RawData[4]) / NormalMagn, np.float(RawData[5]) / NormalMagn])


        # NormalList.append([float(RawData[3]), float(RawData[4]), float(RawData[5])])
        PointIdx.append(False)


    return PointList, PointIdx, NormalList

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

def CovXYZ(Value1, Value2, Value1Mean, Value2Mean):
    # Formula: Cov(x,x) = [Summation (xi-xmean)(yi-ymean)] / (n-1)

    # Summation Part
    Sum = 0
    for point_no in range(0, len(Value1)):
        Sum = Sum + ((Value1[point_no] - Value1Mean)*(Value2[point_no] - Value2Mean))
    # print('SumPart=', Sum)

    # Compute the Covariance
    Cov = Sum / (len(Value1) - 1)

    return Cov

def SeparateXYZ(Point):
    X = []
    Y = []
    Z = []

    XSum = 0
    YSum = 0
    ZSum = 0

    # print('Len', len(Point))
    for PointNo in range(0, len(Point)):

        X.append(Point[PointNo][0])
        Y.append(Point[PointNo][1])
        Z.append(Point[PointNo][2])
        #
        # X = [round(i, 3) for i in X]
        # Y = [round(i, 3) for i in Y]
        # Z = [round(i, 3) for i in Z]
        #
        # XSum = math.floor(XSum + Point[PointNo][0] * 1000) / 1000.0
        # YSum = math.floor(YSum + Point[PointNo][1] * 1000) / 1000.0
        # ZSum = math.floor(ZSum + Point[PointNo][2] * 1000) / 1000.0

        XSum = XSum + Point[PointNo][0]
        YSum = YSum + Point[PointNo][1]
        ZSum = ZSum + Point[PointNo][2]


    XMean = XSum / len(Point)
    YMean = YSum / len(Point)
    ZMean = ZSum / len(Point)

    # XMean = math.floor(XSum / len(Point) * 1000) / 1000.0
    # YMean = math.floor(YSum / len(Point) * 1000) / 1000.0
    # ZMean = math.floor(YSum / len(Point) * 1000) / 1000.0

    # print('Mean', XMean, YMean, ZMean)
    return X, Y, Z, XMean, YMean, ZMean

def OrientedBoundingBox(Point, offset, SF_BB_offset):
    # // FOR FINDING 8 POINTS OF BOUNDING BOX FOR REMOVED POINTS AND BOUNDING BOX FOR SURFACE FITTING

    # // We list the value of all sample points as X, Y, Z list
    x, y, z, XMean, YMean, ZMean = SeparateXYZ(Point)
    # print('XYZMean:', XMean, YMean, ZMean)

    # // Calculate Covariance Matrix
    C = [[CovXYZ(x, x, XMean, XMean), CovXYZ(x, y, XMean, YMean), CovXYZ(x, z, XMean, ZMean)],
         [CovXYZ(y, x, YMean, XMean), CovXYZ(y, y, YMean, YMean), CovXYZ(y, z, YMean, ZMean)],
         [CovXYZ(z, x, ZMean, XMean), CovXYZ(z, y, ZMean, YMean), CovXYZ(z, z, ZMean, ZMean)]]
    # print('\nCovariance Matrix for OBB\n', C)

    # // Calculate the eigenvector
    eigenvalue, eigenvector = np.linalg.eig(C)
    # print('EigenVector:', eigenvector)

    cluster = []
    # // Get the X Y Z min and max for bounding box
    for PointNo in range(0, len(Point)):
        # Project the point using PCA, so the coordinate x0,y0,z0 will change
        a = np.dot(Point[PointNo], eigenvector)
        cluster.append(np.dot(Point[PointNo], eigenvector))

        # Initial the XYZ min and max
        if PointNo == 0:
            XminPCA = a[0]
            XmaxPCA = a[0]
            YminPCA = a[1]
            YmaxPCA = a[1]
            ZminPCA = a[2]
            ZmaxPCA = a[2]


        # Find the XY min and max
        else:
            if a[0] < XminPCA:
                XminPCA = a[0]
            if a[0] > XmaxPCA:
                XmaxPCA = a[0]
            if a[1] < YminPCA:
                YminPCA = a[1]
            if a[1] > YmaxPCA:
                YmaxPCA = a[1]
            if a[2] < ZminPCA:
                ZminPCA = a[2]
            if a[2] > ZmaxPCA:
                ZmaxPCA = a[2]

    # SaveFile('Result/first_cluster{}.xyz'.format(EdgeCluster_no), cluster)

    # // Get the middle and bounding box points (Normal and Surface Fitting) at PCA coordinate

    ## Get the middle
    midPCA = [((XmaxPCA - XminPCA)/2) + XminPCA, ((YmaxPCA - YminPCA)/2) + YminPCA, ((ZmaxPCA - ZminPCA)/2) + ZminPCA]
    # print('mid_PCA:', midPCA)


    ## Get the 8 points of bounding box
    # 1.Bottom box, counter-clockwise rotate
    XminYminZminPCA = [XminPCA - offset, YminPCA - offset, ZminPCA - offset]
    XmaxYminZminPCA = [XmaxPCA + offset, YminPCA - offset, ZminPCA - offset]
    XmaxYmaxZminPCA = [XmaxPCA + offset, YmaxPCA + offset, ZminPCA - offset]
    XminYmaxZminPCA = [XminPCA - offset, YmaxPCA + offset, ZminPCA - offset]
    # 2.Top box, counter-clockwise rotate
    XminYminZmaxPCA = [XminPCA - offset, YminPCA - offset, (ZmaxPCA + offset) ]
    XmaxYminZmaxPCA = [XmaxPCA + offset, YminPCA - offset, (ZmaxPCA + offset) ]
    XmaxYmaxZmaxPCA = [XmaxPCA + offset, YmaxPCA + offset, (ZmaxPCA + offset) ]
    XminYmaxZmaxPCA = [XminPCA - offset, YmaxPCA + offset, (ZmaxPCA + offset) ]


    ## Get the 8 points of bounding box (FOR SURFACE FITTING)
    # X dan Y diperlebar ( makanya pakai SF_BB_offset)
    # 1.Bottom box, counter-clockwise rotate
    XminYminZmin_SF_PCA = [XminPCA - SF_BB_offset, YminPCA - SF_BB_offset, ZminPCA - SF_BB_offset]
    XmaxYminZmin_SF_PCA = [XmaxPCA + SF_BB_offset, YminPCA - SF_BB_offset, ZminPCA - SF_BB_offset]
    XmaxYmaxZmin_SF_PCA = [XmaxPCA + SF_BB_offset, YmaxPCA + SF_BB_offset, ZminPCA - SF_BB_offset]
    XminYmaxZmin_SF_PCA = [XminPCA - SF_BB_offset, YmaxPCA + SF_BB_offset, ZminPCA - SF_BB_offset]
    # 2.Top box, counter-clockwise rotate
    XminYminZmax_SF_PCA = [XminPCA - SF_BB_offset, YminPCA - SF_BB_offset, ZmaxPCA + SF_BB_offset]
    XmaxYminZmax_SF_PCA = [XmaxPCA + SF_BB_offset, YminPCA - SF_BB_offset, ZmaxPCA + SF_BB_offset]
    XmaxYmaxZmax_SF_PCA = [XmaxPCA + SF_BB_offset, YmaxPCA + SF_BB_offset, ZmaxPCA + SF_BB_offset]
    XminYmaxZmax_SF_PCA = [XminPCA - SF_BB_offset, YmaxPCA + SF_BB_offset, ZmaxPCA + SF_BB_offset]

    ## Get the HalfExtent_lengths of x y z (For normal bounding box)
    HalfExtent_lengths = math.sqrt(Sq2(XmaxYmaxZmaxPCA[0] - midPCA[0]) + Sq2(XmaxYmaxZmaxPCA[1] - midPCA[1]) + Sq2(XmaxYmaxZmaxPCA[2] - midPCA[2]))

    # // Return the middle and all of the bounding box points to the original coordinate using inverse matrix
    eigenvector_inv = np.linalg.inv(eigenvector)
    mid = np.dot(midPCA, eigenvector_inv)

    ## Normal Bounding Box
    # 1.Bottom box, counter-clockwise rotate
    XminYminZmin = np.dot(XminYminZminPCA, eigenvector_inv)
    XmaxYminZmin = np.dot(XmaxYminZminPCA, eigenvector_inv)
    XmaxYmaxZmin = np.dot(XmaxYmaxZminPCA, eigenvector_inv)
    XminYmaxZmin = np.dot(XminYmaxZminPCA, eigenvector_inv)
    # 2.Top box, counter-clockwise rotate
    XminYminZmax = np.dot(XminYminZmaxPCA, eigenvector_inv)
    XmaxYminZmax = np.dot(XmaxYminZmaxPCA, eigenvector_inv)
    XmaxYmaxZmax = np.dot(XmaxYmaxZmaxPCA, eigenvector_inv)
    XminYmaxZmax = np.dot(XminYmaxZmaxPCA, eigenvector_inv)

    ## Bounding Box (FOR SURFACE FITTING)
    # 1.Bottom box, counter-clockwise rotate
    XminYminZmin_SF = np.dot(XminYminZmin_SF_PCA, eigenvector_inv)
    XmaxYminZmin_SF = np.dot(XmaxYminZmin_SF_PCA, eigenvector_inv)
    XmaxYmaxZmin_SF = np.dot(XmaxYmaxZmin_SF_PCA, eigenvector_inv)
    XminYmaxZmin_SF = np.dot(XminYmaxZmin_SF_PCA, eigenvector_inv)
    # 2.Top box, counter-clockwise rotate
    XminYminZmax_SF = np.dot(XminYminZmax_SF_PCA, eigenvector_inv)
    XmaxYminZmax_SF = np.dot(XmaxYminZmax_SF_PCA, eigenvector_inv)
    XmaxYmaxZmax_SF = np.dot(XmaxYmaxZmax_SF_PCA, eigenvector_inv)
    XminYmaxZmax_SF = np.dot(XminYmaxZmax_SF_PCA, eigenvector_inv)

    # print('\nShow the center and all bounding box point:')
    # print(mid)
    # print('Bottom Side of Box(Counter Clockwise Rotation)')
    # print(XminYminZmin)
    # print(XmaxYminZmin)
    # print(XmaxYmaxZmin)
    # print(XminYmaxZmin)
    # print('Top Side of Box(Counter Clockwise Rotation)')
    # print(XminYminZmax)
    # print(XmaxYminZmax)
    # print(XmaxYmaxZmax)
    # print(XminYmaxZmax)

    # // Store the BoundingBox Points.
    BBPoint = [XminYminZmin, XmaxYminZmin, XmaxYmaxZmin, XminYmaxZmin, XminYminZmax, XmaxYminZmax, XmaxYmaxZmax,
               XminYmaxZmax]
    BBPoint_SF = [XminYminZmin_SF, XmaxYminZmin_SF, XmaxYmaxZmin_SF, XminYmaxZmin_SF, XminYminZmax_SF, XmaxYminZmax_SF, XmaxYmaxZmax_SF,
               XminYmaxZmax_SF]

    return mid, BBPoint, HalfExtent_lengths, BBPoint_SF

def SurfVar(Point):
    # Method: Surface Variation is the value to determine if the point is edge or not
    # We list the value of all sample points as X, Y, Z list
    x, y, z, XMean, YMean, ZMean = SeparateXYZ(Point)
    # print('XYZMean:', XMean, YMean, ZMean)

    # Calculate Covariance Matrix
    C = [[CovXYZ(x, x, XMean, XMean), CovXYZ(x, y, XMean, YMean), CovXYZ(x, z, XMean, ZMean)],
         [CovXYZ(y, x, YMean, XMean), CovXYZ(y, y, YMean, YMean), CovXYZ(y, z, YMean, ZMean)],
         [CovXYZ(z, x, ZMean, XMean), CovXYZ(z, y, ZMean, YMean), CovXYZ(z, z, ZMean, ZMean)]]
    # print('Covariance Matrix:\n', C)

    # Calculate the eigenvalue(3 Lambda)
    eigenvalue, eigenvector = np.linalg.eig(C)
    # print('EigenValue:', eigenvalue)

    # Sort the lambda as Lambda0<Lambda1<Lambda2
    Lambda = []
    for eigenvalue_no in range(0, len(eigenvalue)):
        if eigenvalue_no == 0:
            Lambda.append(eigenvalue[0])
        else:
            InsertedCheck = False  # To check if the value is smaller than the value in the list or not
            for Lambda_no in range(0, len(Lambda)):
                if eigenvalue[eigenvalue_no] < Lambda[Lambda_no]:
                    Lambda.insert(Lambda_no, eigenvalue[eigenvalue_no])
                    InsertedCheck = True
                    break

            # There's no value in list which is higher than this new value
            if InsertedCheck == False:
                Lambda.append(eigenvalue[eigenvalue_no])

    # print('Lambda 0 to 3:', Lambda)

    # Calculate the Surface Variation
    SurfaceVariation = Lambda[0] / (Lambda[0] + Lambda[1] + Lambda[2])
    # print('Surface Variation:', SurfaceVariation)

    return(SurfaceVariation)

def SampleThePoint(MainPoint, tree, K_number = 5, KNNRadius = 1.5):

    distances, indices = tree.query(MainPoint, K_number)

    SampledPoints = []
    for i in range(0, K_number):
        if distances[0][i] <= KNNRadius:
            SampledPoints.append(Pcd[int(indices[0][i])])

    # print('SampledPoints:', SampledPoints)
    return SampledPoints

def Sq2(value):
    # Code untuk Kuadrat bilangan
    return value*value

def uvw(PointBase, Point):
    # Choose one vector as reference first (from the base point to one random point)
    # Find the vector which has smallest distance
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
        #
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
        d = math.sqrt(
            Sq2(ChosenPoint[i][0] - PointBase[0]) + Sq2(ChosenPoint[i][1] - PointBase[1]) + Sq2(ChosenPoint[i][2] - PointBase[2]))

        DistIsSmaller_Sign = False

        if i == 0:  # initial
            FinalChosenPointD_list.append(d)
            FinalChosenPoint.append([ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])

        else:
            for FinalChosenPoint_no in range(0, len(FinalChosenPoint)):
                if d < FinalChosenPointD_list[FinalChosenPoint_no]:
                    FinalChosenPointD_list.insert(FinalChosenPoint_no, d)
                    FinalChosenPoint.insert(FinalChosenPoint_no,
                                            [ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])
                    DistIsSmaller_Sign = True
                    break

            if DistIsSmaller_Sign == False:
                FinalChosenPointD_list.append(d)
                FinalChosenPoint.append([ChosenPoint[i][0], ChosenPoint[i][1], ChosenPoint[i][2]])

    FinalChosenPoint[2] = Point_ref

    # print('\n Final Chosen point')
    # print(FinalChosenPoint)
    # print('Base Point')
    # print(PointBase)
    return FinalChosenPoint

def CheckPointsInsideOBB(SampledPoint, OBBPoint):
    # print('\nOBB point', OBBPoint)
    # // Have to wider the OBBPoint
    # print('OBBPoint', OBBPoint)

    # // Check if points are insert bounding box, OBBPoint, SampledPoint
    # 1. Get 4 points for the requirement of the next function
    # Illustration:
    #            . (Point 1)
    #            .
    #            .
    #            .
    #            .
    #  (Point 0) . . . . . . . (Point 2)
    #           .
    # (Point 3).

    # Algorithm:
    # The diagonal distance will be more far than not

    uvwPoint = uvw(OBBPoint[0], OBBPoint)
    # print('OBB_BasePoint', OBBPoint[0])
    # print('OBB_uvwPoint', uvwPoint)


    # 2. Check if the sampled points are inside the OBB
    # print('\nCheck if the points are inside the OBB')

    CheckedPoint = SampledPoint

    u = np.subtract(uvwPoint[0], OBBPoint[0])
    v = np.subtract(uvwPoint[1], OBBPoint[0])
    w = np.subtract(uvwPoint[2], OBBPoint[0])
    i = np.subtract(CheckedPoint, OBBPoint[0])

    # Check if the point is inside the bounding box or not
    if 0 <= round(np.dot(i, u), 7) <= round(np.dot(u, u), 7):
        if 0 <= round(np.dot(i, v), 7) <= round(np.dot(v, v), 7):
            if 0 <= round(np.dot(i, w), 7) <= round(np.dot(w, w), 7):
                inside = True
            else:
                inside = False
        else:
            inside = False
    else:
        inside = False
    # if inside == True:
    #     print('Inside')
    # else:
        # print('Outside')
        # print(round(np.dot(i, u), 7), round(np.dot(u, u), 7))
        # print(round(np.dot(i, v), 7), round(np.dot(v, v), 7))
        # print(round(np.dot(i, w), 7), round(np.dot(w, w), 7))
        # print('\n')
    return inside

def Cluster(Point, SampleDistance = 2, min_samples=2):
    data = np.asarray(Point)
    # Start Clustering (but not sorted yet in this part)
    model = DBSCAN(eps=SampleDistance, min_samples=min_samples)
    model.fit_predict(data)

    # Prepare the list, prepare the list inside clusterlist.
    ClusterList = [[] for _ in range(len(set(model.labels_)))]

    # In clustering, we maybe will find points which are noise, so if found noise, noise status will become True.
    # The noise points will be grouped in one cluster (-1) and will be removed.
    noise = False
    print('Total Found ClusterList :', len(ClusterList))

    # Start sorting the data based on index number of clustering [after clustering step]
    for data_no in range(0, len(data)):
        # Check the point belongs to which cluster (cluster 1, cluster 2, cluster 3?)
        clusterIdx = model.labels_[data_no]

        # index = -1 means it is noise point
        if clusterIdx != -1:
            ClusterList[clusterIdx].append(Point[data_no])

        # Tell the program that there are noise points
        elif clusterIdx == -1:
            noise = True

    # Remove List which contains noise points
    if noise:
        ClusterList.pop(len(ClusterList) - 1)
        print('(There is cluster noise)')

    return ClusterList

def solve_plane(A, B, C):
    """
    求解平面方程
    :param A: 點A
    :param B: 點B
    :param C: 點C
    :return: 點(平面上一點), 四元數(平面四元數), Nx(平面的法向量)
    """

    # 兩個常量
    N = np.array([0, 0, 1])
    Pi = 3.141592653589793238462643383279502884197169

    # 計算平面的單位法向量，即BC 與 BA的叉積
    Nx = np.cross(B - C, B - A)
    Nx = Nx / np.linalg.norm(Nx)

    # 計算單位旋轉向量與旋轉角（範圍為0到Pi）
    Nv = np.cross(Nx, N)
    angle = math.acos(np.dot(Nx, N))

    # 考慮到兩個向量夾角不大於Pi/2，這裡需要處理一下
    if angle > Pi / 2.0:
        angle = Pi - angle
        Nv = -Nv

    # FIXME: 此處如何確定平面上的一個點？？？
    # Point = (A + B + C) / 3.0
    Point = B
    # 計算單位四元數
    Quaternion = np.append(Nv * math.sin(angle / 2), math.cos(angle / 2))

    # print("旋轉向量:\t", Nv)
    # print("旋轉角度:\t", angle)
    # print("對應四元數:\t", Quaternion)

    return Point, Quaternion, Nx

def solve_distance(M, P, N):
    """
    求解點M到平面(P,Q)的距離
    :param M: 點M
    :param P: 平面上一點
    :param N: 平面的法向量
    :return: 點到平面的距離
    """

    # 從四元數到法向量
    # A = 2 * Q[0] * Q[2] + 2 * Q[1] * Q[3]
    # B = 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3]
    # C = -Q[0] ** 2 - Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2
    # D = -A * P[0] - B * P[1] - C * P[2]

    # 為了計算簡便，直接使用求解出的法向量
    A = N[0]
    B = N[1]
    C = N[2]
    D = -A * P[0] - B * P[1] - C * P[2]

    return math.fabs(A * M[0] + B * M[1] + C * M[2] + D) / math.sqrt(A ** 2 + B ** 2 + C ** 2)

def RANSAC(data):
    """
    使用RANSAC算法估算模型
    """
    # 數據規模
    SIZE = data.shape[0]
    # 迭代最大次數，每次得到更好的估計會優化iters的數值，默認10000
    iters = 10000
    # 數據和模型之間可接受的差值，默認0.25
    sigma = 0.2
    # 內點數目
    pretotal = 0
    # 希望的得到正確模型的概率，默認為0.99
    Per = 0.999

    P = np.array([])
    Q = np.array([])
    N = np.array([])
    for i in range(iters):
        # 隨機在數據中選出三個點去求解模型
        sample_index = random.sample(range(SIZE), 3)
        P, Q, N = solve_plane(data[sample_index[0]], data[sample_index[1]], data[sample_index[2]])

        # 算出內點數目
        total_inlier = 0
        for index in range(SIZE):
            if solve_distance(data[index], P, N) < sigma:
                total_inlier = total_inlier + 1


        # # 判斷當前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            iters = math.log(1 - Per) / math.log(1 - pow(total_inlier / SIZE, 2))
            pretotal = total_inlier
            maxP = P; maxQ = Q; maxN = N;

        # if Per >= 1:
        #     # Adjust Per or handle this scenario specifically
        #     Per = 0.999  # Example adjustment, choose a sensible default for your case
        #
        # if total_inlier >= SIZE:
        #     # Handle this scenario specifically, perhaps by limiting total_inlier or adjusting SIZE
        #     total_inlier = SIZE - 1  # Example adjustment, ensure this makes sense for your application
        # # Now perform the calculation
        # iters = math.log(1 - Per) / math.log(1 - pow(total_inlier / SIZE, 2))
        # pretotal = total_inlier
        # maxP = P; maxQ = Q; maxN = N;


        # # 判斷當前模型是否已經符合超過一半的點
        # if total_inlier > SIZE * 0.85:
        #     break
    return maxP, maxQ, maxN

# --------- MAIN ---------.
# Input File Location
SourcePointFile = 'Source/metalmodel.xyz' #修正前 metalmodel.xyz   修正後 metalmodelcentor.xyz
NewFileXYZ = 'Result/NewFile.xyz'
# Output File Location
NameOfEdgeFile = 'Result/EdgePoint.xyz'
# NewEdgeFile = 'Result/NewEdgeFile.xyz'
NameOfRemovedPointsFile = 'Result/RemovedPoints.xyz'
NameOfUnremovedPointsFile = 'Result/UnremovedPoints.xyz'
NameOfSF_SampledPoints_ClusterFolder = 'Result'
NameOfSF_ClusterFolder = 'Result'
FinalFile = 'Result/FinalSF.xyz'
NameOfHighest = 'Process/Highest.xyz'
NameOfMesh = 'Process/Plane.xyz'
NameOfFinalPlane = 'Result/SF_Grid_List.xyz'
NameOfBB = 'Result/BB.xyz'
# -------------------------------------------------------------------------------------------------
removed = o3d.io.read_point_cloud(NameOfRemovedPointsFile)
RemovedPoints = np.asarray(removed.points)
unremoved = o3d.io.read_point_cloud(NameOfUnremovedPointsFile)
UnRemovedPoints = np.asarray(unremoved.points)
# unremoved_tree = KDTree(UnRemovedPoints)

# 找尋最高的那一圈 (highest_points)
z_coordinates = UnRemovedPoints[:, 2]
sorted_indices = np.argsort(-z_coordinates)

N = 450  # BIG 450   MID 420   SMALL 350
highest_points_indices = sorted_indices[:N]
highest_points = UnRemovedPoints[highest_points_indices]

SaveFile(NameOfHighest, highest_points)

# 利用 RANSAC 找出 P(平面上的一点)，Q(平面的四元数,此處不會用到)，N(平面的法向量)
P, Q, N = RANSAC(highest_points)
print("N:", N)
if N[2] < 0:
    N[0] = -N[0]
    N[1] = -N[1]
    N[2] = -N[2]

SaveFile('Result/NormalForRef.xyz', N.reshape(-1, 3))
# 定義平面 Ax + By + Cz + D = 0

A, B, C = N
D = -np.dot(N, P)

# 生成點雲網格 (mesh)
num_points = 100
x_range = [-20, 20]
y_range = [-20, 20]
x_vals = np.linspace(x_range[0], x_range[1], num_points)
y_vals = np.linspace(y_range[0], y_range[1], num_points)
xx, yy = np.meshgrid(x_vals, y_vals)
zz = (-A * xx - B * yy - D) / C

mesh = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
SaveFile(NameOfMesh, mesh)

# 生成 highest_points 的 Bounding Box (BB) (為了在平面上切出想要的範圍)
Highest_pcd = o3d.io.read_point_cloud(NameOfHighest)
BB = Highest_pcd.get_oriented_bounding_box()
_, BB, _, _ = OrientedBoundingBox(highest_points, 0, 0)
SaveFile(NameOfBB, BB)

# 將 BB 內範圍保存下來 FinalPlane = 最終結果
FinalPlane = []
for mesh_no in range(0, len(mesh)):
    InsidePoints = CheckPointsInsideOBB(mesh[mesh_no], BB)
    if InsidePoints == True:
        FinalPlane.append(mesh[mesh_no])
#
# FinalPlane = []
# for mesh_no in range(0, len(mesh)):
#     point_in_obb_local = np.linalg.inv(BB.R).dot(mesh[mesh_no] - BB.center)
#     InsidePoints = np.all(-BB.extent / 2 <= point_in_obb_local) and np.all(point_in_obb_local <= BB.extent / 2)
#     if InsidePoints == True:
#         FinalPlane.append(mesh[mesh_no])
# # # 将查询点转换为OBB的局部坐标系
# # point_in_obb_local = np.linalg.inv(BB.R).dot(mesh[mesh_no] - BB.center)
# # point_in_obb_local = np.linalg.inv(obb.R).dot(query_point - obb.center)
# # # 检查点是否在OBB内部
# # inside = np.all(-BB.extent / 2 <= point_in_obb_local) and np.all(point_in_obb_local <= BB.extent / 2)


print('len(FinalPlane)=', len(FinalPlane))

SaveFile(NameOfFinalPlane, FinalPlane)

# ----------------------------------------------------------------------------------------------------

ToeWithSkin = np.vstack([RemovedPoints, highest_points])
SaveFile('Result/SF_SampledPoints_Cluster_0.xyz', ToeWithSkin)
file = 'Result/SF_SampledPoints_Cluster_0.xyz'
_, OBBs, _, _ = OrientedBoundingBox(ToeWithSkin, 0, 0)
SaveFile('Process/OBBpoint_SF0.xyz', OBBs)
OBBsffile = 'Process/OBBpoint_SF0.xyz'

Grid, tra_inv, tra, XYZC_axiel = Nsurf.XYZchange(file, OBBsffile)

SaveFile('Result/tra_inv_0.xyz', tra_inv)  # tra_inv:算完反推原坐標系
SaveFile('Result/tra_0.xyz', tra)  # tra:原點轉換到凸點原點
SaveFile('Process/XYZC_axiel_0.xyz', XYZC_axiel)



print('\n------ PROGRAM END ----------\n')

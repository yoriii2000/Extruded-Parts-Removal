import numpy as np
import math
import scipy.linalg

from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN


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
        RawData = lines[x].strip().split(" ") #[x y z] from File
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

        XSum = XSum + Point[PointNo][0]
        YSum = YSum + Point[PointNo][1]
        ZSum = ZSum + Point[PointNo][2]

    XMean = XSum / len(Point)
    YMean = YSum / len(Point)
    ZMean = ZSum / len(Point)

    # print('Mean', XMean, YMean, ZMean)
    return X, Y, Z, XMean, YMean, ZMean

def OrientedBoundingBox(Point, offset = 0.2, SF_BB_offset = 0.4):
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

    # // Get the X Y Z min and max for bounding box

    for PointNo in range(0, len(Point)):
        # Project the point using PCA, so the coordinate x0,y0,z0 will change
        a = np.dot(Point[PointNo], eigenvector)

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

def SurfaceFit(SourcePoint, GridPoint_dist = 0.1): #1.5

    # // PAC the data because surface fitting needs the points face to Z axis
    # We list the value of all sample points as X, Y, Z list
    x, y, z, XMean, YMean, ZMean = SeparateXYZ(SourcePoint)
    # print('XYZMean:', XMean, YMean, ZMean)

    # Calculate Covariance Matrix
    C = [[CovXYZ(x, x, XMean, XMean), CovXYZ(x, y, XMean, YMean), CovXYZ(x, z, XMean, ZMean)],
         [CovXYZ(y, x, YMean, XMean), CovXYZ(y, y, YMean, YMean), CovXYZ(y, z, YMean, ZMean)],
         [CovXYZ(z, x, ZMean, XMean), CovXYZ(z, y, ZMean, YMean), CovXYZ(z, z, ZMean, ZMean)]]
    # print('\nCovariance Matrix for OBB\n', C)

    # Calculate the eigenvector
    eigenvalue, eigenvector = np.linalg.eig(C)
    print('EigenVector:', eigenvector)

    # Project the points using PCA, so the coordinate x,y,z of points will change
    PAC_point = []
    for PointNo in range(0, len(SourcePoint)):
        PAC_point.append(np.dot(SourcePoint[PointNo], eigenvector))


    # // SURFACE FITTING
    data = np.asarray(PAC_point)

    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)   #Set minimum in X, Y, Z
    mx = np.max(data, axis=0)

    #  Get number of grid X and grid Y
    # No_of_GridpointsX = math.ceil((mx[0] - mn[0]) / GridPoint_dist) #20
    # No_of_GridpointsY = math.ceil((mx[1] - mn[1]) / GridPoint_dist)


    No_of_GridpointsX = math.ceil((mx[0] - mn[0]) / GridPoint_dist)  # 20
    No_of_GridpointsY = math.ceil((mx[1] - mn[1]) / GridPoint_dist)

    # No_of_GridpointsX = math.ceil((mx[2] - mn[2]) / GridPoint_dist) #20
    # No_of_GridpointsY = math.ceil((mx[0] - mn[0]) / GridPoint_dist)

    print('No of grid X', No_of_GridpointsX)
    print('No of grid Y', No_of_GridpointsY)

    #  // Create Grid
    #  ----------- Explanation: ---------------------------
    #  linspace(start, stop, num)
    #  suppose linspace(2,3,5) , it means there are 5 numbers between 2 and 3 , so the result is  [2, 2.25, 2.5, 2.75, 3]

    #  meshgrid(x,y) , e.g x=[1,2,3] y=[1,2,3,4]
    #  so for X =   [1,2,3],
    #               [1,2,3],
    #               [1,2,3],
    #               [1,2,3]     because Y has 4 numbers, so X is copied 4 times (from top to bottom)

    # so for Y =    [1,1,1],
    #               [2,2,2],
    #               [3,3,3],
    #               [4,4,4],    because X has 3 numbers, so Y is copied 3 times (from left to right)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], No_of_GridpointsX), np.linspace(mn[1], mx[1], No_of_GridpointsY))

    # Make X become 1 array, also for Y
    XX = X.flatten()
    YY = Y.flatten()

    # best-fit linear plane (2nd-order)
    # A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    # C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])  # coefficients
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2, data[:, :2] ** 3]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

    # Calculate the Z for all mesh grid point (because meshgrid is 2D and we want to make it become 3D)
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 3, YY ** 3], C).reshape(X.shape)
    ZZ = Z.flatten()

    # Insert the Z value to each mesh grid point
    Grid_PAC = []
    for pointNo in range(0, len(XX)):
        Grid_PAC.append([XX[pointNo], YY[pointNo], ZZ[pointNo]])

    # // Return the result of surface fitting to the original coordinate using inverse matrix
    Grid = []
    eigenvector_inv = np.linalg.inv(eigenvector)
    for PointNo in range(0, len(Grid_PAC)):
        Grid.append(np.dot(Grid_PAC[PointNo], eigenvector_inv))

    return Grid




# --------- MAIN ---------.
# Input File Location
NameOfSourcePointFile = 'Source/model9.xyz' #Golf_Clubhead_3ext.xyz

# Output File Location
NameOfEdgeFile = 'Result/EdgePoint.xyz'
NewEdgeFile = 'Result/NewEdgeFile.xyz'
NameOfRemovedPointsFile = 'Result/RemovedPoints.xyz'
NameOfUnremovedPointsFile = 'Result/UnremovedPoints.xyz'
NameOfSF_SampledPoints_ClusterFolder = 'Result'
NameOfSF_ClusterFolder = 'Result'

# Parameter Setting
SurfVarThreshold = 0.01  # makin kecil ,edge semakin keliatan 0.01
KNNRadius = 5 # 0.5
K_number = 70 # 50
OrientedBoundingBox_Offset = 0.5 # 0.2
SF_BB_offset = OrientedBoundingBox_Offset + 1.5
EachSurfaceFittingPoint_Distance = 0.4


print('\n------ PROGRAM START ----------\n')

# // ----------- READ THE SOURCE FILE--------------------
print('READ THE SOURCE FILE')
# PointList, PointIdx, NormalList = ReadXYZNormalFile(NameOfSourcePointFile)    # Reading File which contains normals
PointList = ReadXyzFile(NameOfSourcePointFile)                                  # Reading File which contains only point



# // ------------- DETECT THE 3D EDGE -----------------------
print('\nDETECTING THE 3D EDGE')
Pcd = PointList
tree = KDTree(Pcd, leaf_size=40)
EdgePoint = []
for Pcd_no in range(0, len(Pcd)):
    # //Prepare the sampled points(KNN Points) for this chosen point
    MainPoint = [Pcd[Pcd_no]]
    SampledPoints = SampleThePoint(MainPoint, tree, K_number=K_number, KNNRadius= KNNRadius)

    # //SampledPoints < 3, we assume as noise point.
    if len(SampledPoints) >= 3:
        # Start to calculate the surface variation
        SurfaceVariation = SurfVar(SampledPoints)
    else:
        SurfaceVariation = 0.0
        print('SampledPoints less than 3.')
    # print('SurfaceVariation =', SurfaceVariation)

    # //Set the point as edge , based on the threshold of Surface Variation
    # SurfaceVariation = 0 if the point belongs to flat surface
    if SurfaceVariation >= SurfVarThreshold:
        EdgePoint.append(MainPoint[0])

#//Tell user if there's edge point
if len(EdgePoint) == 0:
    print('No Edge Point is detected')
else:
    # print('EdgePoint:', EdgePoint[5447][1])
    print('Edge Points are detected')
    print('EdgePoint N', len(EdgePoint))
    SaveFile(NameOfEdgeFile, EdgePoint)


# //---------------------------- CLUSTER AND BOUNDING BOX THE EDGE POINT ------------------------------------------
print('\nCLUSTER AND BOUNDING BOX THE EDGE POINT')
# //Read File
# EdgePoint = ReadXyzFile('{}'.format(NameOfEdgeFile))

# //Cluster the edge points
EdgePoint_Cluster = Cluster(EdgePoint, SampleDistance=2, min_samples=2) #EdgePoint

# // Get the Oriented Bounding Box of each edge point clusters and Bounding Box points for surface fitting.
OBBPoint_list = []
OBBPoint_SF_list =[]  # This is for surface fitting
for EdgeCluster_no in range(0, len(EdgePoint_Cluster)):
    SaveFile('Cluster{}.xyz'.format(EdgeCluster_no), EdgePoint_Cluster[EdgeCluster_no])
    OBBMidPoint, OBBPoint, HalfExtent_lengths, OBBPointSF = OrientedBoundingBox(EdgePoint_Cluster[EdgeCluster_no], offset=OrientedBoundingBox_Offset,SF_BB_offset = SF_BB_offset)

    OBBPoint_list.append(OBBPoint)       # Bounding Box for Removed Point
    OBBPoint_SF_list.append(OBBPointSF)  # Bounding Box for Surface Fitting

for OBBPoint_list_no in range(0, len(OBBPoint_list)):
    SaveFile('OBBpoint{}.xyz'.format(OBBPoint_list_no), OBBPoint_list[OBBPoint_list_no])


for OBBPoint_SF_list_no in range(0, len(OBBPoint_SF_list)):
    SaveFile('OBBpoint_SF{}.xyz'.format(OBBPoint_SF_list_no), OBBPoint_SF_list[OBBPoint_SF_list_no])


# // ----------- SEPARATE REMOVED AND UNREMOVED POINTS ---------------------------
print('\nSEPARATE REMOVED AND UNREMOVED POINTS')
# Method: Check if the point cloud is inside the bounding box, then we set it as removed point.
RemovedPoints = []
UnRemovedPoints = []

# Check the point one by one
for Pcd_no in range(0, len(Pcd)):
    # Check if the point is inside in the one of the bounding box
    for OBBPoint_list_no in range(0, len(OBBPoint_list)):
        # print('OBBno', OBBPoint_list_no)
        # print('Pcd[Pcd_no]:', Pcd[Pcd_no])
        # print('OBBPoint_list[OBBPoint_list_no]:',OBBPoint_list[OBBPoint_list_no])

        # Check if it is inside Bounding Box or not, the function will return True or False
        InsidePoints = CheckPointsInsideOBB(Pcd[Pcd_no], OBBPoint_list[OBBPoint_list_no])

        if InsidePoints == True:
            RemovedPoints.append(Pcd[Pcd_no])
            # print('Removed the point no:", Pcd_no)
            break
        # if non inside, then code will check if the point is in the other bounding box

    # If this point is not in the all of the bounding box, then code set it as unremoved point.
    if InsidePoints != True:
        UnRemovedPoints.append(Pcd[Pcd_no])
        # print('Unremoved the point no:", Pcd_no)

# Save the removed and unremoved point
SaveFile(NameOfRemovedPointsFile, RemovedPoints)
SaveFile(NameOfUnremovedPointsFile, UnRemovedPoints)





# // ----------- FIND THE HOLE PARTS AND FILL THEM WITH SURFACE FITTING   ---------------------------
print('\nFIND THE HOLE PARTS AND FILL THEM WITH SURFACE FITTING')
# Method: Check if the point cloud is inside the bounding box(FOR SURFACE FITTING), then we set it as Surface Fitting sampled point.

SF_SampledPoints = []  # This is for collecting points around the removed parts which will be used as sample point for surface fitting.

for SF_BoundingBox_Cluster_No in range(0, len(OBBPoint_SF_list)):
    SF_SampledPoints.append([])


# Check the point one by one
for UnRemovedPoints_no in range(0, len(UnRemovedPoints)):
    # Check if the point is inside in the one of the bounding box
    for OBBPoint_SF_list_no in range(0, len(OBBPoint_SF_list)):
        # print('OBB_SF no:', OBBPoint_list_no)
        # print('UnRemovedPoints[UnRemovedPoints_no]:', UnRemovedPoints[UnRemovedPoints_no])
        # print('OBBPoint_SF_list[OBBPoint_SF_list_no]:',OBBPoint_SF_list[OBBPoint_SF_list_no])

        # Check if it is inside Bounding Box or not, the function will return True or False
        InsidePoints = CheckPointsInsideOBB(UnRemovedPoints[UnRemovedPoints_no], OBBPoint_SF_list[OBBPoint_SF_list_no])

        if InsidePoints == True:
            SF_SampledPoints[OBBPoint_SF_list_no].append(UnRemovedPoints[UnRemovedPoints_no])
            # print('Removed the point no:", Pcd_no)
            break
        # if non inside, then code will check if the point is in the other bounding box

    # If this point is not in the all of the bounding box, then code just ignore it.
print('SF_SampledPoints = ', SF_SampledPoints)

# Save the SF Sampled Points (or Hole Boundary Points)
for SF_BoundingBox_Cluster_No in range(0, len(SF_SampledPoints)):
    SaveFile('{}/SF_SampledPoints_Cluster_{}.xyz'.format(NameOfSF_SampledPoints_ClusterFolder, SF_BoundingBox_Cluster_No), SF_SampledPoints[SF_BoundingBox_Cluster_No])

# Surface Fitting for each SF_BoundingBox_Cluster
print('\nSURFACE FITTING')
SF_Grid_List = []
for SF_BoundingBox_Cluster_No in range(0, len(SF_SampledPoints)):
    SF_Grid_Filtered = []
    SF_Grid = SurfaceFit(SF_SampledPoints[SF_BoundingBox_Cluster_No], GridPoint_dist=EachSurfaceFittingPoint_Distance)

    # Check the point one by one
    for SF_Grid_no in range(0, len(SF_Grid)):
        # Check if the point is inside in the one of the bounding box
        for OBBPoint_list_no in range(0, len(OBBPoint_list)):
             # Check if it is inside Bounding Box or not, the function will return True or False
            InsidePoints = CheckPointsInsideOBB(SF_Grid[SF_Grid_no], OBBPoint_list[OBBPoint_list_no])

            if InsidePoints == True:
                SF_Grid_Filtered.append(SF_Grid[SF_Grid_no])
                # print('Removed the point no:", Pcd_no)
                SF_Grid_List.append(SF_Grid[SF_Grid_no])

                break
            # if non inside, then code will check if the point is in the other bounding box

        # If this point is not in the all of the bounding box, then code just ignore it.

    SaveFile('{}/SF_Grid_{}.xyz'.format(NameOfSF_ClusterFolder, SF_BoundingBox_Cluster_No), SF_Grid_Filtered)

SaveFile('Result/SF_Grid_List.xyz', SF_Grid_List)

# Fill the removed part

afterSF = []

afterSF = UnRemovedPoints + SF_Grid_List
SaveFile('Result/After_SF.xyz', afterSF)

print('\n------ PROGRAM END ----------\n')




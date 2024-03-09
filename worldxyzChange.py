
import numpy as np

import transformat as trans
import open3d as o3d


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

def ChangeSystem(SourcePointFile):

    allpoint = ReadXyzFile(SourcePointFile)
    print('allpoint')
    basexy = []
    nxyz = []
    minnn = np.min(allpoint, axis=0)
    print(minnn[2])
    for allpoint_no in range(0, len(allpoint)):
        # if (EdgePoint[EdgePoint_no][2]>1.5 and EdgePoint[EdgePoint_no][2]<39):
        if (allpoint[allpoint_no][2] == minnn[2]):
            basexy.append(allpoint[allpoint_no])


    data = np.asarray(basexy)
    print(type(data))
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)

    nx = (mx[0] - mn[0]) / 2
    ny = (mx[1] - mn[1]) / 2
    nxyz.append([nx, ny, 0])

    print('nxyz = ', nxyz)
    # SaveFile('nxyz.xyz', nxyz)
    Uni = 1
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
    # SaveFile('basexyz.xyz', basexyz)

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
    # SaveFile('Oxyz.xyz', Oxyz)

    # target = o3d.io.read_point_cloud("Oxyz.xyz")
    # source = o3d.io.read_point_cloud("basexyz.xyz")
    # text = 'tran.xyz'
    # trans.demo_manual_registration(source, target, text)
    #
    # pcd_1 = o3d.io.read_point_cloud(SourcePointFile)
    # toworld1_2 = np.genfromtxt('tran.xyz', dtype=None, comments='#', delimiter=' ')
    # pcd_1 = pcd_1.transform(toworld1_2)
    # o3d.io.write_point_cloud("Result/modl5Aftertrans.xyz", pcd_1)
    # NameOfSourcePointFile = 'Result/modl5Aftertrans.xyz'

#
    text = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t = np.array([[nx,ny,0]])
    oTb = np.c_[text,t.T]
    e = [[0, 0, 0, 1]]
    oTb = np.r_[oTb, e]
    SaveFile('Process/oTb.xyz', oTb)

    pcd_1 = o3d.io.read_point_cloud(SourcePointFile)
    toworld1_2 = np.genfromtxt('Process/oTb.xyz', dtype=None, comments='#', delimiter=' ')
    bTo = np.linalg.inv(toworld1_2)

    pcd_1 = pcd_1.transform(bTo)
    o3d.io.write_point_cloud("Result/modl5Aftertrans.xyz", pcd_1)
    NameOfSourcePointFile = 'Result/modl5Aftertrans.xyz'





#
    return NameOfSourcePointFile




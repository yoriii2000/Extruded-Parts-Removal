import shutil as shutil
import open3d as o3d
import os
import numpy as np



def ReadXyzFile(filename):
    # print('File Path:', filename)
    f = open(filename, "r")
    lines = f.readlines()
    # print('No of Points [XYZ]:', len(lines))
    PointList = []

    for x in range(0, len(lines)):
        RawData = lines[x].strip().split() #[x y z] from File
        PointList.append([float(RawData[0]), float(RawData[1]), float(RawData[2])])

    return PointList
def SaveFile(Pcd_File_Name, PCDList):
    np.savetxt('{}'.format(Pcd_File_Name), PCDList, delimiter=' ')
    print('Saved File: [{}].'.format(Pcd_File_Name))

shutil.rmtree('c:\\Users\\User\\Desktop\\陳昱廷\\Layering\\AfterGrinding')
os.makedirs('c:\\Users\\User\\Desktop\\陳昱廷\\Layering\\AfterGrinding')


# Load the two point cloud files
pcd1 = o3d.io.read_point_cloud("C:\\Users\\User\\Desktop\\陳昱廷\\Layering\\compare\\Aftergrid.xyz")
pcd2 = o3d.io.read_point_cloud("C:\\Users\\User\\Desktop\\陳昱廷\\Layering\\compare\\After_SF.xyz")


# Visualize the two point clouds before alignment
o3d.visualization.draw_geometries([pcd1, pcd2])

# downsample both point clouds
voxel_size = 0.05
pcd1_down = pcd1.voxel_down_sample(voxel_size)
pcd2_down = pcd2.voxel_down_sample(voxel_size)

# estimate normal vectors for both point clouds
pcd1_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
pcd2_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

# Perform registration
threshold = 3.2
trans_init = np.identity(4)

reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1_down, pcd2_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))

pcd1_transformed = pcd1.transform(reg_p2p.transformation)
pcd2_transformed = pcd2.transform(reg_p2p.transformation)
o3d.io.write_point_cloud("test/icpresult1_6.xyz", combine_result)
# # color point clouds based on distance from corresponding points
# distances = np.asarray(reg_p2p.inlier_rmse)
# inlier_idx = np.where(distances <= threshold)[0]
# outlier_idx = np.where(distances > threshold)[0]
# print('inlier_idx', inlier_idx)
# print('outlier_idx', outlier_idx)
# pcd1.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd1.points), 3)))
# pcd1.colors[inlier_idx] = [0.5, 0.5, 0.5]  # inliers are gray
# # pcd1.colors[outlier_idx] = [1, 0, 0]  # outliers are red
# # pcd2.colors = pcd1.colors
# #
# # # visualize the result
# o3d.visualization.draw_geometries([pcd1, pcd2])
# 將拼合後的點雲設為同一顏色
pcd1_transformed.paint_uniform_color([1, 0, 0])
pcd2_transformed.paint_uniform_color([0, 1, 0])
pcd = pcd1_transformed + pcd2_transformed

distances = pcd.compute_point_cloud_distance(pcd2_transformed)
distances = np.asarray(distances)
# 找出不重疊的點雲
threshold = 1
non_overlap_idx = distances > threshold

# 將不重疊的點雲標示為紅色
pcd.colors = o3d.utility.Vector3dVector(
    [[1, 0, 0] if i else [0, 0, 0] for i in non_overlap_idx])

# 刪除重疊的點雲
pcd = pcd.select_by_index(np.where(non_overlap_idx == True)[0])

# 顯示點雲
o3d.visualization.draw_geometries([pcd])
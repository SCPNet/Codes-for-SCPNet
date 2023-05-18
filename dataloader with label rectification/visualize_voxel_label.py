import os
import glob
import numpy as np
import yaml

import torch

import open3d as o3d


def get_cmap_semanticKITTI20():
  colors = np.array([
    [0  , 0  , 0,  255],
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255]]).astype(np.uint8)

  return colors
  

if __name__ == "__main__":
  
  dataset_root = "./labels"
  voxels_dir = glob.glob(os.path.join(dataset_root, "*.pt"))
  voxels_dir.sort()
  
  max_volume_space = [51.2, 25.6, 4.4]
  min_volume_space = [0, -25.6, -2.0]
  
  
  with open("./config/semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
    
  learning_map = semkittiyaml['learning_map'].copy()
  for key, value in learning_map.items():
    if key != 0 and value == 0:
        learning_map[key] = 255  
  
  color_map = get_cmap_semanticKITTI20()[:,:3]
  
  for voxel_path in voxels_dir:
    
    data = torch.load(voxel_path)
            
    x = np.arange(256)
    y = np.arange(256)
    z = np.arange(32)
    
    Y, X, Z = np.meshgrid(x,y,z)
    
    full_voxel_coord = np.concatenate((X[...,None], Y[...,None], Z[...,None]), axis=-1)
    
    voxel_label_org = data['voxel_label_org']
    voxel_label_rect = data['voxel_label_rect']
    
    mask = (voxel_label_org!=0) & (voxel_label_org!=255)
    full_voxel_coord_org = full_voxel_coord[mask]
    voxel_label_org = voxel_label_org[mask]

    mask = (voxel_label_rect!=0) & (voxel_label_rect!=255)
    full_voxel_coord_rect = full_voxel_coord[mask]
    voxel_label_rect = voxel_label_rect[mask]
    
    
    voxel_label_org_o3d = o3d.geometry.PointCloud()
    voxel_label_org_o3d.points = o3d.utility.Vector3dVector(full_voxel_coord_org)
    voxel_label_org_o3d.colors = o3d.utility.Vector3dVector(color_map[voxel_label_org.astype(int)].astype(float)/255)
    
    voxel_label_rect_o3d = o3d.geometry.PointCloud()
    voxel_label_rect_o3d.points = o3d.utility.Vector3dVector(full_voxel_coord_rect)
    voxel_label_rect_o3d.colors = o3d.utility.Vector3dVector(color_map[voxel_label_rect.astype(int)].astype(float)/255)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]) 
    
    o3d.visualization.draw_geometries([coord_frame, 
                                       voxel_label_org_o3d, 
                                       ], 
                                      zoom=0.1,
                                      front=[-np.sqrt(3)/2, 0, 1/2],
                                      lookat=[-50, 128, 178/np.sqrt(3)],
                                      up=[0, 0, 1],
                                      window_name ='voxel label original')
    
  
    print()
    
    o3d.visualization.draw_geometries([coord_frame, 
                                       voxel_label_rect_o3d, 
                                       ], 
                                      zoom=0.1,
                                      front=[-np.sqrt(3)/2, 0, 1/2],
                                      lookat=[-50, 128, 178/np.sqrt(3)],
                                      up=[0, 0, 1],
                                      window_name ='voxel label rectified')
    
    print()
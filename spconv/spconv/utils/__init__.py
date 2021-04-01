# Copyright 2019 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from spconv import spconv_utils
from spconv.spconv_utils import (non_max_suppression, non_max_suppression_cpu,
                                 points_to_voxel_3d_np, rbbox_iou, points_to_voxel_3d_np_expand2,
                                 rotate_non_max_suppression_cpu)

def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     coor_to_voxelidx,
                     max_points=35,
                     max_voxels=20000,
                     expand=None):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels) 
    with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = points_to_voxel_3d_np(
        points, voxels, coors, num_points_per_voxel, coor_to_voxelidx,
        voxel_size.tolist(), coors_range.tolist(), max_points, max_voxels)
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    if expand == '2':
        max_points_expand2 = 2 * max_points
        voxels_expand2 = np.zeros(
            shape=(max_voxels, max_points_expand2, points.shape[-1]), dtype=points.dtype)
        num_points_per_voxel_expand2 = np.zeros(shape=(max_voxels,), dtype=np.int32)
        coors_expand2 = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
        voxel_num_expand2 = points_to_voxel_3d_np_expand2(
            points, voxels_expand2, coors_expand2,
            num_points_per_voxel_expand2, coor_to_voxelidx, voxel_size.tolist(),
            coors_range.tolist(), max_points_expand2, max_voxels)
        voxels_expand2 = voxels_expand2[:voxel_num]
        num_points_per_voxel_expand2 = num_points_per_voxel_expand2[:voxel_num]
        coor_to_voxelidx[:voxel_num] = -1
        return voxels, voxels_expand2, coors, num_points_per_voxel, num_points_per_voxel_expand2
    else:
        coor_to_voxelidx[:voxel_num] = -1
        return voxels, coors, num_points_per_voxel

# def points_to_fv(rangeview, rv_pts_3d, theta, points, distances, pts_2d):
#     dtype = rangeview.dtype
#     rv_pts_3d = rv_pts_3d.astype(dtype)
#     theta = theta.astype(dtype)
#     points = points.astype(dtype)
#     distances = distances.astype(dtype)
#     num_features = points_to_fv_np(rangeview, rv_pts_3d, theta, points, distances, pts_2d)
#     return num_features

class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 expand=None):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self.expand = expand

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range, self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels, self.expand)
        return res 


    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
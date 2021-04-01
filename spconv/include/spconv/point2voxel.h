// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>

namespace spconv {
namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType>
int points_to_fv_np(py::array_t<DType> rangeview,
                    py::array_t<DType> rv_pts_3d,
                    py::array_t<DType> theta,
                    py::array_t<DType> points,
                    py::array_t<DType> distances,
                    py::array_t<int> pts_2d) {
  auto rangeview_rw = rangeview.template mutable_unchecked<3>();
  auto rv_pts_3d_rw = rv_pts_3d.template mutable_unchecked<3>();
  auto theta_rw = theta.template mutable_unchecked<1>();
  auto pts_2d_rw = pts_2d.mutable_unchecked<2>();
  auto points_rw = points.template mutable_unchecked<2>();
  auto distances_rw = distances.template mutable_unchecked<1>();
  auto N = pts_2d_rw.shape(0);
  auto num_features = rangeview.shape(0);
  auto rangeview_h = rangeview.shape(1);
  auto rangeview_w = rangeview.shape(2);
  auto dis = distances_rw(0);
  auto pt_2d_h = pts_2d_rw(0,1);
  auto pt_2d_w = pts_2d_rw(0,0);
  for (int i = 0; i < N; ++i) {
    dis = distances_rw(i);
    pt_2d_w = pts_2d_rw(i,0);
    pt_2d_h = pts_2d_rw(i,1);
    if (pt_2d_h<rangeview_h && pt_2d_w<rangeview_w) {
      if (dis>0) {
        if (rangeview_rw(0,pt_2d_h,pt_2d_w) == 0 || dis < rangeview_rw(0,pt_2d_h,pt_2d_w)) {
          rangeview_rw(0,pt_2d_h,pt_2d_w) = dis;
          rangeview_rw(1,pt_2d_h,pt_2d_w) = theta_rw(i);
          rangeview_rw(2,pt_2d_h,pt_2d_w) = points_rw(i,0);
          rangeview_rw(3,pt_2d_h,pt_2d_w) = points_rw(i,1);
          rangeview_rw(4,pt_2d_h,pt_2d_w) = points_rw(i,2);
          rangeview_rw(5,pt_2d_h,pt_2d_w) = points_rw(i,3);
          rv_pts_3d_rw(pt_2d_h,pt_2d_w,0) = points_rw(i,0);
          rv_pts_3d_rw(pt_2d_h,pt_2d_w,1) = points_rw(i,1);
          rv_pts_3d_rw(pt_2d_h,pt_2d_w,2) = points_rw(i,2);
        }
      }
    }
  }
  return dis;
}

template <typename DType, int NDim>
int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels)
        break;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  //for (int i = 0; i < voxel_num; ++i) {
  //  coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
  //}
  return voxel_num;
}

template <typename DType, int NDim>
int points_to_voxel_3d_np_expand2(py::array_t<DType> points, py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int coor_n[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  int voxelidn;
  for (int i = 0; i < N; ++i) {
    if (voxel_num >= max_voxels)
      break;
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    for (int ni = -1; ni < 2; ++ni){
      for (int nj = -1; nj < 2; ++nj){
        if ((coor[2]+ni<0 || coor[2]+ni>=grid_size[0]) || (coor[1]+nj<0 || coor[1]+nj>=grid_size[1]))
          continue;
        voxelidn = coor_to_voxelidx_rw(coor[0],coor[1]+nj,coor[2]+ni);
        if (voxelidn != -1){
          num = num_points_per_voxel_rw(voxelidn);
          if (num<max_points){
            for (int k=0; k<num_features; ++k){
              voxels_rw(voxelidn,num,k) = points_rw(i,k);
            }
            num_points_per_voxel_rw(voxelidn) += 1;
          }
          }
      }
    }
   }
  return voxel_num;
}
} // namespace spconv


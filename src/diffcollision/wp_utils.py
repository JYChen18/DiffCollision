import torch
import warp as wp

from diffcollision.utils import DCTensorSpec


@wp.kernel
def _sphere_distance_kernel_f32(
    T1_lst: wp.array(dtype=wp.mat44f),
    T2_lst: wp.array(dtype=wp.mat44f),
    sph1_lst: wp.array(dtype=wp.vec4f),
    sph2_lst: wp.array(dtype=wp.vec4f),
    T2s_idx1: wp.array(dtype=int),
    T2s_idx2: wp.array(dtype=int),
    n_T: int,
    n_sph: int,
    dist_max: wp.array(dtype=wp.float32),
    dist_min: wp.array(dtype=wp.float32),
):
    b, k = wp.tid()
    idx1 = T2s_idx1[k]
    idx2 = T2s_idx2[k]

    T1 = T1_lst[b * n_T + idx1]
    T2 = T2_lst[b * n_T + idx2]

    sph1 = sph1_lst[k]
    sph2 = sph2_lst[k]

    p1 = wp.transform_point(T1, sph1[:3])
    p2 = wp.transform_point(T2, sph2[:3])

    dist_max[b * n_sph + k] = wp.length(p1 - p2)
    dist_min[b * n_sph + k] = wp.length(p1 - p2) - sph1[-1] - sph2[-1]


@wp.kernel
def _sphere_distance_kernel_f64(
    T1_lst: wp.array(dtype=wp.mat44d),
    T2_lst: wp.array(dtype=wp.mat44d),
    sph1_lst: wp.array(dtype=wp.vec4d),
    sph2_lst: wp.array(dtype=wp.vec4d),
    T2s_idx1: wp.array(dtype=int),
    T2s_idx2: wp.array(dtype=int),
    n_T: int,
    n_sph: int,
    dist_max: wp.array(dtype=wp.float64),
    dist_min: wp.array(dtype=wp.float64),
):
    b, k = wp.tid()
    idx1 = T2s_idx1[k]
    idx2 = T2s_idx2[k]

    T1 = T1_lst[b * n_T + idx1]
    T2 = T2_lst[b * n_T + idx2]

    sph1 = sph1_lst[k]
    sph2 = sph2_lst[k]

    p1 = wp.transform_point(T1, sph1[:3])
    p2 = wp.transform_point(T2, sph2[:3])

    dist_max[b * n_sph + k] = wp.length(p1 - p2)
    dist_min[b * n_sph + k] = wp.length(p1 - p2) - sph1[-1] - sph2[-1]


@wp.kernel
def _sphere_distance_kernel_f16(
    T1_lst: wp.array(dtype=wp.mat44h),
    T2_lst: wp.array(dtype=wp.mat44h),
    sph1_lst: wp.array(dtype=wp.vec4h),
    sph2_lst: wp.array(dtype=wp.vec4h),
    T2s_idx1: wp.array(dtype=int),
    T2s_idx2: wp.array(dtype=int),
    n_T: int,
    n_sph: int,
    dist_max: wp.array(dtype=wp.float16),
    dist_min: wp.array(dtype=wp.float16),
):
    b, k = wp.tid()

    idx1 = T2s_idx1[k]
    idx2 = T2s_idx2[k]

    T1 = T1_lst[b * n_T + idx1]
    T2 = T2_lst[b * n_T + idx2]

    sph1 = sph1_lst[k]
    sph2 = sph2_lst[k]

    p1 = wp.transform_point(T1, sph1[:3])
    p2 = wp.transform_point(T2, sph2[:3])

    dist_max[b * n_sph + k] = wp.length(p1 - p2)
    dist_min[b * n_sph + k] = wp.length(p1 - p2) - sph1[-1] - sph2[-1]


def _torch2wp_type_map(dtype):
    if dtype == torch.float16:
        return wp.float16, wp.vec4h, wp.mat44h, _sphere_distance_kernel_f16
    elif dtype == torch.float32:
        return wp.float32, wp.vec4f, wp.mat44f, _sphere_distance_kernel_f32
    elif dtype == torch.float64:
        return wp.float64, wp.vec4d, wp.mat44d, _sphere_distance_kernel_f64
    else:
        raise TypeError(
            f"Unsupported dtype {dtype}. Only float16/float32/float64 are supported."
        )


class _WarpSphereDist:
    def __init__(self, ts: DCTensorSpec = DCTensorSpec()):
        wp.init()

        self.dtype_float, self.dtype_vec, self.dtype_mat, self.kernel_typed = (
            _torch2wp_type_map(ts.dtype)
        )
        self.ts = ts
        self.device = "cuda" if "cuda" in str(ts.device) else "cpu"
        return

    def forward(self, T1, T2, sph1, sph2, T2s_idx1, T2s_idx2):
        n_T, n_sph = T1.shape[-3], sph1.shape[-2]
        assert sph1.numel() == n_sph * 4 and sph2.numel() == n_sph * 4
        assert T2s_idx1.numel() == n_sph and T2s_idx2.numel() == n_sph
        dist_max = self.ts.to(torch.empty(T1.shape[:-3] + (n_sph,)))
        dist_min = self.ts.to(torch.empty(T1.shape[:-3] + (n_sph,)))
        wp.launch(
            kernel=self.kernel_typed,
            dim=(T1.shape[:-3].numel(), n_sph),
            inputs=[
                wp.from_torch(T1.reshape(-1, 4, 4), dtype=self.dtype_mat),
                wp.from_torch(T2.reshape(-1, 4, 4), dtype=self.dtype_mat),
                wp.from_torch(sph1.reshape(-1, 4), dtype=self.dtype_vec),
                wp.from_torch(sph2.reshape(-1, 4), dtype=self.dtype_vec),
                wp.from_torch(T2s_idx1.view(-1).to(torch.int32)),
                wp.from_torch(T2s_idx2.view(-1).to(torch.int32)),
                n_T,
                n_sph,
                wp.from_torch(dist_max.view(-1)),
                wp.from_torch(dist_min.view(-1)),
            ],
            device=self.device,
        )
        return dist_max, dist_min

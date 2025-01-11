"""
Convert params.npz output of dynamic or static gaussian splatting to .ply formate readable by tools such as SuperFacto
Jan 2025
"""

import numpy as np
from plyfile import PlyElement, PlyData
import os
import argparse
from tqdm import tqdm

def npz_to_ply(src, dest_folder, ply_file_prefix, timestep=0, static=False):
    
    def get_params(f_dc, scale, rotation):
        pos = ['x', 'y', 'z']
        normals = ['nx', 'ny', 'nz']
        return pos + normals + \
            [f'f_dc_{i}' for i in range(f_dc.shape[1])] + \
            ['opacity'] + \
            [f'scale_{i}' for i in range(scale.shape[1])] + \
            [f'rot_{i}' for i in range(rotation.shape[1])]

    params = np.load(src)
    
   
    xyz = params['means3D'][timestep] if not static else params['means3D'] 
    normals = np.zeros_like(xyz)
    f_dc = params['rgb_colors'][timestep] if not static else params['rgb_colors']
    opacities = params['logit_opacities']
    s = params['log_scales'].repeat(3, axis=-1)
    rot = params['unnorm_rotations'][timestep] if not static else params['unnorm_rotations']

    typ = 'f4'
    param_types = [(param, typ) for param in get_params(f_dc, s, rot)]
    splatting_data = np.empty(xyz.shape[0], dtype=param_types)

    print(f"{'Param':<10} {'Shape':<10}")
    print(f"{'xyz':<10} {xyz.shape}")
    print(f"{'normals':<10} {normals.shape}")
    print(f"{'f_dc':<10} {f_dc.shape}")
    print(f"{'opacities':<10} {opacities.shape}")
    print(f"{'scale':<10} {s.shape}")
    print(f"{'rotation':<10} {rot.shape}")

    params_ply = np.concatenate((xyz, normals, f_dc, opacities, s, rot), axis=1)
    splatting_data[:] = list(map(tuple, params_ply))
    
    ply_element = PlyElement.describe(splatting_data, 'vertex')

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest_file = os.path.join(dest_folder, f'{ply_file_prefix}_{timestep}.ply' if not static else f'{ply_file_prefix}.ply')
    PlyData([ply_element]).write(dest_file)
    print(f"Saved: {dest_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NPZ files to PLY format.')
    parser.add_argument('--npz', type=str, required=True, help='Path to the input NPZ file.')
    parser.add_argument('--ply', type=str, required=True, help='Prefix for the output PLY files.')
    parser.add_argument('--dest', type=str, required=True, help='Destination folder to save the PLY files.')
    parser.add_argument('--static', action='store_true', help='Flag to indicate the NPZ file has no time dimension.')
    args = parser.parse_args()

    params = np.load(args.npz)
    
    if args.static:
        npz_to_ply(args.npz, args.dest, args.ply, static=True)
    else:
        timesteps = params['means3D'].shape[0]
        for t in tqdm(range(timesteps), desc="Processing timesteps"):
            npz_to_ply(args.npz, args.dest, args.ply, timestep=t)
            
# Example Call
# Dynamic: python npz_to_ply.py --npz /path/to/npz/file.npz --ply splat.ply --dest /path/to/save/folder
# Static: python npz_to_ply.py --npz /path/to/npz/file.npz --ply splat.ply --dest /path/to/save/folder --static

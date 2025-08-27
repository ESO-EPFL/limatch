# %% Import Libraries
import argparse
import yaml
import torch
import time
import numpy as np


from lib import stats, icp
from lib.utils_LCD import getFeatures
from lib.tools import *
from lib.vis import visKpts, visMatchPts
from lib.filter import *
import multiprocessing as mp

from submodules.lcd.lcd import models

time0 = time.time()

parser = argparse.ArgumentParser(description='Point cloud matching pipeline')
parser.add_argument('--yml','-y', type=str, help='Path to yml configuration file')
parser.add_argument('--cloud1', '-c1', type=str, help='Path to the first point cloud')
parser.add_argument('--cloud2', '-c2', type=str, help='Path to the second point cloud')

args = parser.parse_args()

cfg = yaml.safe_load(open(args.yml))
cfg['tile_id'] = f'{args.cloud1.split("/")[-1].split(".")[0]}_{args.cloud2.split("/")[-1].split(".")[0]}'
createProjectFolder(cfg['prj_folder'])

print(f"Processing  {cfg['tile_id']} ...")
print('Visualization set to '+str(cfg['visualize']))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.serialization.add_safe_globals([np.dtype,np.core.multiarray.scalar,np.dtypes.Float32DType])

model = models.PointNetAutoencoder(256, 6, 6, True)

model.load_state_dict(torch.load(cfg["nn_path"], map_location=device, weights_only=True)["model"])
model = model.to(device)
time1 = time.time()

# %% ----------------- Step 00 - Preprocessing ------------------- %% #

print("Loading data...")
if cfg['cloud_fmt'] == 'txt':
    tile_a = tile.fromASCII(args.cloud1, cfg)
    tile_b = tile.fromASCII(args.cloud2, cfg)
elif cfg['cloud_fmt'] == 'las' or cfg['cloud_fmt'] == 'laz':
    tile_a = tile.fromLAS(args.cloud1, cfg)
    tile_b = tile.fromLAS(args.cloud2, cfg)

print("Preprocessing data...")
prepOverlap(tile_a, tile_b, cfg)

if cfg['vox_size'] > 0:
    print("Initial voxelization...")
    tile_a.voxTracing(cfg)
    tile_b.voxTracing(cfg)

if cfg['save_tiles']:
    print("Saving tile to csv...")
    np.savetxt(cfg['prj_folder'] + f"tiles/{cfg['tile_id']}_a.csv", np.concatenate([tile_a.rsc_id.reshape(-1,1),tile_a.xyz], axis=1), delimiter=',')
    np.savetxt(cfg['prj_folder'] + f"tiles/{cfg['tile_id']}_b.csv", np.concatenate([tile_b.rsc_id.reshape(-1,1),tile_b.xyz], axis=1), delimiter=',')

time2 = time.time()
# %% ----------------- Step 01 - Kpts detection ------------------ %% #

print("Keypoints estimation and tracking...")
kpts_a = issKpts(tile_a.xyz,cfg)
kpts_b = issKpts(tile_b.xyz,cfg)
_, tile_a.kpts_id = tile_a.kdt.query(np.asarray(kpts_a.points), workers=-1)
_, tile_b.kpts_id = tile_b.kdt.query(np.asarray(kpts_b.points), workers=-1)

if 'max_kpts' in cfg and cfg['max_kpts'] is not None:
    downSampleKpts(tile_a, cfg)
    downSampleKpts(tile_b, cfg)

cleanKpts(tile_a, tile_b, cfg)
cleanKpts(tile_b, tile_a, cfg)


if cfg['visualize']:
    visKpts(tile_a, tile_b, kpts_a, kpts_b)

del kpts_a, kpts_b

time3 = time.time()
# %% ----------------- Step 02 - Pts description ----------------- %% #
tile_a.feat = np.zeros((tile_a.kpts_id.shape[0], 256),dtype='float32')
tile_b.feat = np.zeros((tile_b.kpts_id.shape[0], 256),dtype='float32')

print("Description...")
tile_a.feat = getFeatures(tile_a, model, device, cfg)
tile_b.feat = getFeatures(tile_b, model, device, cfg)
print(f"\033[FDescription... Done")
time4 = time.time()
# %% ------------------ Step 03 - Pts matching ------------------- %% #
print("Point cloud matching...")
getCandidates(tile_a, tile_b, cfg)
getCandidates(tile_b, tile_a, cfg)

tile_a.cor_id,feat_dist_a, corr_xyz_a = featSearch(tile_a, tile_b)
tile_b.cor_id,feat_dist_b, corr_xyz_b = featSearch(tile_b, tile_a)

del tile_a.feat, tile_b.feat

corres = buildCorres(tile_a, tile_b, feat_dist_a)

if cfg['reciprocity_test']:
    print("Reciprocity test...", end=' ')
    reciprocal_mask = np.zeros(tile_a.kpts_id.shape[0],dtype=bool)
    def check_reciprocal(i):
        return tile_a.kpts_id[i] == tile_b.cor_id[np.where(tile_b.kpts_id == tile_a.cor_id[i])[0][0]]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        reciprocal_mask = pool.map(check_reciprocal, range(tile_a.kpts_id.shape[0]))
    reciprocal_mask = np.array(reciprocal_mask, dtype=bool)

    print(f"{100*np.sum(reciprocal_mask)/tile_a.kpts_id.shape[0]:.2f}% of correspondences kept")   
    corres = corres[reciprocal_mask]

del tile_a.candidates, tile_b.candidates

time5 = time.time()
# %% ------------------- Step 04.1 - RANSAC ---------------------- %% #
print("RANSAC filtering...")

corres_rsc_list = []  # collect results

for i in np.unique(tile_a.rsc_id):
    print(f"\033[FRansac filtering, tile {int(i)},", end=' ')
    corres_tile = corres[corres[:, 10] == i]
    idx_a_rsc = ransacFilter(corres_tile, cfg)

    if idx_a_rsc.shape[0] < 50:
        print(f"Warning, tile {int(i)} has less than 50 matches after RANSAC -> deleting...\n")
        continue  # skip tiles with uncertain ransac convergence

    corres_rsc_list.append(corres_tile[idx_a_rsc[:, 0], :])

corres_rsc = np.concatenate(corres_rsc_list, axis=0)
print("Ransac filtering... Done                       ")
time6 = time.time()

if cfg['visualize']:
    visMatchPts(tile_a.xyz, corres_rsc)
# %% -------------------- Step 04.2 - ICP ----------------------- %% #
print("ICP refinement...")
for i in np.unique(tile_a.rsc_id):
    print(f"\033[FICP refinement, tile {int(i)}...")
    corres_tile = corres_rsc[corres_rsc[:, 10] == i]
    icp.corrICP(tile_a, tile_b, corres_tile, cfg)
    if i == 1:
        corres_icp = corres_tile
    else:
        corres_icp = np.concatenate((corres_icp, corres_tile), axis=0)

icp_vec = -corres_icp[:, -3:]
print("\033[FICP refinement... Done    ")
time7 = time.time()
# %% ----------------- Step 05 - Save output, Compute Stats & Visualization ----------------- %% #

print("Building correspondences file ...")
buildCorresFile(corres_rsc, tile_a, tile_b, cfg, icp_vec)

if cfg['save_stats']:
    print("Building stats & plots...")
    stats_raw = stats.compute_stats(corres, cfg['tile_id'])
    stats_rsc = stats.compute_stats(corres_rsc, cfg['tile_id'])
    stats_icp = stats.compute_stats(corres_icp, cfg['tile_id'])

    stats.plot_stats(stats_raw, cfg['prj_folder'] + "plots/", f'raw_{cfg["tile_id"]}')
    stats.plot_stats(stats_rsc, cfg['prj_folder'] + "plots/", f'rsc_{cfg["tile_id"]}')
    stats.plot_stats(stats_icp, cfg['prj_folder'] + "plots/", f'icp_{cfg["tile_id"]}')

    stats.plot_final(stats_raw, stats_rsc, stats_icp, cfg)

if cfg['visualize']:
    visMatchPts(tile_a.xyz, corres_icp)

print(f"Per step time:\n\
    Setup: {time1-time0:.2f}s\n\
    Preprocessing: {time2-time1:.2f}s\n\
    Detection: {time3-time2:.2f}s\n\
    Description: {time4-time3:.2f}s\n\
    Matching: {time5-time4:.2f}s\n\
    RANSAC: {time6-time5:.2f}s\n\
    ICP: {time7-time6:.2f}s\n\
    Total: {time7-time0:.2f}s")

import numpy as np
import math
import matplotlib.pyplot as plt

def compute_stats(corres, tile_num, inlier_thr=np.linspace(0, 1, 50)):
    """
    Build stats on correspondences. Only valid on groundtruth aligned point clouds.
    """
    corres_num = corres.shape[0]

    stats = {}
    stats["Tile number"] = tile_num
    stats["Correspondences"] = corres
    stats["Number corr"] = corres.shape[0]
    stats["Distances"] = corres[:, 2]

    inlier_ratios = np.empty((inlier_thr.shape))

    for i in range(inlier_thr.shape[0]):
        match = np.sum(corres[:, 2] < inlier_thr[i])

        inlier_ratios[i] = match/corres_num

    stats["Inlier thresholds"] = inlier_thr
    stats["Inlier ratios"] = inlier_ratios

    error_vec = corres[:, 7:10]-corres[:, 4:7]
    stats["Error vectors"] = error_vec
    error_vec = error_vec/np.linalg.norm(error_vec, axis=1).reshape(-1, 1)
    stats["Error angles"] = np.degrees(np.arctan2(error_vec[:, 0], error_vec[:, 1])) % 360


    return stats


def plot_stats(stats, path, tile_num):
    """
    Build plots on correspondences. Only valid on groundtruth aligned point clouds

    """
    corr_num = stats["Number corr"]

    # Compute average misalignment vector
    avg_vector = np.mean(stats["Error vectors"], axis=0)
    avg_vector_str = f"Avg misalign: [{avg_vector[0]:.3f}, {avg_vector[1]:.3f}, {avg_vector[2]:.3f}] m"

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Misalignment statistics\n{tile_num}\n Correspondences n°: {corr_num}\n{avg_vector_str}", fontsize=14)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_title(f"3D distribution", fontsize=10)
    ax2.set_title(f"3D Norm", fontsize=10)
    ax3.set_title(f"Heading [°]", fontsize=10, loc='left')
    ax4.set_title(f"Cumulative distribution function ", fontsize=10)

    ax1.set_xlabel("East [m]"), ax1.set_ylabel("North [m]"), ax1.set_zlabel("Up [m]")
    ax2.set_xlabel("Misalignment norm [m]"), ax2.set_ylabel("Amount [-]")
    ax4.set_xlabel("Misalignment [m]"), ax4.set_ylabel("Probability [%]")

    ax1.set_xlim([-1, 1]), ax1.set_ylim([-1, 1]), ax1.set_zlim([-1, 1])
    ax2.set_xlim(0, np.max([1, np.quantile(stats["Distances"], 0.99)]))
    ax4.set_xlim(0, 1)

    ax1.scatter(stats["Error vectors"][:, 0], stats["Error vectors"][:, 1],
                stats["Error vectors"][:, 2], marker='.',
                c=np.log(stats["Distances"]), s=0.5)
    ax1.scatter(0, 0, 0, marker='o', s=2, c='r')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax2.grid(color='grey', linestyle='--', which='both')
    ax2.hist(stats["Distances"], bins=25, facecolor='#007480', alpha=0.75, edgecolor='black', linewidth=0.2)

    num_bins = 64
    bins = np.linspace(0, 360, num_bins + 1)
    counts, _ = np.histogram(stats["Error angles"], bins=bins)
    bin_centers_deg = (bins[:-1] + bins[1:]) / 2
    bin_centers_rad = np.radians(bin_centers_deg)
    bars = ax3.bar(bin_centers_rad, counts, width=np.radians(360 / num_bins), bottom=0.0,
                   align='center', edgecolor=(0, 0, 0, 0.1))
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_xticks(np.radians(np.linspace(0, 315, 8)))
    ax3.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax3.set_yticklabels([])
    for bar in bars:
        bar.set_facecolor(plt.cm.viridis(bar.get_height() / max(counts) if max(counts) > 0 else 1))
        bar.set_alpha(0.7)

    ax4.grid(color='grey', linestyle='--', which='both')
    ax4.plot(stats["Inlier thresholds"], stats["Inlier ratios"]*100, color='#007480')

    plt.savefig(path + f"/{tile_num}_stats.svg", dpi=600)


def plot_final(stats_raw, stats_rsc, stats_icp, cfg):
    n_raw = stats_raw["Number corr"]
    n_rs = stats_rsc["Number corr"]

    fig1 = plt.figure()
    fig1.suptitle(f"L2L error cumulative distribution function")
    ax1 = fig1.add_subplot(111)

    ax1.grid(color='grey', linestyle='--')
    ax1.set_title(f"Tile {cfg['tile_id']} | Cor number: raw {n_raw}, ransac & icp {n_rs}", fontsize=8)
    ax1.set_ylabel("CDF [%]")
    ax1.set_xlim(0, 1)
    ax1.plot(stats_raw["Inlier thresholds"], stats_raw["Inlier ratios"]*100, color='orangered', label=f'Raw, RMSE={np.mean(stats_raw["Distances"]):.3f}')
    ax1.plot(stats_rsc["Inlier thresholds"], stats_rsc["Inlier ratios"]*100, color='hotpink', label=f'Ransac, RMSE={np.mean(stats_rsc["Distances"]):.3f}')
    ax1.plot(stats_icp["Inlier thresholds"], stats_icp["Inlier ratios"]*100, color='deepskyblue', label=f'ICP point, RMSE={np.mean(stats_icp["Distances"]):.3f}')

    ax1.legend()
    plt.savefig(cfg['prj_folder'] + "/plots/glob_" + f"{cfg['tile_id']}_inlier_threshold.png", dpi=600)

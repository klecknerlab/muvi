import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def plot_2D(R, LCS, out_dir):
    """
    Plots the forward and backward FTLE field and saves the plots in a 'plots' directory.

    Parameters:
    - R: Regular grid.
    - LCS: List of FTLE fields (forward and backward).
    - out_dir: Name of output directory to store plots.
    """
     
    print(f"Plotting forward and backward FTLE field...")

    os.makedirs(out_dir, exist_ok=True)

    ofn = ["fwd_ftle", "bwd_ftle"]
    xx, yy = R[..., 0], R[..., 1]
    for i in range(len(ofn)):
        fig = plt.figure()  
        ax = fig.add_subplot(111)  
        pcolor_plot = ax.pcolor(xx, yy, LCS[i], cmap='cool', clim=(0, 0.5))

        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_rasterization_zorder(1)

        cbar = plt.colorbar(pcolor_plot, ax=ax, fraction=0.03, pad=0.05)

        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        plt.savefig(os.path.join(out_dir, ofn[i] + ".png"), dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_3D(R, LCS, out_dir):
    x = R[..., 0].flatten()
    y = R[..., 1].flatten()
    z = R[..., 2].flatten()
    
    os.makedirs(out_dir, exist_ok=True)

    ofn = ["fwd_ftle", "bwd_ftle"]
    for i in range(len(LCS)):
        c = LCS[i].flatten()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot
        sc = ax.scatter(x, y, z, c=c, cmap='inferno', marker='o')

        # Add color bar
        plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

        # Labels and title
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.savefig(os.path.join(out_dir, ofn[i] + ".png"), dpi=600)
        plt.close()

def plot_slices(lcs_path, ftle):

    data = np.load(lcs_path)

    print(f"data shape: {data.shape}")
    print(f"Plotting 2D slices...")
    # Get the base directory
    base_dir = os.path.dirname(lcs_path)  # This gives you 'C:/Users/dtapi/github/muvi/muvi/lcs/abc_flow'

    # Create the 'fwd' directory under 'abc_flow'
    if ftle == 'fwd':
        out_dir = os.path.join(base_dir, "fwd")
    elif ftle =='bwd':
        out_dir = os.path.join(base_dir, "bwd")
    else:
        raise ValueError(f"select 'fwd' or 'bwd' for ftle.")

    # Define the output directories for each axis
    output_dirs = {
        "z": os.path.join(out_dir, "Nz"),
        "y": os.path.join(out_dir, "Ny"),
        "x": os.path.join(out_dir, "Nx")
    }

    # Define the axis labels (same as output_dirs in this case)
    plt_labels = output_dirs

    # Create the subdirectories if they don't exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Plot each slice along the first axis and save the images
    for i in range(data.shape[0]):
        plt.figure()
        plt.imshow(data[i, :, :], aspect='auto')
        plt.colorbar()
        plt.title(f'Slice {i}')
        plt.savefig(os.path.join(output_dirs["z"], f"{i:06d}.png"))
        plt.close()

    # Plot each slice along the second axis and save the images
    for i in range(data.shape[1]):
        plt.figure()
        plt.imshow(data[:, i, :], aspect='auto')
        plt.colorbar()
        plt.title(f'Slice {i}')
        plt.savefig(os.path.join(output_dirs["y"], f"{i:06d}.png"))
        plt.close()

    # Plot each slice along the third axis and save the images
    for i in range(data.shape[2]):
        plt.figure()
        plt.imshow(data[:, :, i], aspect='auto')
        plt.colorbar()
        plt.title(f'Slice {i}')
        plt.savefig(os.path.join(output_dirs["x"], f"{i:06d}.png"))
        plt.close()

    print(f"Plotting complete.")
import numpy as np
from muvi.lcs import FTLEs
from muvi.geometry.volume import rectilinear_grid
from muvi.lcs.flow_models import abc_flow
from muvi.lcs.visualization import plot_3D
import time 
import os

T = 8
dt = 1/10
num_particles = 500

#Initialize randomly spaced particles within our domain of interest
X0 = np.random.rand(num_particles, 3) * 2*np.pi
Nx, Ny, Nz = 128, 128, 128
x,y,z = np.linspace(0, 2*np.pi, Nx), np.linspace(0, 2*np.pi, Ny), np.linspace(0, 2*np.pi, Nz)
R = rectilinear_grid(x,y,z) #see muvi.geoemetry.volume for more info
interpolator = 'rbf'
interp_params = dict(kernel='cubic', smoothing=2, neighbors=20)
method = 'FMC'
save_lcs = True

def main():
    start_time = time.time()
    LCS = compute_ftle(T, dt, X0, R, save_lcs)
    plot_3D(R, LCS, out_dir = 'abc_' + method)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    print(f"Computing time: {elapsed_time_seconds:.2f} s, "
          f"{elapsed_time_seconds / 60:.2f} min, "
          f"{elapsed_time_seconds / 3600:.2f} hr.")

def compute_ftle(T, dt, X0, R, save_lcs = False):
    """
    Computes the forward and backward FTLE field.

    Parameters:
    - T: Final time.
    - dt: Time step.
    - X0: Coordinates of randomly spaced particles.
    - R: Regular grid to interpolate on.
    - interpolator: Interpolating function.
    - method: velocity field integration 'VFI' or flow map compiliation 'FMC'
    - interp_params: Interpolating parameters.
    """
    t = np.arange(0, T + dt, dt)

    if method == 'VFI':
        #Appending to t guarantees that flow map will be computed from 0 to T (VFI method uses RK4)
        t = np.append(t, np.array([T+dt, T+2*dt]))
        traj = FTLEs().RK4(X0, abc_flow, t, dt)

        vel = np.zeros_like(traj)
        for i, time_step in enumerate(t):
            vel[i, ...] = abc_flow(traj[i, ...], time_step)

        F = FTLEs().VFI(R, dt, traj, vel, interpolator, **interp_params)
    
    elif method == 'FMC':
        traj = FTLEs().RK4(X0, abc_flow, t, dt)
        F = FTLEs().FMC(R=R, traj=traj, interpolator=interpolator, **interp_params)
   
    else:
        raise ValueError(f"Select method 'VFI' or 'FMC'.")
    
    G = FTLEs().grad_flowmap(F)
    
    LCS = FTLEs().ftle_field(G, T)
    
    if save_lcs:
        out_dir = 'abc_' + method
        os.makedirs(out_dir, exist_ok=True)

        np_fn = ['fwd_ftle', 'bwd_ftle']

        for i in range(len(LCS)):
            file_path = os.path.join(out_dir, np_fn[i])
            np.save(file_path, LCS[i])
            print(f"Saved: {file_path}")

    return LCS
    

if __name__ == "__main__":
    main()
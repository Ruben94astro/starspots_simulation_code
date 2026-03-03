import numpy as np
import astropy.units as u
import pandas as pd
import tqdm
from concurrent.futures import ProcessPoolExecutor
from create_sphere import add_spots, fibonacci_sphere, cartesian_from_spherical, quadratic, spot_mask_geodesic
from parameters import n_points, elev, azim, r_val, u1, u2, total_frames, spots, observing_baseline_days, cadence_time
import matplotlib.pyplot as plt
import os
import glob
from multiprocessing import Pool
from PIL import Image
from sklearn.preprocessing import normalize
from scipy.interpolate import interp1d


def compute_flux(filename):
    img = Image.open(filename).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return np.sum(img_array)



def run_parallel_frames(points, base_intensity, elev, azim,
                        total_frames, vmin, vmax, spots, r_val, cadence_time,
                        n_workers=198):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(animate, i, points, base_intensity,
                            elev, azim, total_frames, vmin, vmax,
                            spots, r_val, cadence_time)
            for i in range(total_frames)

        ]
        for f in tqdm.tqdm(futures):
            f.result()


def animate(i, points, base_intensity, elev, azim,
                  total_frames, vmin, vmax, spots, r_val, cadence_time):
    """ function to animate spots"""
            # background configurations
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax_sphere = fig.add_subplot(111, projection='3d') 
    ax_sphere.set_axis_off()
    ax_sphere.set_box_aspect([1, 1, 1])
    ax_sphere.view_init(elev=elev, azim=azim)
    

    intensity = np.copy(base_intensity)

    for spot in spots:
        theta_mov = spot['theta'] + spot['angular_velocity'] *i * cadence_time.to(u.day)
        spot_x = r_val * np.sin(spot['phi']) * np.cos(theta_mov.value)
        spot_y = r_val * np.sin(spot['phi']) * np.sin(theta_mov.value)
        spot_z = r_val * np.cos(spot['phi'])
        spot_center = np.array([spot_x, spot_y, spot_z])

        mask = spot_mask_geodesic(points[:,0], points[:,1], points[:,2],
                                  spot_center, spot['radius'])
        contrast = 0.138   # c_TESS calculado físicamente
        intensity *= (1 - mask + contrast * mask)

    ax_sphere.scatter(points[:,0], points[:,1], points[:,2],
                      c=np.clip(intensity,0,1), cmap='gray', s=1,
                      vmin=vmin, vmax=vmax)




    max_range = r_val * 1.1
    ax_sphere.set_xlim(-max_range, max_range)
    ax_sphere.set_ylim(-max_range, max_range)
    ax_sphere.set_zlim(-max_range, max_range)

    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/frame_{i:05d}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def star_animate(theta_vec):
    
    #lat,lon,radii = theta_vec
    lat,lon,radii = theta_vec
    add_spots(lat, lon, radii)
        # spherical grid with fibonacci points
    print("Generate spherical grid")
    phi, theta = fibonacci_sphere(n_points)
    x, y, z = cartesian_from_spherical(phi, theta)
    points = np.vstack([x, y, z]).T
    
    # Calculate point of view
    elev_rad = np.deg2rad(elev) #elevation of point of view
    azim_rad = np.deg2rad(azim)#azimut of point of view
    
    v_x = np.cos(elev_rad) * np.cos(azim_rad)
    v_y = np.cos(elev_rad) * np.sin(azim_rad)
    v_z = np.sin(elev_rad)
    
    # rearrange of calculating mu parameter for limb darkening
    mu = (points[:, 0] * v_x + points[:, 1] * v_y + points[:, 2] * v_z) / r_val
    mu = np.clip(mu, 0, 1)
    #base_intensity = limbdarkening(constant, mu)# applying to the texture
    base_intensity = quadratic(u1,u2,mu) 
        #    Calculates the extreme values of the base intensity: 
    #vmin: Minimum value of intensity in the whole star.
    #vmax: Maximum value of intensity over the whole star.
    vmin=  0.0
    vmax=  1.0

   # Defines a reference range for color mapping that will be used consistently across all frames.
            # background configurations
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax_sphere = fig.add_subplot(111, projection='3d') 
    ax_sphere.set_axis_off()
    ax_sphere.set_box_aspect([1, 1, 1])
    
 
    

    print("start render...")

    run_parallel_frames(points, base_intensity, elev, azim,
                    total_frames, vmin, vmax, spots, r_val, cadence_time,
                    n_workers=198) 
    
    plt.style.use('default')
    spots.clear()
    plt.close(fig) 
 
    return flux_plot(theta_vec)


def flux_plot(theta_vec):
    '''
    Function take a list normalizing the flux, converting the list in a 
csv file and rename the columns
    and return a plot 

    '''
    #lat,lon,radii = theta_vec
    lat, lon, radii = theta_vec
    frame_files = sorted(glob.glob("frames/frame_*.png"))
    fluxes = []

    with Pool() as pool:
        fluxes = pool.map(compute_flux, frame_files)

    # Normalized fluxes
    #flux_norm.append(flux_total / fluxes[i])
    flux_norm = normalize([fluxes], norm="max")[0]
    #flux_norm = np.array(fluxes)/np.max(np.array(fluxes))
    df = pd.DataFrame({'flux_normalized': flux_norm})
    df.index.name = 'Frame'
    df.reset_index(inplace=True)
    df['Days'] = df['Frame'] * (cadence_time.to(u.day)).value
    df = df[['Days', 'flux_normalized']]
    
    # Interpolación a 2 minutos
    step_2min = 30 / (24 * 60)
    
    new_days = np.arange(df['Days'].min(), df['Days'].max() , step_2min)
    new_days = new_days[new_days <= df['Days'].max()]
    
    f_interp = interp1d(df['Days'], df['flux_normalized'], kind='cubic', fill_value='extrapolate')
    new_flux = f_interp(new_days)
    
    # Guardar
    result = pd.DataFrame({'Days': new_days, 'flux_normalized': new_flux})
    result.to_csv(f'lat_{lat}_lon_{lon}_radii{radii}_.csv', index=False)


    
    return result["flux_normalized"]

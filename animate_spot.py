import astropy.units as u
from mcmc_code import load_tess, function_mse, main, lnprob, plot_simple_traces, print_detailed_results
import os 
import numpy as np
from parameters import observing_baseline_days,cadence_time,total_frames

import emcee


os.makedirs("frames", exist_ok=True)

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

    





if __name__ == '__main__':
    # stellar parameterq  
    
    # base lines time parameter
    #observing_baseline_days = 5 * u.day
    #cadence_time = 60 * u.minute
    #total_frames = int((observing_baseline_days / cadence_time).decompose().value)
    
    #load flux and days of test light curve
    flux_tess, days_tess,flux_tess_error = load_tess()
    print(f"data loaded: {len(flux_tess)} ")
    #parameters where we are selecting the top1 simulated lightcurve by MSE
    initial_params = function_mse(flux_tess, days_tess)
    print(f"Initial Parameters: {initial_params}")
   
    true_params = np.array([16.81, 176.05, 12.17])  # Your true values
    labels = ['lat', 'lon','radii']
    

  
    
    data = (flux_tess,flux_tess_error)
    nwalkers =15
    niter = 200
    initial = np.array(initial_params)
    
    #ndim = len(initial)
    ndim = len(initial)
    scales = np.array([0.5, 0.5, 0.4])  # 0.5° en lat/lon, 0.1 en radio
    #p0 = np.array([initial + 2*np.random.randn(ndim) for i in range(nwalkers)])[:, None]
    p0 = [np.array(initial) +  scales * np.random.randn(ndim) for _ in range(nwalkers)]
    filename = "backend_mcmc.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
        
    
    #sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
    sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data, initial, backend)
    
        # Obtener los mejores parámetros
    samples = sampler.flatchain
    best_params = samples[np.argmax(sampler.flatlnprobability)]
    plot_simple_traces(samples,labels) 
    print_detailed_results(true_params, best_params, sampler, data)


 

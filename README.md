This repository contains the list of codes for simulating starspots.

requeriments.txt will show you the packages need it for running

1) In parameters.py, configure the star value, such as n_points, which refers to the resolution of the sphere; the limb darkening coefficients
 u1 and u2, which can be estimated using atmospheric models (Phoenix, etc.); the rotation period; the inclination angle (elev); the observation time; and the cadence between each observation.  
2) The light curve you want to analyze must be named tess_curve.csv, and the dimensions must be appropriate, for example total_frames = cadence_time x observing_line_days 
warning(you have to delete in star_animate.py   # Interpolation at 2 minutes
    step_2min = 30 / (24 * 60)
    
    new_days = np.arange(df[‘Days’].min(), df[‘Days’].max() , step_2min)
    new_days = new_days[new_days <= df[‘Days’].max()]
    
    f_interp = interp1d(df[‘Days’], df[‘flux_normalized’], kind=‘cubic’, fill_value=‘extrapolate’)
    new_flux = f_interp(new_days)) this line of code because it works for interpolating frames in order to save processing time, so if
   you want to interpolate 2 minutes froms 60 minutes dont modify otherwise delete it


3)to star animate please run animate_spot.py, this pipeline contains the animatios setup from the parameters, also in this file you will manage the numbers of walkers and iteration from mcmc file,
also will save the progress using a backend_mcmc.h5 if you lose progress

brief description about the files
-----------animate_spot.py------------- 
in this document that I mentioned before you can manage the numbers of walkers and iteration of mcmc also the save file in .h5 and the scale of exploration


----------create_sphere,py--------------
this file contains the instruction to draw the fibonacci sphere also the mask of the spot and differential rotation if need it, the equation of limb dakening
linear and quadratic and a function call gif if you want to animate te spot


----------flux_plot.py----------------
This file contains function to calculate the flux of the image array taken from the frames, returnr a flux plot normalized


-----------mcmc_code.py------------
This is the heart of the MCMC algorithm here you can found how mcmc works, also there is function like functcion_mse that calulate the mean square 
error from the simulation folder, and load_tess who called the function of tess_curve.csv 

---------star_animate.py----------
this file contains instructions to about how to generate each frame and use paralelization to improve the running time 
Warning if you are not workin in a cluster you need to modify n_worker to the number of your core computer this is inside fo run_parallel_frames function





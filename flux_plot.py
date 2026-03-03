
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


    #------pruebas
def compute_flux(filename):
    img = Image.open(filename).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return np.sum(img_array)

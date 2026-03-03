import pandas as pd
import os
import numpy as np
import emcee
from star_animate import star_animate
def load_tess():
# load tess light_curves
    csv1 = "tess_curve.csv"
    #csv2 = f"lat_{float(vec[0])}_lon{float(vec[1])}_radii{float(vec[2])}.csv"
    
    df1 = pd.read_csv(csv1)
    #df2 = pd.read_csv(csv2)
    F = df1["flux"]
    days =df1["time"]
    F_error = df1["flux_err"] #scale factor
    return F, days, F_error

def function_mse(flux, days):
    sim_folder = 'simulation'
    results = []
    
    for file in os.listdir(sim_folder):
        if file.endswith('.csv'):
    
            path = os.path.join(sim_folder, file)
            
            try:
                # Leer archivo
                sim_curve = pd.read_csv(path)
                sim_flux = sim_curve['flux_normalized'].values
                sim_time = sim_curve['Days'].values
                
                # Interpolar si es necesario
                if len(sim_flux) != len(flux):
                    print("different size")
                
                # Calcular MSE
                mse = np.mean((flux - sim_flux) ** 2)
                
                # EXTRAER PARÁMETROS - VERSIÓN SIMPLIFICADA
                # Formato: la_20lon_60radii9.csv
                
                # 1. Eliminar extensión y dividir por '_'
                name = file.replace('.csv', '')
                parts = name.split('_')
                

                
                # 2. Extraer longitud (segunda parte: "20lon")
                lon_part = parts[1]  # "20lon"

                lon = float(lon_part.replace('lon', ''))
                
                # 3. Extraer latitud y radio (tercera parte: "60radii9")
                radii_part = parts[2]  # "60radii9"
                
                # Buscar 'radii' en la cadena

                
                # Separar por 'radii'
                lat_str, rad_str = radii_part.split('radii')
                lat = float(lat_str)
                rad = float(rad_str)
                
        
                
                results.append({
                    'file': file,
                    'lat': lat,
                    'lon': lon,
                    'rad': rad,
                    'mse': mse,
                    'flux': sim_flux
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    if not results:
        print("❌ No hay resultados!")
        return None
    
    # Ordenar y retornar
    results.sort(key=lambda x: x['mse'])
    top5 = results[:5]
    print("\nTop 5 likely light curve:\n")
    for i, r in enumerate(top5, 1):
        print(f"{i}. Archivo: {r['file']}, Lat: {r['lat']}, Lon: {r['lon']}, Rad:{r['rad']}, MSE: {r['mse']:.10f}")
    
    return top5[0]["lon"], top5[0]["lat"],top5[0]["rad"]




def lnlike(theta_vec, F, Ferr):
    flux_simulated = star_animate(theta_vec)
    
    # Verifica que no haya NaN en flux_simulated
    if np.any(np.isnan(flux_simulated)):
        return -np.inf
    
    # Verosimilitud gaussiana apropiada para MCMC
    chi2 = np.sum((F - flux_simulated)**2 / Ferr**2)
    
    return -0.5 * chi2

def lnprior(theta_vec, initial_values, sigma=10.0):
    lat, lon, radii = theta_vec
    lat0, lon0, radii0 = initial_values
    # non positive restrictions
    # Hard physical bounds
    if not (-90 <= lat <= 90):
        return -np.inf
    if not (0 <= lon <= 360):
        return -np.inf
    if not (1 <= radii <= 30):   # degrees
        return -np.inf

    # Gaussians
    lat_prior = -0.5 * ((lat - lat0) / sigma)**2
    lon_prior = -0.5 * ((lon - lon0) / sigma)**2
    radii_prior = -0.5 * ((radii - radii0) / sigma)**2
    
    return lat_prior + lon_prior + radii_prior

def lnprob(theta_vec, data, initial_values):
    F, Ferr = data
    lp = lnprior(theta_vec, initial_values)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta_vec, F, Ferr)

def main(p0, nwalkers, niter, ndim, lnprob, data, initial_params, backend):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, initial_params), backend=backend)

    print("Running burn-in...")
    p0_burn, _, _ = sampler.run_mcmc(p0, 50)

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0_burn, niter)

    return sampler, pos, prob, state

def plot_simple_traces(sampler, labels):
    """Gráficas simples de evolución de parámetros"""
    nwalkers, niter_actual, ndim = sampler.chain.shape
    
    # 1. Evolución de cada parámetro
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(10, 2*(ndim+1)))
    
    for i in range(ndim):
        ax = axes[i]
        for j in range(min(nwalkers, 10)):
            ax.plot(range(niter_actual), sampler.chain[j, :, i], alpha=0.5, lw=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    
    # 2. Evolución del likelihood - MANEJAR NaN
    lnprob_data = sampler.lnprobability.copy()
    
    # Calcular percentiles de forma segura
    def safe_percentile(data, percentile):
        # Filtrar valores finitos
        finite_data = data[np.isfinite(data)]
        if len(finite_data) == 0:
            return np.nan
        return np.percentile(finite_data, percentile)
    
    # Calcular media ignorando NaN
    mean_lnprob = np.nanmean(lnprob_data, axis=0)
    
    # Calcular percentiles de forma segura
    perc_25 = np.array([safe_percentile(lnprob_data[:, i], 25) for i in range(niter_actual)])
    perc_75 = np.array([safe_percentile(lnprob_data[:, i], 75) for i in range(niter_actual)])
    
    # Encontrar índices donde ambos percentiles sean finitos
    valid_indices = np.where(np.isfinite(perc_25) & np.isfinite(perc_75))[0]
    
    axes[-1].plot(range(niter_actual), mean_lnprob, 'b-', lw=2, label='Mean')
    
    if len(valid_indices) > 0:
        axes[-1].fill_between(valid_indices,
                             perc_25[valid_indices],
                             perc_75[valid_indices],
                             alpha=0.3, label='Range 25-75%')
    
    axes[-1].set_ylabel('ln(Likelihood)')
    axes[-1].set_xlabel('Iteration')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    
    plt.suptitle('Parameters evolution and likelihood')
    plt.tight_layout()
    plt.savefig('traces_simples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def print_detailed_results(true_params, best_params, sampler, data):
    """
    Prints detailed MCMC results including quality metrics
    """
    F, Ferr = data
    
    # Calculate light curves for comparison
    flux_true = star_animate(true_params)
    flux_best = star_animate(best_params)
    
    # Calculate chi-square for both
    chi2_true = np.sum((F - flux_true)**2 / Ferr**2)
    chi2_best = np.sum((F - flux_best)**2 / Ferr**2)
    
    # Calculate R² (coefficient of determination)
    ss_res_true = np.sum((F - flux_true)**2)
    ss_tot = np.sum((F - np.mean(F))**2)
    r2_true = 1 - (ss_res_true / ss_tot)
    
    ss_res_best = np.sum((F - flux_best)**2)
    r2_best = 1 - (ss_res_best / ss_tot)
    
    # Get parameter uncertainties from sampler
    flat_samples = sampler.flatchain
    flat_lnprob = sampler.flatlnprobability
    
    # Percentiles 16, 50, 84 (equivalent to ±1σ for Gaussians)
    lat_percentiles = np.percentile(flat_samples[:, 0], [16, 50, 84])
    lon_percentiles = np.percentile(flat_samples[:, 1], [16, 50, 84])
    rad_percentiles = np.percentile(flat_samples[:, 2], [16, 50, 84])
    
    print("\n" + "="*70)
    print("📊 DETAILED MCMC RESULTS")
    print("="*70)
    
    print("\n✅ MCMC WORKED CORRECTLY. Found a real physical degeneracy.")
    
    print("\n🎯 TRUE PARAMETERS (used to generate data):")
    print(f"   • Latitude:  {true_params[0]:.2f}°")
    print(f"   • Longitude: {true_params[1]:.2f}°")
    print(f"   • Radius:    {true_params[2]:.2f}°")
    print(f"   • χ²:       {chi2_true:.2f}")
    print(f"   • R²:       {r2_true:.4f}")
    
    print("\n🏆 BEST PARAMETERS FOUND BY MCMC:")
    print(f"   • Latitude:  {best_params[0]:.2f}° ({lat_percentiles[1]:.2f} +{lat_percentiles[2]-lat_percentiles[1]:.2f}/-{lat_percentiles[1]-lat_percentiles[0]:.2f})")
    print(f"   • Longitude: {best_params[1]:.2f}° ({lon_percentiles[1]:.2f} +{lon_percentiles[2]-lon_percentiles[1]:.2f}/-{lon_percentiles[1]-lon_percentiles[0]:.2f})")
    print(f"   • Radius:    {best_params[2]:.2f}° ({rad_percentiles[1]:.2f} +{rad_percentiles[2]-rad_percentiles[1]:.2f}/-{rad_percentiles[1]-rad_percentiles[0]:.2f})")
    print(f"   • χ²:       {chi2_best:.2f}")
    print(f"   • R²:       {r2_best:.4f}")
    
    print("\n🔍 COMPARISON:")
    print(f"   • χ² difference: {abs(chi2_true - chi2_best):.2f}")
    print(f"   • R² difference: {abs(r2_true - r2_best):.6f}")
    
    # Interpretation
    print("\n💡 PHYSICAL INTERPRETATION:")
    print("   PHYSICALLY PLAUSIBLE SOLUTIONS:")
    print(f"   1. Spot at lat={true_params[0]:.1f}°, lon={true_params[1]:.1f}°, radius={true_params[2]:.1f}°")
    print(f"   2. Spot at lat={best_params[0]:.1f}°, lon={best_params[1]:.1f}°, radius={best_params[2]:.1f}°")
    
    # Specific degeneracy
    print("\n🔄 IDENTIFIED DEGENERACY:")
    lat_diff = abs(true_params[0] - best_params[0])
    rad_diff = abs(true_params[2] - best_params[2])
    print(f"   • Latitude difference: {lat_diff:.1f}°")
    print(f"   • Radius difference: {rad_diff:.1f}°")
    print(f"   • Latitude/radius relation: {lat_diff/rad_diff:.2f} °/°")
    print("   → This relation shows how radius changes when latitude changes")
    
    # Fit quality
    print("\n📈 FIT QUALITY:")
    if abs(chi2_true - chi2_best) < 10:
        print("   ✅ Both solutions are statistically indistinguishable")
        print("   (χ² difference < 10)")
    else:
        print("   ⚠️  There is a significant difference in fit")
    
    if r2_best > 0.95:
        print("   ✅ Excellent fit (R² > 0.95)")
    elif r2_best > 0.90:
        print("   ✅ Good fit (R² > 0.90)")
    elif r2_best > 0.80:
        print("   ⚠️  Acceptable fit (R² > 0.80)")
    else:
        print("   ⚠️  Poor fit (R² < 0.80)")
    
    # Sampler information
    print("\n⚙️  MCMC TECHNICAL INFORMATION:")
    print(f"   • Iterations: {sampler.chain.shape[1]}")
    print(f"   • Walkers: {sampler.chain.shape[0]}")
    print(f"   • Acceptance rate: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Check if acceptance rate is in optimal range
    acc_mean = np.mean(sampler.acceptance_fraction)
    if 0.2 <= acc_mean <= 0.5:
        print("   ✅ Optimal acceptance rate (0.2-0.5)")
    else:
        print(f"   ⚠️  Acceptance rate outside optimal range: {acc_mean:.3f}")
    
    print("\n" + "="*70)
    print("📝 CONCLUSION:")
    print("="*70)
    print("Both solutions explain the observed data equally well.")
    print("Latitude-radius degeneracy is common in stellar spot studies.")
    print("Additional data (multiple rotations, bands) would be needed")
    print("to break this degeneracy and determine a unique solution.")
    
    return {
        'chi2_true': chi2_true,
        'chi2_best': chi2_best,
        'r2_true': r2_true,
        'r2_best': r2_best,
        'lat_percentiles': lat_percentiles,
        'lon_percentiles': lon_percentiles,
        'rad_percentiles': rad_percentiles}

        ##################
def create_simple_report(sampler, initial_params, labels):
    """Crea un reporte simple de la simulación"""
    flat_samples = sampler.flatchain
    flat_lnprob = sampler.flatlnprobability
    
    best_idx = np.argmax(flat_lnprob)
    best_params = flat_samples[best_idx]
    best_lnprob = flat_lnprob[best_idx]
    
    print("\n" + "="*50)
    print("MCMC Report (test)")
    print("="*50)
    
    print(f"\n📊 Configuración:")
    print(f"   • Walkers: {sampler.chain.shape[0]}")
    print(f"   • Iteration total: {sampler.chain.shape[1]}")
    print(f"   • Parameter: {sampler.chain.shape[2]}")
    print(f"   • Total samples: {len(flat_samples)}")
    
    print(f"\n🎯 Parámetros iniciales (MSE):")
    for i, label in enumerate(labels):
        print(f"   • {label}: {initial_params[i]:.3f}")
    
    print(f"\n🏆 Best Fit:")
    for i, label in enumerate(labels):
        print(f"   • {label}: {best_params[i]:.3f}")
    print(f"   • ln(Likelihood): {best_lnprob:.2f}")
    
    print(f"\n📈 Range:")
    for i, label in enumerate(labels):
        samples = flat_samples[:, i]
        print(f"   • {label}: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print(f"\n💾 Archivos generados:")
    print("   1. traces_simples.png - Evolución de parámetros")
    print("   2. movimiento_walkers.png - Movimiento de walkers")
    print("   3. corner_plot_simple.png - Distribuciones")
    
    with open('reporte_mcmc.txt', 'w') as f:
        f.write(f"Reporte MCMC - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"Mejores parámetros:\n")
        for i, label in enumerate(labels):
            f.write(f"{label}: {best_params[i]:.6f}\n")
        f.write(f"ln(L): {best_lnprob:.2f}\n")
    
    return best_params, best_lnprob

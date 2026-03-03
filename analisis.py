import corner
import matplotlib.pyplot as plt
import emcee
import numpy as np


def plot_simple_traces_backend(reader, labels):
    """Gráficas simples de evolución de parámetros usando HDFBackend"""
    
    # Obtener la cadena completa del backend
    chain = reader.get_chain(flat=False)  # shape: (nsteps, nwalkers, ndim)
    niter_actual, nwalkers, ndim = chain.shape
    
    # Obtener las log-probabilidades
    lnprob_data = reader.get_log_prob(flat=False)  # shape: (nsteps, nwalkers)
    lnprob_data = lnprob_data.T  # Transponer para tener (nwalkers, nsteps)
    
    # 1. Evolución de cada parámetro
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(10, 2*(ndim+1)))
    
    for i in range(ndim):
        ax = axes[i]
        for j in range(min(nwalkers, 10)):  # Mostrar hasta 10 walkers
            ax.plot(range(niter_actual), chain[:, j, i], alpha=0.5, lw=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    
    # 2. Evolución del likelihood - MANEJAR NaN
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


def create_adaptive_corner_plot(sampler_data, true_params=None, grid_params=None,
                                labels=None, save_path='corner_plot.png'):
    """
    Creates an adaptive corner plot that works for any number of iterations.
    Automatically adjusts: with few samples makes simple plot, with many makes complete corner plot.
    
    Args:
        sampler_data: Dictionary with MCMC results OR sampler object
        true_params: Array with true parameters [lat, lon, radii]
        grid_params: Array with grid search best fit
        labels: List of parameter names (optional)
        save_path: Path to save the plot
    """
    

    
    # Helper function for enhanced corner plot
    def create_enhanced_corner(flat_samples, true_params, grid_params, mcmc_best, labels):
        """Enhanced corner plot with burn-in discarded"""
        # Create the corner plot with more customization
        fig = corner.corner(
            flat_samples,
            labels=labels,
            truths=mcmc_best,
            truth_color='darkblue',
            truth_linestyle=':',
            truth_linewidth=0,
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 10},
            quantiles=[0.16, 0.5, 0.90],
            levels=[0.68, 0.95],
            plot_density=True,
            plot_datapoints=False,
            fill_contours=True,
            smooth=1.0,
            color='skyblue',
            alpha=0.5,
            verbose=False,
            range=[
            (60, 90),      # Latitud: de 55 a 75
            (90, 220),    # Longitud: de 295 a 305
            (20, 55)        # Radio: de 5 a 15 ← ¡SOLUCIÓN DIRECTA!
            ]
        )

        # Get axes for additional annotations
        ndim = flat_samples.shape[1]
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        # Add grid parameters as GREEN SQUARES
        if grid_params is not None:
            for i in range(ndim):
                for j in range(ndim):
                    ax = axes[i, j]
                    if i == j:
                        # 1D histograms - vertical dotted line
                        ylim = ax.get_ylim()
                        ax.plot([grid_params[i], grid_params[i]], ylim, 
                               color='green', linestyle=':', linewidth=2.5, 
                               alpha=0.9, zorder=10)
                    elif i > j:
                        # 2D scatter plots - SQUARE marker
                        ax.scatter(grid_params[j], grid_params[i], 
                                  color='green', marker='s', s=100, 
                                  edgecolor='darkgreen', linewidth=2,
                                  alpha=0.9, zorder=10)



        #tru parameter
        if true_params is not None:
            for i in range(ndim):
                for j in range(ndim):
                    ax = axes[i, j]
                    if i == j:
                        # 1D histograms - vertical dotted line
                        ylim = ax.get_ylim()
                        ax.plot([true_params[i], true_params[i]], ylim, 
                               color='red', linestyle='none', 
                               alpha=0.9, zorder=10)
                    elif i > j:
                        # 2D scatter plots - SQUARE marker
                        ax.scatter(true_params[j], true_params[i], 
                                  color='red', marker='^', s=100, 
                                  edgecolor='darkred', linewidth=2,
                                  alpha=0.9, zorder=10)
        
        # Add MCMC best parameters as BLUE STARS
        if mcmc_best is not None:
            for i in range(ndim):
                for j in range(ndim):
                    ax = axes[i, j]
                    if i == j:
                        # 1D histograms - vertical solid line
                        ylim = ax.get_ylim()
                        ax.plot([mcmc_best[i], mcmc_best[i]], ylim, 
                               color='blue', linestyle='-', linewidth=2.5, 
                               alpha=0.9, zorder=11)
                    elif i > j:
                        # 2D scatter plots - STAR marker
                        ax.scatter(mcmc_best[j], mcmc_best[i], 
                                  color='blue', marker='*', s=150, 
                                  edgecolor='darkblue', linewidth=2,
                                  alpha=0.9, zorder=11)
        
        # Add BEST PARAMETER VALUES as text annotations
        if mcmc_best is not None:
            for i in range(ndim):
                ax = axes[i, i]  # Diagonal plots
                
                # Get current title
                current_title = ax.get_title()
                
                # Add best parameter value to title
                param_name = labels[i] if labels else f'Parameter {i}'
                best_value = mcmc_best[i]
                
                # Format: "Param Name = median +up/-down (best = X.XXX)"
                if '=' in current_title:
                    # Extract median value from existing title
                    parts = current_title.split('=')
                    if len(parts) > 1:
                        median_part = parts[1].strip()
                        # Add best parameter info
                        new_title = f"{param_name}\nmedian: {median_part}\nBest {param_name}: {best_value:.3f}"
                        ax.set_title(new_title, fontsize=9)
        
        # Create legend OUTSIDE the corner plot
        legend_elements = []
        legend_labels = []
        
        # Posterior distribution (filled contours)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.5))
        legend_labels.append('Posterior')
        
        # True parameters (red dashed line)
        if true_params is not None:
            legend_elements.append(plt.Line2D([], [], color='red', 
                                             marker='^', markersize=10, markeredgecolor='darkred',linestyle='none',
                                             markerfacecolor='red'))
            legend_labels.append('True Parameters')
        
        # Grid parameters (green square + dotted line)
        if grid_params is not None:
            # Combined element showing both line and marker
            legend_elements.append(plt.Line2D([], [], color='green', 
                                             marker='s', markersize=10, markeredgecolor='darkgreen',linestyle='none',
                                             markerfacecolor='green'))
            legend_labels.append('Best Grid')
        
        # MCMC best parameters (blue star + solid line)
        if mcmc_best is not None:
            legend_elements.append(plt.Line2D([], [], color='blue', linestyle='none', 
                                             marker='*', markersize=12, markeredgecolor='darkblue',
                                             markerfacecolor='blue'))
            legend_labels.append('MCMC (best)')
        
                # Add legend to the FIGURE (outside, upper left corner)
        # Add legend to the FIGURE (outside, upper left corner)
        fig.legend(legend_elements, legend_labels, 
                  loc='upper left', 
                  bbox_to_anchor=(0.92, 0.98),  # ESQUINA SUPERIOR IZQUIERDA
                  fontsize=10, frameon=True, fancybox=True, 
                  shadow=True, borderpad=1, handlelength=2)
        
        # Add summary box with best parameters (in the last subplot)
        if mcmc_best is not None:
            summary_text = "Best MCMC parameters:\n"
            for i in range(ndim):
                param_name = labels[i] if labels else f'Param {i}'
                best_val = mcmc_best[i]
                summary_text += f"{param_name}: {best_val:.3f}\n"
            
            if true_params is not None:  # ¡FALTABA ESTO!
                summary_text += "\nTrue parameters:\n"
                for i in range(ndim):
                    param_name = labels[i] if labels else f'Param {i}'
                    true_val = true_params[i]
                    summary_text += f"{param_name}: {true_val:.3f}\n"
            
            if grid_params is not None:
                summary_text += "\nBest Grid parameters:\n"
                for i in range(ndim):
                    param_name = labels[i] if labels else f'Param {i}'
                    grid_val = grid_params[i]
                    summary_text += f"{param_name}: {grid_val:.3f}\n"
            
            # Usar fig.text en lugar de axes.text
            fig.text(0.90, 0.98, summary_text,  # 85% derecha, 95% arriba - ESQUINA SUPERIOR DERECHA
                     fontsize=8, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        return fig
    
    # MAIN FUNCTION LOGIC STARTS HERE
    
    # 1. Get the chains from MCMC data
    if isinstance(sampler_data, dict):
        # Use your saved data structure
        chain = sampler_data['sampler_chain']
        
        # Transpose to match emcee format: [steps, walkers, params]
        # Your data is [walkers, steps, params], so we need to transpose
        chain = np.transpose(chain, (1, 0, 2))
        
        # Get log probability if available
        if 'sampler_lnprobability' in sampler_data:
            log_prob = sampler_data['sampler_lnprobability']
            # Transpose if needed
            if log_prob.shape[0] == chain.shape[1]:  # [walkers, steps]
                log_prob = np.transpose(log_prob, (1, 0))  # to [steps, walkers]
        else:
            log_prob = None
        
        # Get flat samples
        flat_samples = sampler_data['flat_chain']
        
        # Get best parameters from saved data
        if 'best_params' in sampler_data:
            mcmc_best = sampler_data['best_params']
            best_lnprob = sampler_data.get('best_lnprob', None)
        else:
            mcmc_best = None
            best_lnprob = None
        
    else:
        # Original emcee sampler object logic
        try:
            chain = sampler_data.get_chain()  # emcee v3
            log_prob = sampler_data.get_log_prob()
        except AttributeError:
            chain = sampler_data.chain  # emcee v2
            log_prob = getattr(sampler_data, 'lnprobability', None)
        
        flat_samples = chain.reshape(-1, chain.shape[2])
        
        # Calculate best parameters
        if log_prob is not None:
            best_idx = np.unravel_index(np.argmax(log_prob), log_prob.shape)
            mcmc_best = chain[best_idx]
            best_lnprob = log_prob[best_idx]
        else:
            mcmc_best = np.median(flat_samples, axis=0)
            best_lnprob = None
    
    n_steps, n_walkers, n_params = chain.shape
    total_samples = n_steps * n_walkers
    
    print(f"\n📊 MCMC STATS: {n_steps} steps × {n_walkers} walkers = {total_samples} samples")
    if best_lnprob is not None:
        print(f"📈 Best log probability: {best_lnprob:.2f}")
    
    # 2. Determine the best MCMC parameter (if not already set)
    if mcmc_best is None:
        if log_prob is not None:
            # Find maximum log probability
            best_idx = np.unravel_index(np.argmax(log_prob), log_prob.shape)
            mcmc_best = chain[best_idx]
        else:
            # Fallback: use median of all samples
            mcmc_best = np.median(flat_samples, axis=0)
    
    # 3. DECISION: What type of plot to make?
    # Use default labels if not provided
    if labels is None:
        if isinstance(sampler_data, dict) and 'labels' in sampler_data:
            labels = sampler_data['labels']
        else:
            labels = ['Latitude [°]', 'Longitude [°]', 'Spot Radius [°]']
    
    # Ensure labels match parameter count
    if len(labels) > n_params:
        labels = labels[:n_params]
    elif len(labels) < n_params:
        labels = labels + [f'Parameter {i}' for i in range(len(labels), n_params)]
    
    print(f"\n🔍 Best MCMC parameters:")
    for i, (label, value) in enumerate(zip(labels, mcmc_best)):
        print(f"   {label}: {value:.4f}")
        print("📊 Many samples (≥100): Complete corner plot with burn-in...")
        # Discard burn-in (first 20% or 50 steps, whichever is smaller)
        discard = min(50, int(0.2 * n_steps))
        remaining = n_steps - discard
        
        if remaining > 0:
            # Thin if there are many samples
            thin = max(1, int(remaining / 50))  # We want ~50 points per walker
            if isinstance(sampler_data, dict):
                # For saved data, manually discard and thin
                chain_burned = chain[discard:, :, :]
                if thin > 1:
                    chain_burned = chain_burned[::thin, :, :]
                flat_samples = chain_burned.reshape(-1, n_params)
            else:
                # For sampler object, use emcee's method
                flat_samples = sampler_data.get_chain(discard=discard, thin=thin, flat=True)
        else:
            # If discard leaves 0 steps, use everything
            flat_samples = flat_samples
        
        fig = create_enhanced_corner(flat_samples, true_params, grid_params, mcmc_best, labels)
    
    # 4. Adjust layout to make space for the external legend
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Leave space at top for legend
    
    # 5. Save and display
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {save_path}")
    
    # 6. Print parameter comparison
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)
    print(f"{'Parameter':<15} {'True':<12} {'Grid':<12} {'MCMC (best)':<12} {'Δ (MCMC-True)':<12}")
    print("-"*70)
    
    for i, name in enumerate(labels[:n_params]):
        true_val = true_params[i] if true_params is not None else np.nan
        grid_val = grid_params[i] if grid_params is not None else np.nan
        mcmc_val = mcmc_best[i]
        delta = mcmc_val - true_val if true_params is not None else np.nan
        
        true_str = f"{true_val:.4f}" if true_params is not None else "N/A"
        grid_str = f"{grid_val:.4f}" if grid_params is not None else "N/A"
        delta_str = f"{delta:+.4f}" if true_params is not None else "N/A"
        
        print(f"{name:<15} {true_str:<12} {grid_str:<12} {mcmc_val:<12.4f} {delta_str:<12}")
    
    # 7. Also create a separate text file with best parameters

    
    return fig, mcmc_best


    


# EJEMPLO DE USO
if __name__ == "__main__":

    reader = emcee.backends.HDFBackend("backend_mcmc.h5")


    labels = ['lat', 'lon','radii']
    # Tus parámetros (ajusta estos valores)
    true_params = None  # Valores verdaderos
    grid_params = None  # Resultados del grid search
    
    # Crear plot con leyenda externa
    fig, mcmc_best = create_adaptive_corner_plot(
        sampler_data=reader,
        true_params=true_params,
        grid_params=grid_params,
        labels=['Latitude [°]', 'Longitude [°]', 'Spot Radius [°]'],
        save_path='mcmc_results_with_external_legend.png',

       
        
    )

    plt.show()
    
    print(f"\n✅ Best parameters:")
    print(f"   Latitude: {mcmc_best[0]:.4f}°")
    print(f"   Longitude: {mcmc_best[1]:.4f}°")
    print(f"   Spot Radius: {mcmc_best[2]:.4f}°")

# Usarla con tu reader
    fig = plot_simple_traces_backend(reader, labels)

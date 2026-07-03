"""
Visualization module for laminate state analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .laminate import Laminate


class StateVisualizer:
    """
    Visualizer for plotting strain and stress distributions through laminate thickness.
    
    Attributes
    ----------
    laminate : Laminate
        The laminate object containing analysis results
    N : np.ndarray
        In-plane force resultants [Nx, Ny, Nxy] (N/m)
    M : np.ndarray
        Moment resultants [Mx, My, Mxy] (N·m/m)
    eps0 : np.ndarray
        Midplane strains [εx⁰, εy⁰, γxy⁰]
    kappa : np.ndarray
        Midplane curvatures [κx, κy, κxy] (1/m)
    """
    
    def __init__(self, laminate: 'Laminate', N: np.ndarray, M: np.ndarray):
        """
        Initialize visualizer with a Laminate object and load case.
        
        Parameters
        ----------
        laminate : Laminate
            Laminate object
        N : array_like
            In-plane force resultants [Nx, Ny, Nxy] in N/m
        M : array_like
            Moment resultants [Mx, My, Mxy] in N·m/m
        """
        self.laminate = laminate
        self.layup = laminate.layup
        self.N = np.asarray(N, dtype=float)
        self.M = np.asarray(M, dtype=float)
        
        # Compute midplane strains and curvatures
        self.eps0, self.kappa = self.laminate.get_midplane_strains(self.N, self.M)
        
        # Get layer positions
        self.layer_positions = self.layup.get_layer_positions()
    
    def _get_z_coords(self) -> np.ndarray:
        """Get all unique z coordinates (layer boundaries)."""
        z_list = []
        for z_bot, z_top in self.layer_positions:
            z_list.extend([z_bot, z_top])
        return np.unique(z_list)
    
    def plot_strain_distribution(
        self,
        component: Literal['x', 'y', 'xy'] = 'x',
        surface: Literal['top', 'mid', 'bottom'] = 'mid',
        theme: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (10, 6),
        show_layers: bool = True
    ) -> None:
        """
        Plot strain distribution through laminate thickness.
        
        Parameters
        ----------
        component : {'x', 'y', 'xy'}
            Strain component to plot (εₓ, εᵧ, γₓᵧ)
        surface : {'top', 'mid', 'bottom'}
            Which surface of each layer to evaluate
        theme : str
            Matplotlib style theme
        figsize : tuple
            Figure size (width, height)
        show_layers : bool
            Whether to show layer boundaries with vertical lines
        """
        idx = {'x': 0, 'y': 1, 'xy': 2}[component]
        labels = {'x': r'$\varepsilon_x$', 'y': r'$\varepsilon_y$', 'xy': r'$\gamma_{xy}$'}
        
        with plt.style.context(theme):
            fig, ax = plt.subplots(figsize=figsize)
            
            z_coords = self._get_z_coords()
            n_layers = len(self.layup.layers)
            
            # Plot each layer segment
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                layer = self.layup.layers[i]
                
                # Get strains at bottom and top
                eps_bot = self.laminate.get_strains_at_z(z_bot, self.eps0, self.kappa)
                eps_top = self.laminate.get_strains_at_z(z_top, self.eps0, self.kappa)
                
                # Extract component
                e_bot = eps_bot[idx]
                e_top = eps_top[idx]
                
                # Linear interpolation
                z_plot = np.array([z_bot, z_top])
                e_plot = np.array([e_bot, e_top])
                
                ax.plot(e_plot * 1e6, z_plot * 1e3, 
                       label=f'Layer {i+1}: {layer.angle}°', 
                       marker='o', markersize=4, linewidth=2)
            
            # Layer boundaries
            if show_layers:
                for z in z_coords:
                    ax.axhline(z * 1e3, color='gray', linestyle='--', 
                              alpha=0.3, linewidth=0.8)
            
            ax.set_xlabel(f'{labels[component]} (με)', fontsize=12)
            ax.set_ylabel('$z$ (mm)', fontsize=12)
            ax.set_title(f'Strain Distribution: {labels[component]}', fontsize=14, weight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def plot_stress_distribution(
        self,
        component: Literal['x', 'y', 'xy'] = 'x',
        surface: Literal['top', 'mid', 'bottom'] = 'mid',
        theme: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (10, 6),
        show_layers: bool = True
    ) -> None:
        """
        Plot stress distribution through laminate thickness.
        
        Parameters
        ----------
        component : {'x', 'y', 'xy'}
            Stress component to plot (σₓ, σᵧ, τₓᵧ)
        surface : {'top', 'mid', 'bottom'}
            Which surface of each layer to evaluate
        theme : str
            Matplotlib style theme
        figsize : tuple
            Figure size (width, height)
        show_layers : bool
            Whether to show layer boundaries
        """
        idx = {'x': 0, 'y': 1, 'xy': 2}[component]
        labels = {'x': r'$\sigma_x$', 'y': r'$\sigma_y$', 'xy': r'$\tau_{xy}$'}
        
        with plt.style.context(theme):
            fig, ax = plt.subplots(figsize=figsize)
            
            z_coords = self._get_z_coords()
            n_layers = len(self.layup.layers)
            
            # Plot each layer segment
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                layer = self.layup.layers[i]
                
                # Get stresses at bottom and top
                sig_bot = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='bottom')
                sig_top = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='top')
                
                # Extract component
                s_bot = sig_bot[idx]
                s_top = sig_top[idx]
                
                z_plot = np.array([z_bot, z_top])
                s_plot = np.array([s_bot, s_top])
                
                ax.plot(s_plot * 1e-6, z_plot * 1e3, 
                       label=f'Layer {i+1}: {layer.angle}°', 
                       marker='o', markersize=4, linewidth=2)
            
            # Layer boundaries
            if show_layers:
                for z in z_coords:
                    ax.axhline(z * 1e3, color='gray', linestyle='--', 
                              alpha=0.3, linewidth=0.8)
            
            ax.set_xlabel(f'{labels[component]} (MPa)', fontsize=12)
            ax.set_ylabel('$z$ (mm)', fontsize=12)
            ax.set_title(f'Stress Distribution: {labels[component]}', fontsize=14, weight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def plot_all_strains(
        self,
        theme: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (15, 5)
    ) -> None:
        """
        Plot all three strain components in subplots.
        """
        with plt.style.context(theme):
            fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
            
            components = ['x', 'y', 'xy']
            labels = [r'$\varepsilon_x$ (με)', r'$\varepsilon_y$ (με)', r'$\gamma_{xy}$ (με)']
            
            z_coords = self._get_z_coords()
            n_layers = len(self.layup.layers)
            
            for ax, comp, label in zip(axes, components, labels):
                idx = {'x': 0, 'y': 1, 'xy': 2}[comp]
                
                for i in range(n_layers):
                    z_bot, z_top = self.layer_positions[i]
                    
                    eps_bot = self.laminate.get_strains_at_z(z_bot, self.eps0, self.kappa)
                    eps_top = self.laminate.get_strains_at_z(z_top, self.eps0, self.kappa)
                    
                    e_bot = eps_bot[idx]
                    e_top = eps_top[idx]
                    
                    z_plot = np.array([z_bot, z_top])
                    e_plot = np.array([e_bot, e_top])
                    
                    ax.plot(e_plot * 1e6, z_plot * 1e3, 
                           marker='o', markersize=3, linewidth=2)
                
                # Layer boundaries
                for z in z_coords:
                    ax.axhline(z * 1e3, color='gray', linestyle='--', 
                              alpha=0.3, linewidth=0.8)
                
                ax.set_xlabel(label, fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{comp}-component', fontsize=12, weight='bold')
            
            axes[0].set_ylabel('$z$ (mm)', fontsize=11)
            fig.suptitle('Strain Distribution Through Thickness', fontsize=14, weight='bold')
            plt.tight_layout()
            plt.show()
    
    def plot_all_stresses(
        self,
        theme: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (15, 5)
    ) -> None:
        """
        Plot all three stress components in subplots.
        """
        with plt.style.context(theme):
            fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
            
            components = ['x', 'y', 'xy']
            labels = [r'$\sigma_x$ (MPa)', r'$\sigma_y$ (MPa)', r'$\tau_{xy}$ (MPa)']
            
            z_coords = self._get_z_coords()
            n_layers = len(self.layup.layers)
            
            for ax, comp, label in zip(axes, components, labels):
                idx = {'x': 0, 'y': 1, 'xy': 2}[comp]
                
                for i in range(n_layers):
                    sig_bot = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='bottom')
                    sig_top = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='top')
                    
                    z_bot, z_top = self.layer_positions[i]
                    
                    s_bot = sig_bot[idx]
                    s_top = sig_top[idx]
                    
                    z_plot = np.array([z_bot, z_top])
                    s_plot = np.array([s_bot, s_top])
                    
                    ax.plot(s_plot * 1e-6, z_plot * 1e3, 
                           marker='o', markersize=3, linewidth=2)
                
                # Layer boundaries
                for z in z_coords:
                    ax.axhline(z * 1e3, color='gray', linestyle='--', 
                              alpha=0.3, linewidth=0.8)
                
                ax.set_xlabel(label, fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{comp}-component', fontsize=12, weight='bold')
            
            axes[0].set_ylabel('$z$ (mm)', fontsize=11)
            fig.suptitle('Stress Distribution Through Thickness', fontsize=14, weight='bold')
            plt.tight_layout()
            plt.show()
    
    def plot_combined_overview(
        self,
        figsize: tuple = (16, 10)
    ) -> None:
        """
        Create a comprehensive 2×3 grid showing all strain and stress distributions.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
        
        components = ['x', 'y', 'xy']
        strain_labels = [r'$\varepsilon_x$ (με)', r'$\varepsilon_y$ (με)', r'$\gamma_{xy}$ (με)']
        stress_labels = [r'$\sigma_x$ (MPa)', r'$\sigma_y$ (MPa)', r'$\tau_{xy}$ (MPa)']
        
        z_coords = self._get_z_coords()
        n_layers = len(self.layup.layers)
        
        # Row 1: Strains
        for col, (comp, label) in enumerate(zip(components, strain_labels)):
            ax = axes[0, col]
            idx = {'x': 0, 'y': 1, 'xy': 2}[comp]
            
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                layer = self.layup.layers[i]
                
                eps_bot = self.laminate.get_strains_at_z(z_bot, self.eps0, self.kappa)
                eps_top = self.laminate.get_strains_at_z(z_top, self.eps0, self.kappa)
                
                z_plot = np.array([z_bot, z_top])
                e_plot = np.array([eps_bot[idx], eps_top[idx]])
                
                ax.plot(e_plot * 1e6, z_plot * 1e3, 
                       label=f'{i+1}: {layer.angle}°', 
                       marker='o', markersize=3, linewidth=1.5)
            
            for z in z_coords:
                ax.axhline(z * 1e3, color='gray', linestyle='--', alpha=0.2, linewidth=0.6)
            
            ax.set_xlabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Strain: {comp}', fontsize=11, weight='bold')
            if col == 0:
                ax.set_ylabel('$z$ (mm)', fontsize=10)
        
        # Row 2: Stresses
        for col, (comp, label) in enumerate(zip(components, stress_labels)):
            ax = axes[1, col]
            idx = {'x': 0, 'y': 1, 'xy': 2}[comp]
            
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                layer = self.layup.layers[i]
                
                sig_bot = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='bottom')
                sig_top = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='top')
                
                z_plot = np.array([z_bot, z_top])
                s_plot = np.array([sig_bot[idx], sig_top[idx]])
                
                ax.plot(s_plot * 1e-6, z_plot * 1e3, 
                       label=f'{i+1}: {layer.angle}°', 
                       marker='o', markersize=3, linewidth=1.5)
            
            for z in z_coords:
                ax.axhline(z * 1e3, color='gray', linestyle='--', alpha=0.2, linewidth=0.6)
            
            ax.set_xlabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Stress: {comp}', fontsize=11, weight='bold')
            if col == 0:
                ax.set_ylabel('$z$ (mm)', fontsize=10)
        
        # Global legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=min(n_layers, 6), 
                  bbox_to_anchor=(0.5, 0.98), fontsize=9, title='Layers')
        
        fig.suptitle('Laminate State: Strain and Stress Distributions', 
                    fontsize=14, weight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def summary_table(self) -> None:
        """
        Print formatted table of strains and stresses at each layer interface.
        """
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(title="Laminate State Summary", show_header=True, header_style="bold magenta")
            
            table.add_column("Layer", style="cyan", justify="center")
            table.add_column("z (mm)", justify="right")
            table.add_column("εₓ (με)", justify="right")
            table.add_column("εᵧ (με)", justify="right")
            table.add_column("γₓᵧ (με)", justify="right")
            table.add_column("σₓ (MPa)", justify="right")
            table.add_column("σᵧ (MPa)", justify="right")
            table.add_column("τₓᵧ (MPa)", justify="right")
            
            n_layers = len(self.layup.layers)
            
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                
                eps_bot = self.laminate.get_strains_at_z(z_bot, self.eps0, self.kappa)
                eps_top = self.laminate.get_strains_at_z(z_top, self.eps0, self.kappa)
                
                sig_bot = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='bottom')
                sig_top = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='top')
                
                # Bottom interface
                table.add_row(
                    f"{i+1} ↓",
                    f"{z_bot*1e3:.3f}",
                    f"{eps_bot[0]*1e6:.2f}",
                    f"{eps_bot[1]*1e6:.2f}",
                    f"{eps_bot[2]*1e6:.2f}",
                    f"{sig_bot[0]*1e-6:.2f}",
                    f"{sig_bot[1]*1e-6:.2f}",
                    f"{sig_bot[2]*1e-6:.2f}"
                )
                
                # Top interface (only for last layer)
                if i == n_layers - 1:
                    table.add_row(
                        f"{i+1} ↑",
                        f"{z_top*1e3:.3f}",
                        f"{eps_top[0]*1e6:.2f}",
                        f"{eps_top[1]*1e6:.2f}",
                        f"{eps_top[2]*1e6:.2f}",
                        f"{sig_top[0]*1e-6:.2f}",
                        f"{sig_top[1]*1e-6:.2f}",
                        f"{sig_top[2]*1e-6:.2f}"
                    )
            
            console.print(table)
            
        except ImportError:
            # Fallback to simple print
            print("\n" + "="*80)
            print("LAMINATE STATE SUMMARY")
            print("="*80)
            print(f"{'Layer':<10} {'z(mm)':<10} {'εₓ(με)':<12} {'εᵧ(με)':<12} {'γₓᵧ(με)':<12} {'σₓ(MPa)':<12} {'σᵧ(MPa)':<12} {'τₓᵧ(MPa)':<12}")
            print("-"*110)
            
            n_layers = len(self.layup.layers)
            
            for i in range(n_layers):
                z_bot, z_top = self.layer_positions[i]
                
                eps_bot = self.laminate.get_strains_at_z(z_bot, self.eps0, self.kappa)
                eps_top = self.laminate.get_strains_at_z(z_top, self.eps0, self.kappa)
                
                sig_bot = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='bottom')
                sig_top = self.laminate.get_layer_stresses(i, self.eps0, self.kappa, surface='top')
                
                print(f"{i+1} ↓        {z_bot*1e3:>8.3f}  {eps_bot[0]*1e6:>10.2f}  {eps_bot[1]*1e6:>10.2f}  "
                      f"{eps_bot[2]*1e6:>10.2f}  {sig_bot[0]*1e-6:>10.2f}  {sig_bot[1]*1e-6:>10.2f}  {sig_bot[2]*1e-6:>10.2f}")
                
                if i == n_layers - 1:
                    print(f"{i+1} ↑        {z_top*1e3:>8.3f}  {eps_top[0]*1e6:>10.2f}  {eps_top[1]*1e6:>10.2f}  "
                          f"{eps_top[2]*1e6:>10.2f}  {sig_top[0]*1e-6:>10.2f}  {sig_top[1]*1e-6:>10.2f}  {sig_top[2]*1e-6:>10.2f}")
            
            print("="*80 + "\n")

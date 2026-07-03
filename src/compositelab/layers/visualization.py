"""
Visualization module for composite layup structures.
Clean 3D visualization with labels ON the front face of layers.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.console import Console
from rich.table import Table


class LayupVisualizer:
    """Professional visualization tools for composite layups."""
    
    # Standard angle color palette
    ANGLE_COLORS = {
        0: '#2E86AB',      # Blue
        45: '#A23B72',     # Magenta
        -45: '#F18F01',    # Orange
        90: '#C73E1D',     # Red
    }
    
    @staticmethod
    def _get_color_for_angle(angle: float) -> tuple:
        """Get RGB color for a given fiber angle with smooth interpolation."""
        angle = angle % 180
        if angle > 90:
            angle = angle - 180
            
        standard_angles = sorted(LayupVisualizer.ANGLE_COLORS.keys())
        
        if angle in LayupVisualizer.ANGLE_COLORS:
            hex_color = LayupVisualizer.ANGLE_COLORS[angle]
            return tuple(int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5))
        
        below = max([a for a in standard_angles if a <= angle], default=standard_angles[0])
        above = min([a for a in standard_angles if a >= angle], default=standard_angles[-1])
        
        if below == above:
            hex_color = LayupVisualizer.ANGLE_COLORS[below]
            return tuple(int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5))
        
        t = (angle - below) / (above - below)
        color_below = LayupVisualizer.ANGLE_COLORS[below]
        color_above = LayupVisualizer.ANGLE_COLORS[above]
        
        rgb_below = np.array([int(color_below[i:i+2], 16)/255 for i in (1, 3, 5)])
        rgb_above = np.array([int(color_above[i:i+2], 16)/255 for i in (1, 3, 5)])
        
        rgb = (1 - t) * rgb_below + t * rgb_above
        return tuple(rgb)
    
    @staticmethod
    def print_table(layup):
        """
        Print formatted table of layup information.
        
        Handles SI units correctly:
        - Input thickness and positions are in meters (m)
        - Display is converted to millimeters (mm) for readability
        
        Args:
            layup: Layup object containing layer information
        """
        console = Console()
        table = Table(title=f"\n[bold cyan]{layup.name}[/bold cyan]", 
                     title_style="bold cyan",
                     show_header=True, 
                     header_style="bold magenta")
        
        table.add_column("Layer", justify="center", style="cyan")
        table.add_column("Material", justify="left", style="green")
        table.add_column("Thickness (mm)", justify="right", style="yellow")
        table.add_column("Angle (°)", justify="right", style="magenta")
        table.add_column("Z_bottom (mm)", justify="right", style="blue")
        table.add_column("Z_top (mm)", justify="right", style="blue")
        
        positions = layup.get_layer_positions()
        
        for i, (layer, (z_bot, z_top)) in enumerate(zip(layup.layers, positions), 1):
            # Convert from meters to millimeters for display
            thickness_mm = layer.thickness * 1000
            z_bot_mm = z_bot * 1000
            z_top_mm = z_top * 1000
            
            table.add_row(
                str(i),
                layer.material.name,
                f"{thickness_mm:.4f}",
                f"{layer.angle:+.1f}",
                f"{z_bot_mm:.4f}",
                f"{z_top_mm:.4f}"
            )
        
        console.print(table)
        
        # Convert total thickness to mm
        total_thickness_mm = layup.total_thickness * 1000
        console.print(f"\n[bold]Total Thickness:[/bold] [yellow]{total_thickness_mm:.4f} mm[/yellow]")
        console.print(f"[bold]Number of Layers:[/bold] [yellow]{len(layup.layers)}[/yellow]\n")

    @staticmethod
    def plot_3d(layup, 
                width: float = 0.05,      # 50 mm in meters
                length: float = 0.05,     # 50 mm in meters
                figsize: tuple = (16, 12),
                elevation: float = 20, 
                azimuth: float = -60,
                layupfontsize: float =9, 
                show: bool = True):
        """
        Create clean 3D visualization with labels ON the front face.
        
        All dimensions are in meters (SI base unit).
        
        Args:
            layup: Layup object
            width: Width in meters (default: 0.05 m = 50 mm)
            length: Length in meters (default: 0.05 m = 50 mm)
            figsize: Figure size in inches
            elevation: View elevation angle in degrees
            azimuth: View azimuth angle in degrees
            show: Whether to display the plot
            
        Returns:
            matplotlib figure and axes objects
        """
        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        # Get layer positions (in meters)
        positions = layup.get_layer_positions()
        
        # Convert to mm for display
        width_mm = width * 1000
        length_mm = length * 1000
        
        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Track unique angles for legend
        angle_legend = {}
        
        # Draw each layer
        for i, (layer, (z_bot, z_top)) in enumerate(zip(layup.layers, positions)):
            # Convert z to mm
            z_bot_mm = z_bot * 1000
            z_top_mm = z_top * 1000
            z_mid_mm = (z_bot_mm + z_top_mm) / 2
            
            color = LayupVisualizer._get_color_for_angle(layer.angle)
            
            angle_key = f"{layer.angle:+.0f}°"
            if angle_key not in angle_legend:
                angle_legend[angle_key] = color
            
            # Define box vertices in mm
            vertices = [
                # Bottom face
                [[-width_mm/2, -length_mm/2, z_bot_mm],
                 [width_mm/2, -length_mm/2, z_bot_mm],
                 [width_mm/2, length_mm/2, z_bot_mm],
                 [-width_mm/2, length_mm/2, z_bot_mm]],
                # Top face
                [[-width_mm/2, -length_mm/2, z_top_mm],
                 [width_mm/2, -length_mm/2, z_top_mm],
                 [width_mm/2, length_mm/2, z_top_mm],
                 [-width_mm/2, length_mm/2, z_top_mm]]
            ]
            
            # Create all 6 faces
            faces = []
            faces.append(vertices[0])  # Bottom
            faces.append(vertices[1])  # Top
            
            # Four sides
            for j in range(4):
                next_j = (j + 1) % 4
                faces.append([
                    vertices[0][j],
                    vertices[0][next_j],
                    vertices[1][next_j],
                    vertices[1][j]
                ])
            
            # Draw the layer box
            poly = Poly3DCollection(faces, alpha=0.85, 
                                   facecolors=color, 
                                   edgecolors='black',
                                   linewidths=0.8)
            ax.add_collection3d(poly)
            
            # *** LABEL ON FRONT FACE ***
            # Position on the front vertical face (y = length/2, middle of x and z)
            x_label = -width_mm/2 * 1.5   # Center horizontally
            y_label = width_mm/2 * 1.5  # On the front face
            z_label = z_mid_mm  # Middle of the layer
            
            # Single-line compact label
            label_text = f"L{i+1}: {layer.material.name} {layer.angle:+.0f}°"
            
            ax.text(x_label, y_label, z_label, 
                   label_text,
                   fontsize=layupfontsize, 
                   ha='center', 
                   va='center',
                   color='white',
                   weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='black', 
                           edgecolor='white', 
                           linewidth=1.5, 
                           alpha=0.8))
        
        # Configure axes (all in mm)

        ax.set_zlabel('Z (mm)', fontsize=12, labelpad=10, weight='bold')
        ax.set_xlabel('X', fontsize=12, labelpad=10, weight='bold')
        ax.set_ylabel('Y', fontsize=12, labelpad=10, weight='bold')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

    
        # Set axis limits
        ax.set_xlim(-width_mm/2 * 1.5, width_mm/2 * 1.5)
        ax.set_ylim(-length_mm/2 * 1.2, length_mm/2 * 1.8)
        
        z_min = min(pos[0] for pos in positions) * 1000
        z_max = max(pos[1] for pos in positions) * 1000
        z_range = z_max - z_min
        
        # Smart Z-axis padding for thin laminates
        z_padding = max(z_range * 0.4, 0.15)
        ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        # Clean grid
        ax.grid(True, linewidth=0.5, alpha=0.3, color='gray')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        
        ax.tick_params(labelsize=10)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, linewidth=5, label=angle)
                          for angle, color in sorted(angle_legend.items())]
        ax.legend(handles=legend_elements, 
                 title='Ply Angles',
                 loc='upper left', 
                 framealpha=0.95,
                 fontsize=11,
                 title_fontsize=12)
        
        # Title
        total_mm = layup.total_thickness * 1000
        title = f"{layup.name}\n{len(layup.layers)} Plies | Total: {total_mm:.4f} mm"
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.97)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax

"""Layup module for defining composite laminates."""

import re
import numpy as np
from typing import List, Tuple, Optional
from compositelab.layers import Layer

class Layup:
    """
    Represents a complete laminate layup.
    
    Parameters
    ----------
    layers : List[Layer], optional
        List of layers from bottom to top
    name : str, optional
        Layup identifier
    """
    
    _id_counter = 0
    
    def __init__(self, layers: Optional[List[Layer]] = None, name: Optional[str] = None):
        Layup._id_counter += 1
        self.id = Layup._id_counter
        self._layers = layers if layers is not None else []
        self.name = name or f"Layup_{self.id}"
    
    def add_layer(
        self,
        material,
        thickness: float,
        angle: float = 0.0,
        name: Optional[str] = None
    ) -> None:
        """
        Add a layer to the layup.
        
        Parameters
        ----------
        material : Material
            Material object
        thickness : float
            Layer thickness in meters (m)
        angle : float
            Fiber orientation angle in degrees
        name : str, optional
            Layer name
        """
        layer = Layer(material, thickness, angle, name)
        self._layers.append(layer)
    
    @property
    def layers(self) -> List[Layer]:
        """List of all layers."""
        return self._layers
    
    @property
    def num_layers(self) -> int:
        """Total number of layers."""
        return len(self._layers)
    
    @property
    def total_thickness(self) -> float:
        """Total laminate thickness in meters (m)."""
        return sum(layer.thickness for layer in self._layers)
    
    @property
    def stacking_sequence(self) -> str:
        """
        Stacking sequence notation.
        
        Returns
        -------
        str
            Notation like '[0/90/45/-45]' or '[0/90]s'
        """
        if not self._layers:
            return "[]"
        
        angles = [layer.angle for layer in self._layers]
        n = len(angles)
        
        # Check for symmetry
        mid = n // 2
        if n % 2 == 0:
            is_symmetric = angles[:mid] == angles[mid:][::-1]
        else:
            is_symmetric = angles[:mid] == angles[mid+1:][::-1]
        
        if is_symmetric:
            if n % 2 == 0:
                half = angles[:mid]
            else:
                half = angles[:mid+1]
            return f"[{'/'.join(map(str, half))}]s"
        else:
            return f"[{'/'.join(map(str, angles))}]"
    
    def make_symmetric(self) -> 'Layup':
        """Make the layup symmetric by mirroring existing layers."""
        if not self._layers:
            return self
        
        # Store original layers
        original_layers = self._layers.copy()
        
        # Mirror: reverse order, excluding the last to avoid duplication
        mirrored = []
        for layer in reversed(original_layers):
            mirrored.append(
                Layer(layer.material, layer.thickness, layer.angle, layer.name)
            )
        
        self._layers.extend(mirrored)
        return self
    
    @classmethod
    def from_stacking(
        cls,
        notation: str,
        material,
        thickness: float
    ) -> 'Layup':
        """
        Create a layup from stacking sequence notation.
        
        Parameters
        ----------
        notation : str
            Stacking sequence like '[0/90]s', '[45/-45/0]2s', '[0/90/45/-45]'
        material : Material
            Common material for all layers
        thickness : float
            Thickness of each ply in meters (m)
        
        Returns
        -------
        Layup
            Layup object
        
        Examples
        --------
        >>> layup = Layup.from_stacking('[0/90]s', carbon, 0.000125)
        # Result: [0/90/90/0]
        >>> layup = Layup.from_stacking('[0/45/90]s', carbon, 0.000125)
        # Result: [0/45/90/90/45/0]
        >>> layup = Layup.from_stacking('[45/-45/0]2', glass, 0.0002)
        # Result: [45/-45/0/45/-45/0]
        """
        notation = notation.strip()
        
        if not (notation.startswith('[') and ']' in notation):
            raise ValueError("Invalid format. Must be in [angles] format")
        
        # Parse notation
        bracket_end = notation.rfind(']')
        core = notation[1:bracket_end]
        suffix = notation[bracket_end + 1:].strip().lower()
        
        # Extract angles
        angle_strs = [s.strip() for s in core.split('/')]
        angles = []
        for angle_str in angle_strs:
            try:
                angles.append(float(angle_str))
            except ValueError:
                raise ValueError(f"Invalid angle: {angle_str}")
        
        # Parse suffix for repetition and symmetry
        repeat = 1
        symmetric = False
        
        if suffix:
            if suffix.endswith('s'):
                symmetric = True
                suffix = suffix[:-1]
            
            if suffix:
                try:
                    repeat = int(suffix)
                except ValueError:
                    raise ValueError(f"Invalid repetition factor: {suffix}")
        
        # Build angle sequence
        final_angles = angles * repeat
        
        # Apply symmetry - FIXED: Now includes the middle layer(s)
        if symmetric:
            # Correct symmetry: [0/45/90]s → [0/45/90/90/45/0]
            final_angles = final_angles + final_angles[::-1]
        
        # Create layers
        layers = [Layer(material, thickness, angle) for angle in final_angles]
        
        return cls(layers)
    
    def get_layer_positions(self) -> List[Tuple[float, float]]:
        """
        Calculate z-coordinates of layers relative to mid-plane.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (z_bottom, z_top) tuples for each layer in meters (m)
        """
        h = self.total_thickness
        z = -h / 2
        positions = []
        
        for layer in self._layers:
            z_bottom = z
            z_top = z + layer.thickness
            positions.append((z_bottom, z_top))
            z = z_top
        
        return positions
    
    def summary(self) -> None:
        """Print layup summary table."""
        from compositelab.layers.visualization import LayupVisualizer
        LayupVisualizer.print_table(self)
    
    def plot_3d(self, width: float = 100, length: float = 100, 
                figsize: tuple = (16, 12),
                elevation: float = 25, azimuth: float = 45, 
                layupfontsize: float = 9,
                show: bool = True):
        """Create 3D visualization of composite layup with fiber cylinders.
        
        Args:
            width: Width dimension in mm (default: 100)
            length: Length dimension in mm (default: 100)
            figsize: Figure size in inches (default: (16, 12))
            elevation: Camera elevation angle in degrees (default: 25)
            azimuth: Camera azimuth angle in degrees (default: 45)
            layupfontsize: Font size for layer labels (default: 9)
            show: Whether to display the plot immediately (default: True)
            
        Returns:
            fig, ax: matplotlib figure and 3D axis objects
        """
        from compositelab.layers.visualization import LayupVisualizer
        return LayupVisualizer.plot_3d(
            self, width, length, figsize, elevation, azimuth, layupfontsize, show
        )
    
    def __repr__(self) -> str:
        # Display in mm for readability, store in meters
        return (
            f"Layup(name={self.name}, "
            f"layers={self.num_layers}, "
            f"h={self.total_thickness*1000:.4f}mm)"
        )
    
    def __len__(self) -> int:
        return len(self._layers)
    
    def __getitem__(self, index) -> Layer:
        return self._layers[index]

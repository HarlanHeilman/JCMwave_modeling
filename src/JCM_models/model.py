import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import pickle
import json
from typing import Any, Dict, List, Optional
from .utils import eVnm_converter

colors20 = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


class Shape:
    """
    🔷 Shape: A domain-aware polygonal object with optical properties.

    Represents a geometric shape defined by:
    • A name and domain ID
    • A priority for simulation layering
    • A side length constraint (for meshing or physical limits)
    • A list of 2D points (flattened [x0, y0, x1, y1, ...])
    • A complex refractive index (nk)
    • Boundary conditions per edge (default: ['Transparent','Periodic','Transparent','Periodic'])
    • Gradient dic, if gradient is present

    Automatically computes:
    • Permittivity (ε) as nk²

    Methods:
    • describe(): returns a summary of the shape's identity and optical properties
    • plot(ax=None, **kwargs): visualizes the shape as a closed polygon using Matplotlib

    Example:
        shape = Shape(
            name='Slab',
            domain_id=1,
            priority=0,
            side_length_constraint=1.0,
            points=[-1, -1, 1, -1, 1, 1, -1, 1],
            nk=2.0 + 0.1j
        )
        shape.plot()
        print(shape.describe())
    """

    def __init__(self, name, domain_id, priority, side_length_constraint, points,nk,boundary = ['Transparent','Periodic','Transparent','Periodic'],gradient_dict=None):
        self.name = name
        self.domain_id = domain_id
        self.priority = priority
        self.side_length_constraint = side_length_constraint
        self.points = np.array([float(p) for p in points])
        self.nk = nk
        self.boundary = boundary
        self.gradient_dict = gradient_dict

        self.permittivity = np.square(self.nk)

    def describe(self):
        return f"""Shape: {self.name}
  Domain ID: {self.domain_id}
  Priority: {self.priority}
  Side Length Constraint: {self.side_length_constraint}
  Refractive Index (nk): {self.nk}
  Permittivity (ε): {self.permittivity}
"""
    def plot(self, ax=None, shift_x=0,shift_y=0 , **kwargs):
        x = self.points[::2]+shift_x
        y = self.points[1::2]+shift_y
        x = np.append(x, x[0])  # Close the polygon
        y = np.append(y, y[0])
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_title(f"Shape: {self.name}")
            ax.legend()
            
            ax.plot(x, y, label=self.name, **kwargs)
            return fig
        else:
            ax.plot(x, y, label=self.name, **kwargs)
            return ax
        
    def plot_colored_geometry(self, ax=None, shift_x=0, shift_y=0, **kwargs):
        points = np.asarray(self.points, copy=True).reshape(-1, 2)
        points[:, 0] += shift_x
        points[:, 1] += shift_y

        # Allow user override of facecolor
        facecolor = kwargs.pop("facecolor", colors20[self.domain_id - 1])

        polygon = Polygon(
            points,
            closed=True,
            facecolor=facecolor,
            edgecolor='black',
            alpha=1,
            label=self.name,
            zorder=self.priority,
            **kwargs
        )

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_title(f"Shape: {self.name}")
            ax.legend()
            ax.add_patch(polygon)
            return fig
        else:
            ax.add_patch(polygon)
            return ax

        
    def to_dict(self):
        return {
            'name': self.name,
            'domain_id': self.domain_id,
            'priority': self.priority,
            'side_length_constraint': self.side_length_constraint,
            'points': self.points,
            'nk': [str(n) for n in self.nk] if isinstance(self.nk, list) else str(self.nk),
            'boundary': self.boundary,
            'permittivity': str(self.permittivity)
        }
        
    def _to_jcm_constant(self, energy_index):
        if self.name == "ComputationalDomain":
            perm = self.permittivity
        else:
            if isinstance(self.permittivity, (list, np.ndarray)):
                if energy_index is None:
                    raise ValueError("energy_index required for energy-dependent nk.")
                perm = self.permittivity[energy_index]
            else:
                perm = self.permittivity

        perm_str = f"{perm:.6e}"

        return f"""
    Material {{
    Name = "{self.name}"
    Id = {self.domain_id}
    RelPermittivity = {perm_str}
    RelPermeability = 1.0
    }}
    """.strip()
    
    def _to_jcm_gradient(self, python_expression):
        """
        Generate a JCMwave Material block with a gradient permittivity defined by a Python expression.
        Define python_expression as a string that can use variables like x, y, z to represent spatial coordinates.
        """


        return f"""
    Material {{
    Name = "{self.name}"
    Id = {self.domain_id}
    RelPermeability = 1.0
    RelPermittivity {{
        Python {{
        Expression = "
        {python_expression}
        "
        }}
    
    }}
    }}
    """.strip()
    
    def _make_gradient_text(self, energy_index, max_depth=3,exponent=1,permittivity_surface=1,uol=1e-9):
        clean_list = [float(v)*uol for v in self.points]
        max_depth = max_depth*uol
        text = f"""
\n
empty = 0;
def point_to_segment_distance(px, py, x1, y1, x2, y2):
\tvx, vy = x2 - x1, y2 - y1
\twx, wy = px - x1, py - y1

\tseg_len2 = vx*vx + vy*vy
\tif seg_len2 == 0:
\t\treturn hypot(px - x1, py - y1)

\tt = (wx*vx + wy*vy) / seg_len2
\tt = clip(t, 0.0, 1.0)

\tproj_x = x1 + t * vx
\tproj_y = y1 + t * vy

\treturn hypot(px - proj_x, py - proj_y)
pts = asarray({clean_list}).reshape(-1, 2)
dists = []
for i in range(len(pts)):
\tx1, y1 = pts[i]
\tx2, y2 = pts[(i + 1) % len(pts)]
\td = point_to_segment_distance(X[0], X[1], x1, y1, x2, y2)
\tdists.append(d)
d = min(dists)

if d >= {max_depth}:
\ttemp = {self.permittivity[energy_index]}
else:
\tt = (d / {max_depth}) ** {exponent}
\ttemp = {self.permittivity[energy_index]} + ({permittivity_surface} - {self.permittivity[energy_index]}) * (1 - t)
value = temp
value = value*eye(3,3)
        """
        return text.strip()
        
    def to_jcm(self, energy_index=None):
        """
        Export this shape as a JCMwave Material{} block.

        Parameters
        ----------
        energy_index : int or None
            Used for energy-dependent nk.
        python_expression : str or None
            Define python_expression as a string
        """
        if self.gradient_dict is None:
            return self._to_jcm_constant(energy_index)
        else:
            max_depth = self.gradient_dict.get("max_depth", 3)
            exponent = self.gradient_dict.get("exponent", 1)
            uol = self.gradient_dict.get("uol", 1e-9)
            permittivity_surface = self.gradient_dict.get("permittivity_surface", 1)
            if isinstance(permittivity_surface, (list, tuple, np.ndarray)):
                permittivity_surface = permittivity_surface[energy_index]
            
            python_expression = self._make_gradient_text(energy_index, 
                                                         max_depth=max_depth, 
                                                         exponent=exponent, 
                                                         permittivity_surface=permittivity_surface,
                                                         uol=uol)
            return self._to_jcm_gradient(python_expression)



    def save(self, filename=None):
        """🔹 Save shape to a JSON file."""
        if filename is None:
            filename = f"{self.name}_shape.json"
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"🔸 Shape '{self.name}' saved to {filename}")

    @classmethod
    def from_dict(cls, data):
        nk_raw = data['nk']
        if isinstance(nk_raw, list):
            nk_value = [complex(n.replace(' ', '')) for n in nk_raw]
        elif isinstance(nk_raw, str) and nk_raw.startswith('['):
            # Handle stringified list like "[0.9+0.1j, 1.0+0.0j]"
            nk_value = [complex(n.strip()) for n in nk_raw.strip('[]').split(',')]
        else:
            nk_value = complex(nk_raw.replace(' ', ''))

        return cls(
            name=data['name'],
            domain_id=data['domain_id'],
            priority=data['priority'],
            side_length_constraint=data['side_length_constraint'],
            points=data['points'],
            nk=nk_value,
            boundary=data.get('boundary', ['Transparent','Periodic','Transparent','Periodic'])
        )


    @classmethod
    def load(cls, filename):
        """🔹 Load a Shape from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        shape = cls.from_dict(data)
        print(f"🔸 Shape '{shape.name}' loaded from {filename}")
        return shape


        
class Source:
    """
    🔆 Source: A physically grounded illumination object for optical simulation.

    Represents an incident wave defined by:
    • Wavelength (`lam`) in nm, eV, or meters
    • Polarization vector: [1, 0] → S-polarized, [0, 1] → P-polarized
    • Angle of incidence (θ) in degrees
    • Azimuthal angle (phi) in degrees
    • Direction of incidence: 'FromAbove' or 'FromBelow'
    • Type of source: default is 'PlaneWave'

    Automatically converts wavelength to meters and validates input formats.

    Methods:
    • polarization_label(): returns 'S', 'P', or 'Mixed or custom'
    • describe(): narrates all physical parameters in ceremonial format
    • to_jcm(): emits structured simulation-ready block format
    """

    def __init__(self, lam, polarization, angle_of_incidence, phi, incidence='FromAbove', unit='nm', type='PlaneWave',PowerFluxScaling=None):
        allowed = {'FromAbove', 'FromBelow'}
        if incidence not in allowed:
            raise ValueError(f"incidence must be one of {allowed}, got '{incidence}'")

        if not (isinstance(polarization, list) and len(polarization) == 2):
            raise ValueError("polarization must be a list of two numbers")

        if unit == 'nm':
            self.lam = lam * 1e-9
        elif unit == 'eV':
            self.lam = eVnm_converter(lam) * 1e-9
        elif unit == 'm':
            self.lam = lam
        else:
            raise ValueError(f"unit must be 'nm', 'eV', or 'm', got '{unit}'")

        self.polarization = polarization
        self.angle_of_incidence = angle_of_incidence
        self.phi = phi
        self.incidence = incidence
        self.type = type
        self.PowerFluxScaling = PowerFluxScaling

    def polarization_label(self):
        if self.polarization == [1, 0]:
            return 'S'
        elif self.polarization == [0, 1]:
            return 'P'
        else:
            return 'Mixed or custom'

    def describe(self):
        lines = [f"🔆 Source description:"]
        lines.append(f"• Wavelength (m): {self.lam}")
        lines.append(f"• Polarization: {self.polarization} → {self.polarization_label()}-polarized")
        lines.append(f"• Angle of incidence (deg): {self.angle_of_incidence}°")
        lines.append(f"• Azimuthal angle (phi) (deg): {self.phi}°")
        lines.append(f"• Incidence direction: {self.incidence}")
        lines.append(f"• Type: {self.type}")
        if self.PowerFluxScaling is not None:
            lines.append(f"• PowerFluxScaling: {self.PowerFluxScaling}")
        return "\n".join(lines)

    def to_jcm(self):
        block = f"""
    SourceBag {{
    Source {{
    ElectricFieldStrength {{
        {self.type} {{
        Lambda0 = {self.lam}
        SP = [{self.polarization[0]} {self.polarization[1]}]
        ThetaPhi = [{self.angle_of_incidence}, {self.phi}]
        3DTo2D = yes
        Incidence = {self.incidence}"""
        
        if self.PowerFluxScaling is not None:
            block += f"\n      PowerFluxScaling = {self.PowerFluxScaling}"

        block += f"""
        }}
    }}
    }}
    }}
    """
        return block
    

class Cartesian:
    """
    📐 Cartesian grid definition for field export.

    You must specify *either*:
    • NGridPointsX / NGridPointsY (discrete grid definition), OR
    • Spacing (uniform spacing in meters)

    Not both at the same time.
    """

    def __init__(self, spacing=None, n_grid_points_x=None, n_grid_points_y=None):
        if spacing is not None and (n_grid_points_x or n_grid_points_y):
            raise ValueError("Specify either spacing OR NGridPoints, not both.")

        if spacing is None and (n_grid_points_x is None or n_grid_points_y is None):
            raise ValueError("If spacing is not given, both NGridPointsX and NGridPointsY must be provided.")

        self.spacing = spacing
        self.n_grid_points_x = n_grid_points_x
        self.n_grid_points_y = n_grid_points_y

    def describe(self):
        if self.spacing is not None:
            return f"📐 Cartesian grid with spacing = {self.spacing} m"
        else:
            return f"📐 Cartesian grid with NGridPointsX={self.n_grid_points_x}, NGridPointsY={self.n_grid_points_y}"

    def to_dict(self):
        if self.spacing is not None:
            return {"Cartesian": {"Spacing": self.spacing}}
        else:
            return {
                "Cartesian": {
                    "NGridPointsX": self.n_grid_points_x,
                    "NGridPointsY": self.n_grid_points_y,
                }
            }
        
    def to_jcm(self, indent=2):
        pad = " " * indent
        lines = [f"{pad}Cartesian {{"]

        if self.spacing is not None:
            lines.append(f"{pad}  Spacing = {self.spacing}")
        else:
            lines.append(f"{pad}  NGridPointsX = {self.n_grid_points_x}")
            lines.append(f"{pad}  NGridPointsY = {self.n_grid_points_y}")

        lines.append(f"{pad}}}")
        return "\n".join(lines)


class PostProcess:
    """
    🌀 PostProcess: Ritual container for simulation field analysis.

    Two distinct modes are supported:

    • ExportFields:
        - field_bag_path (str)
        - output_file_name (str)
        - output_quantity (str)
        - domain_ids (list[int], optional)
        - cartesian (dict, optional) e.g. {"Spacing": 0.1e-9}

    • FourierTransform:
        - field_bag_path (str)
        - output_file_name (str)
        - normal_direction (str: 'X','Y','Z')
        - rotation (str, optional)

    Methods:
    • describe(): narrates the chosen post-process in ceremonial format
    """

    def __init__(self, mode,field_bag_path,output_file_name, **kwargs):
        allowed_modes = {"ExportFields", "FourierTransform"}
        if mode not in allowed_modes:
            raise ValueError(f"mode must be one of {allowed_modes}, got '{mode}'")

        self.mode = mode
        self.field_bag_path = field_bag_path
        self.output_file_name = output_file_name

        if mode == "ExportFields":
            required = ["output_quantity"]
            for r in required:
                if r not in kwargs:
                    raise ValueError(f"Missing required parameter '{r}' for ExportFields")

            self.output_quantity = kwargs["output_quantity"]
            self.domain_ids = kwargs.get("domain_ids")
            self.cartesian = kwargs.get("cartesian")

        elif mode == "FourierTransform":
            self.normal_direction = kwargs.get("normal_direction")
            self.rotation = kwargs.get("rotation")
            self.numerical_aperture = kwargs.get("numerical_aperture")

    def describe(self):
        lines = [f"🌀 PostProcess description:"]
        lines.append(f"• Mode: {self.mode}")
        lines.append(f"• FieldBagPath: {self.field_bag_path}")
        lines.append(f"• OutputFileName: {self.output_file_name}")

        if self.mode == "ExportFields":
            lines.append(f"• OutputQuantity: {self.output_quantity}")
            if self.domain_ids:
                lines.append(f"• DomainIds: {self.domain_ids}")
            if self.cartesian:
                lines.append(self.cartesian.describe())

        elif self.mode == "FourierTransform":
            lines.append(f"• NormalDirection: {self.normal_direction}")
            if self.rotation:
                lines.append(f"• Rotation: {self.rotation}")

        return "\n".join(lines)

    def to_jcm(self, indent=0):
        pad = " " * indent
        lines = [f"{pad}PostProcess {{", f"{pad}  {self.mode} {{"]

        # Shared fields
        lines.append(f'{pad}    FieldBagPath = "{self.field_bag_path}"')
        lines.append(f'{pad}    OutputFileName = "{self.output_file_name}"')

        if self.mode == "ExportFields":
            lines.append(f'{pad}    OutputQuantity = "{self.output_quantity}"')
            if self.domain_ids:
                ids = ", ".join(map(str, self.domain_ids))
                lines.append(f"{pad}    DomainIds = [{ids}]")
            if self.cartesian:
                lines.append(self.cartesian.to_jcm(indent + 4))

        elif self.mode == "FourierTransform":
            if self.normal_direction:
                lines.append(f"{pad}    NormalDirection = {self.normal_direction}")
            if self.rotation:
                lines.append(f"{pad}    Rotation = {self.rotation}")
            if self.numerical_aperture:
                lines.append(f"{pad}    NumericalAperture = {self.numerical_aperture}")
        lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
    

class ComputationalCosts:
    def __init__(self, title: str, header: Dict[str, Any], **kwargs):
        self.title = title
        self.header = header
        self.data = kwargs  # arrays like CpuTime, Unknowns, etc.

    def summary(self) -> str:
        return (
            f"📊 {self.title}: "
            f"CPU={self.header.get('AccumulatedCPUTime', 'N/A'):.2f}s, "
            f"Total={self.header.get('AccumulatedTotalTime', 'N/A'):.2f}s, "
            f"Unknowns={self.data.get('Unknowns', ['?'])[0]}"
        )

class FieldData:
    def __init__(self, field, grid, X, Y, Z, header):
        self.field = field
        self.grid = grid
        self.X, self.Y, self.Z = X, Y, Z
        self.header = header

    def shape(self):
        return self.field[0].shape if self.field else None

    def summary(self) -> str:
        return (
            f"🌐 FieldData: Quantity={self.header.get('QuantityType')}, "
            f"Shape={self.shape()}, "
            f"Grid points={self.X.shape}"
        )

    def intensity(self, index=0):
        """Compute intensity = |E|^2 from complex field."""
        amplitude = self.field[index]  # shape (Nx, Ny, 3)
        return (amplitude.conj() * amplitude).sum(2).real

    def plot_field(self, index=0, log=True, cmap="viridis", scale=1e9, ax=None):
        """
        Plot the field intensity on the XY grid.

        Parameters:
        • index: which field array to use (default 0)
        • log: whether to plot log(intensity)
        • cmap: matplotlib colormap
        • scale: scaling factor for axes (default 1e9 → nm)
        • ax: optional matplotlib axis to plot into. If None, a new figure/axis is created.
        """
        intensity = self.intensity(index)
        Z = np.log(intensity) if log else intensity

        # Create new fig/ax if none provided
        created_fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            created_fig = fig

        mesh = ax.pcolormesh(
            self.X * scale,
            self.Y * scale,
            Z,
            cmap=cmap,
            shading="auto"
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("log(Intensity)" if log else "Intensity")
        ax.set_xlabel(f"X [{'nm' if scale==1e9 else 'm'}]")
        ax.set_ylabel(f"Y [{'nm' if scale==1e9 else 'm'}]")
        ax.set_title(f"Field intensity ({self.header.get('QuantityType')})")
        ax.set_aspect("equal")

        if created_fig is not None:
            plt.tight_layout()
            return created_fig, ax
        else:
            return ax

    def to_dataframe(self, index=0, log=False):
        """Export field data to a Pandas DataFrame."""
        intensity = self.intensity(index)
        values = np.log(intensity) if log else intensity

        df = pd.DataFrame({
            "X": self.X.flatten(),
            "Y": self.Y.flatten(),
            "Z": self.Z.flatten(),
            "Intensity": values.flatten()
        })
        # Add header metadata as extra columns if desired
        for k, v in self.header.items():
            df[k] = v
        return df
    
    def save(self, filename):
        """Save FieldData to a .npz file with header pickled."""
        np.savez_compressed(
            filename,
            field=self.field,
            grid=self.grid,
            X=self.X,
            Y=self.Y,
            Z=self.Z,
            header=pickle.dumps(self.header)
        )

    @classmethod
    def load(cls, filename):
        """Load FieldData from a .npz file."""
        data = np.load(filename, allow_pickle=True)
        field = data["field"]
        grid = data["grid"]
        X, Y, Z = data["X"], data["Y"], data["Z"]
        header = pickle.loads(data["header"].item())
        return cls(field, grid, X, Y, Z, header)




class FourierCoefficients:
    def __init__(self, title: str, header: dict, **kwargs):
        self.title = title
        self.header = header
        self.data = kwargs  # contains K, N1, N2, ElectricFieldStrength, etc.

    def summary(self) -> str:
        return (
            f"🔊 {self.title}: NormalDirection={self.header.get('NormalDirection')}, "
            f"K={self.data.get('K').shape if 'K' in self.data else None}"
        )

    def compute_order_intensities(self, orders_uni=(-1, 0, 1)):
        """maybe don't us this"""
        orders_uni = np.array(orders_uni)

        orders = self.data["N1"]
        K = self.data["K"]
        E = self.data["ElectricFieldStrength"][0]  # shape (nOrders, 3)

        intensity = np.abs(E[:, 2]) ** 2

        zero_order = orders.searchsorted(0)
        k_in = K[zero_order]
        k_norm = np.linalg.norm(k_in)

        cos_theta_in = np.abs(k_in[1]) / k_norm
        cos_theta_out = K[:, 1] / k_norm

        intensity_corrected = intensity * cos_theta_out / cos_theta_in

        raw_out, cor_out, k_vals = [], [], []

        for order in orders_uni:
            x_o = np.where(orders == order)[0]
            if x_o.size > 0:
                idx = x_o[0]
                raw_out.append(intensity[idx].real)
                cor_out.append(intensity_corrected[idx].real)
                k_vals.append(K[idx, 1])  # take y-component as propagation axis
            else:
                raw_out.append(0.0)
                cor_out.append(0.0)
                k_vals.append(np.nan)

        return {
            "orders": orders_uni,
            "raw": np.array(raw_out),
            "corrected": np.array(cor_out),
            "K": np.array(k_vals),
        }
    
    def to_dataframe(self):
        dfs = []

        # Precompute constants
        eps0 = 8.85418781762039e-12
        mu0  = 1.25663706143592e-06

        eps = np.real(self.header["RelPermittivity"] * eps0)
        mu  = np.real(self.header["RelPermeability"] * mu0)

        # Determine factor depending on field type
        field_type = "ElectricFieldStrength"
        factor = 0.5 * np.sqrt(eps / mu)

        for key in self.data["ElectricFieldStrength"]:

            # --- Extract fields ---
            E = self.data["ElectricFieldStrength"][key]      # shape (N, 3)
            K = self.data["K"]                               # shape (N, 3)
            k_in = self.header["IncomingPlaneWaveKVector"][key]

            Kx, Ky, Kz = K[:, 0], K[:, 1], K[:, 2]
            Kx_in, Ky_in, Kz_in = k_in[0], k_in[1], k_in[2]

            amp_x, amp_y, amp_z = E[:, 0], E[:, 1], E[:, 2]

            # --- Intensity ---
            intensity = (
                (amp_x.conj() * amp_x).real +
                (amp_y.conj() * amp_y).real +
                (amp_z.conj() * amp_z).real
            )

            n_orders = len(K)

            # --- Compute power flux (your convert2powerflux logic) ---
            k_norm = np.linalg.norm(K, axis=1)  # |k|
            nfield = np.sum(np.abs(E)**2, axis=1) / k_norm
            kron = np.kron(np.ones((3,1)), nfield).T   # shape (N, 3)
            power_flux_vec = factor * kron * K         # shape (N, 3)

            # Components of power flux
            P_x = power_flux_vec[:, 0]
            P_y = power_flux_vec[:, 1]
            P_z = power_flux_vec[:, 2]

            # --- Build dataframe ---
            df = pd.DataFrame({
                "key": np.full(n_orders, key),
                "Kx": Kx,
                "Ky": Ky,
                "Kz": Kz,
                "Kx_in": np.full(n_orders, Kx_in),
                "Ky_in": np.full(n_orders, Ky_in),
                "Kz_in": np.full(n_orders, Kz_in),
                "amp_x": (amp_x.conj() * amp_x).real,
                "amp_y": (amp_y.conj() * amp_y).real,
                "amp_z": (amp_z.conj() * amp_z).real,
                "Intensity_calc": intensity,
                "P_x": P_x,
                "P_y": P_y,
                "P_z": P_z,
                "P_norm": P_x + P_y + P_z
            })

            # Add diffraction order if available
            if "N1" in self.data and self.data["N1"] is not None:
                df["order"] = self.data["N1"]

            # --- Angular corrections ---
            df["k_norm"] = np.linalg.norm([Kx_in, Ky_in, Kz_in])
            df["cos_theta_in"] = df["Kz_in"] / df["k_norm"]
            df["cos_theta_out"] = df["Kz"] / df["k_norm"]

            with np.errstate(invalid='ignore', divide='ignore'):
                df["cos_phi_out"] = np.sqrt(
                    1 - np.square(np.abs(df["Kx"] - df["Kx_in"]) / df["k_norm"])
                )
                df["Intensity_calc_corrected"] = (
                    df["Intensity_calc"]
                    * df["cos_theta_out"]
                    / df["cos_theta_in"]
                    * df["cos_phi_out"]
                )

            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
    



    def plot_intensities(self, orders_uni=(-1, 0, 1), use_k=False, corrected=True, **kwargs):
        """
        Plot diffraction order intensities.

        Parameters
        ----------
        orders_uni : iterable of int
            Orders to plot.
        use_k : bool
            If True, plot K_y vs intensity. If False, plot order vs intensity.
        corrected : bool
            If True, plot corrected intensities. If False, raw.
        kwargs : passed to plt.bar or plt.plot
        """
        res = self.compute_order_intensities(orders_uni)
        y = res["corrected"] if corrected else res["raw"]

        if use_k:
            x = res["K"]
            xlabel = "K_y (1/m)"
        else:
            x = res["orders"]
            xlabel = "Diffraction Order"

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y ,'.-', **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Intensity")
        ax.set_title(f"{self.title} ({'corrected' if corrected else 'raw'})")
        plt.tight_layout()
        return fig, ax


class SimulationResult:
    """
    🌀 SimulationResult: container for JCMwave postprocess outputs.
    """

    def __init__(self, file: str,
                 computational_costs: Optional[ComputationalCosts] = None,
                 field_data: Optional[List[FieldData]] = None,
                 fourier: Optional[List[FourierCoefficients]] = None):
        self.file = file
        self.computational_costs = computational_costs
        self.field_data = field_data or []   # list of FieldData
        self.fourier = fourier or []         # list of FourierCoefficients

    def summary(self) -> str:
        lines = [f"📂 SimulationResult from {self.file}"]
        if self.computational_costs:
            lines.append(self.computational_costs.summary())
        for i, fd in enumerate(self.field_data):
            lines.append(f"FieldData[{i}]: {fd.summary()}")
        for i, ft in enumerate(self.fourier):
            lines.append(f"Fourier[{i}]: {ft.summary()}")
        return "\n".join(lines)

    @classmethod
    def from_raw(cls, raw: list):
        comp = ComputationalCosts(**raw[0]["computational_costs"]) if "computational_costs" in raw[0] else None

        field_blocks = []
        fourier_blocks = []

        # loop through all blocks after the first
        for block in raw[1:]:
            if "field" in block:  # FieldData block
                field_blocks.append(FieldData(
                    field=block["field"],
                    grid=block["grid"],
                    X=block["X"],
                    Y=block["Y"],
                    Z=block["Z"],
                    header=block["header"]
                ))
            elif "title" in block and "ElectricFieldStrength" in block:
                fourier_blocks.append(FourierCoefficients(
                    title=block["title"],
                    header=block["header"],
                    K=block["K"],
                    ElectricFieldStrength=block["ElectricFieldStrength"],
                    **{k: block[k] for k in ("N1", "N2") if k in block}
                ))


        return cls(
            file=raw[0].get("file", "unknown"),
            computational_costs=comp,
            field_data=field_blocks,
            fourier=fourier_blocks
        )


    @classmethod
    def from_list(cls, raws: list):
        """Build a list of SimulationResult objects from a list of raw results."""
        return [cls.from_raw(r) for r in raws if r and len(r) > 0]

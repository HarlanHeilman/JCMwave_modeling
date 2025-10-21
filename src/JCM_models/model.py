import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
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

    def __init__(self, name, domain_id, priority, side_length_constraint, points,nk,boundary = ['Transparent','Periodic','Transparent','Periodic']):
        self.name = name
        self.domain_id = domain_id
        self.priority = priority
        self.side_length_constraint = side_length_constraint
        self.points = points
        self.nk = nk
        self.boundary = boundary

        self.permittivity = np.square(self.nk)

    def describe(self):
        return f"""Shape: {self.name}
  Domain ID: {self.domain_id}
  Priority: {self.priority}
  Side Length Constraint: {self.side_length_constraint}
  Refractive Index (nk): {self.nk}
  Permittivity (ε): {self.permittivity}
"""
    def plot(self, ax=None, **kwargs):
        x = self.points[::2]
        y = self.points[1::2]
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
        
    def plot_colored_geometry(self, ax=None, **kwargs):
        points = np.asarray(self.points).reshape(-1, 2)
        polygon = Polygon(
            points,
            closed=True,
            facecolor=colors20[self.domain_id-1],
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
        block = f"""Source {{
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
    }}"""
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

    def plot_field(self, index=0, log=True, cmap="viridis", scale=1e9):
        """
        Plot the field intensity on the XY grid.

        Parameters:
        • index: which field array to use (default 0)
        • log: whether to plot log(intensity)
        • cmap: matplotlib colormap
        • scale: scaling factor for axes (default 1e9 → nm)
        """
        intensity = self.intensity(index)
        Z = np.log(intensity) if log else intensity

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        mesh = ax.pcolormesh(
            self.X * scale,
            self.Y * scale,
            Z,
            cmap=cmap,
            shading="auto"
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("log(Intensity)" if log else "Intensity")
        ax.set_xlabel(f"X [{ 'nm' if scale==1e9 else 'm'} ]")
        ax.set_ylabel(f"Y [{ 'nm' if scale==1e9 else 'm'} ]")
        ax.set_title(f"Field intensity ({self.header.get('QuantityType')})")
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig, ax


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
        K = self.data["K"]
        E = self.data["ElectricFieldStrength"][0]

        Kx, Ky, Kz = K[:, 0], K[:, 1], K[:, 2]
        k_in = self.header["IncomingPlaneWaveKVector"][0]
        Kx_in, Ky_in, Kz_in = k_in[0], k_in[1], k_in[2]

        amp_x, amp_y, amp_z = E[:, 0], E[:, 1], E[:, 2]
        intensity = (amp_x.conj() * amp_x).real + \
                    (amp_y.conj() * amp_y).real + \
                    (amp_z.conj() * amp_z).real

        n_orders = len(K)

        df = pd.DataFrame({
            "Kx": Kx,
            "Ky": Ky,
            "Kz": Kz,
            "Kx_in": np.full(n_orders, Kx_in),
            "Ky_in": np.full(n_orders, Ky_in),
            "Kz_in": np.full(n_orders, Kz_in),
            "amp_x": amp_x,
            "amp_y": amp_y,
            "amp_z": amp_z,
            "Intensity_calc": intensity
        })

        if "N1" in self.data and self.data["N1"] is not None:
            df["order"] = self.data["N1"]

        df["k_norm"] = df.apply(lambda row: np.linalg.norm([row["Kx_in"], row["Ky_in"], row["Kz_in"]]), axis=1)
        df["cos_theta_in"] = df["Kz_in"] / df["k_norm"]
        df["cos_theta_out"] = df["Kz"] / df["k_norm"]

        with np.errstate(invalid='ignore', divide='ignore'):
            df["cos_phi_out"] = np.sqrt(1 - np.square(np.abs(df["Kx"] - df["Kx_in"]) / df["k_norm"]))
            df["Intensity_calc_corrected"] = (
                df["Intensity_calc"] * df["cos_theta_out"] / df["cos_theta_in"] * df["cos_phi_out"]
            )

        return df



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

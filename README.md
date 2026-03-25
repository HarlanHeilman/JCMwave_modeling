# JCMwave_modeling
This is a python wrapper for JCMsuite to more easly set up and start JCM calculations.
You can find a helpful notebook (start.ipynb) in the notebooks folder.
You need to generate a dictonary (for example keys) to run JCMsuite.
keys = {'uol1': 1e-9,
        'fem_deg': 4,}
keys['shape'] = shape
keys['source'] = [s_eV]
keys['postprocess']=[pp1,pp2]
You can generate the JCMfoler by running "write_project_files("JCMfolder_test")" in python.


## Use the ShapeGenerator to generate a shape
ShapeGenerator: A modular geometry engine for parametric shape creation.

    Supports multiple shape types including:
    - Rectangle
    - Trapezoid
    - Stacked trapezoids
    - B-splines
    - polygon from JCMwave

    Features:
    • Flexible parameter dictionary per shape
    • Optional offset for x and y positioning
    • Corner rounding via arc interpolation
    • Centering and flattening utilities
    • Matplotlib-based plotting
    • Ceremonial narration via `.describe()`

    Shape-specific parameters:
    - Rectangle: height, width
    - Trapezoid: height, width, side_angle_deg
    - Stacked trapezoids: height (list), width (list of len+1)
    - B-splines: control_points, num_points
    - Polygon: points

    Example:
        sg = ShapeGenerator('rectangle', {'height': 10, 'width': 20})
        sg.plot()
        print(sg.describe())

## Generate a Shape without ShapeGenerator
Generate a Shape class:
    Shape: A domain-aware polygonal object with optical properties.

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

## Source Class
Source: A physically grounded illumination object for optical simulation.

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

## PostProcess Class
PostProcess: Ritual container for simulation field analysis.

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

### Cartesian Class
Cartesian grid definition for field export.

    You must specify *either*:
    • NGridPointsX / NGridPointsY (discrete grid definition), OR
    • Spacing (uniform spacing in meters)

    Not both at the same time.

## To save the Simulation results into a df or plot them directly: SimulationResult
SimulationResult: container for JCMwave postprocess outputs.
Works with the output of JCMsuite

all_dfs = []

# Suppose you have a list of SimulationResult objects
sim_results = SimulationResult.from_list(results)  

for i, res in enumerate(sim_results):
    print(res.summary())
    print("-" * 40)

    # Plot first field export
    #fd = res.field_data[0]
    #fd.plot_field(index=0, log=True, cmap="plasma")

    # Fourier analysis
    fo = res.fourier[0].compute_order_intensities(orders_uni=[-1, 0, 1])
    #print("Orders:", fo["orders"])
    #print("Raw intensities:", fo["raw"])
    #print("Corrected intensities:", fo["corrected"])

    # Get DataFrame
    df = res.fourier[0].to_dataframe()

    # Add theta column from your external keys array
    df["theta"] = keys["Theta_unique"][i]

    # Collect
    all_dfs.append(df)

# Concatenate into one big DataFrame
df_all = pd.concat(all_dfs, ignore_index=True)

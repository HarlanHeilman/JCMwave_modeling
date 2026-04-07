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

## Installation 

1. git clone repo
```shell
git clone https://github.com/kasandrle/JCMwave_modeling.git
cd JCMwave_modeling
```
2. Create new environment
```shell
uv venv --python=3.12 # project is valid for python >= 3.12
# for windows activate your environment using 
.venv/Scripts/activate 
# for linux/posix activate your environment using
source .venv/bin/activate
```
3. Install project
```shell
uv sync # this will sync the project as editable as if you ran pip install -e . 
```


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
    - Stacked trapezoids (stack_trapezoids): height (list), width (list of len+1)
    - B-splines (bsplines): control_points, num_points
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

### Gradient
The Gradient is calculated like this:

    def build_surface_gradient(points, n_surface, n_bulk, max_depth, exponent=1.0):
        """
        Create a gradient refractive-index function that transitions from the
        polygon boundary inward.

        Parameters
        ----------
        points : array-like
            Flattened polygon coordinates [x0, y0, x1, y1, ...].
        n_surface : complex
            Refractive index at the boundary.
        n_bulk : complex
            Refractive index deep inside the material.
        max_depth : float
            Depth over which the transition occurs.
        exponent : float
            Controls steepness (1 = linear, 2 = quadratic, etc.)

        Returns
        -------
        gradient_fn : function
            A function f(x, y) → complex refractive index.
        """

        pts = np.asarray(points).reshape(-1, 2)

        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            """Distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
            vx, vy = x2 - x1, y2 - y1
            wx, wy = px - x1, py - y1

            seg_len2 = vx*vx + vy*vy
            if seg_len2 == 0:
                return np.hypot(px - x1, py - y1)

            t = (wx*vx + wy*vy) / seg_len2
            t = np.clip(t, 0.0, 1.0)

            proj_x = x1 + t * vx
            proj_y = y1 + t * vy

            return np.hypot(px - proj_x, py - proj_y)

        def min_distance_to_polygon(px, py):
            """Minimum distance from point to polygon boundary."""
            dists = []
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                d = point_to_segment_distance(px, py, x1, y1, x2, y2)
                dists.append(d)
            return min(dists)

        def gradient_fn(x, y):
            """Return nk(x,y) with a smooth inward gradient."""
            d = min_distance_to_polygon(x, y)

            if d >= max_depth:
                return n_bulk

            t = (d / max_depth) ** exponent
            return n_bulk + (n_surface - n_bulk) * (1 - t)

        return gradient_fn

## Source Class
Source: A physically grounded illumination object for optical simulation.

    Represents an incident wave defined by:
    • Wavelength (`lam`) in nm, eV, or meters
    • Polarization vector: [1, 0] → S-polarized, [0, 1] → P-polarized
    • Angle of incidence (θ) in degrees (0 deg = Normal Incidence)
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

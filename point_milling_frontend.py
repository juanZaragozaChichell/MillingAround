r"""Frontend helpers for the point-milling backend.

This module calls into :mod:`point_milling_backend` to construct higher-level
geometric and machining objects.
"""

from math import comb

import numpy as np
import open3d as o3d
from scipy.integrate import quad as integral
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, approx_fprime
from scipy.spatial import Delaunay

import point_milling_backend as pmb
#%% auxiliar functions
def bernstein_cubic(i, t):
    r"""Evaluate the cubic Bernstein polynomial :math:`B_i^3(t)`.

    Parameters
    ----------
    i : int
        Polynomial index in ``{0, 1, 2, 3}``.
    t : float or np.ndarray
        Evaluation point(s) in the unit interval.

    Returns
    -------
    float or np.ndarray
        Value of the polynomial at ``t``.
    """
    return comb(3, i) * (t**i) * ((1 - t)**(3 - i))

def fit_bicubic_bezier_height(u_vals, v_vals, z_vals):
    r"""Fit a bicubic Bézier surface :math:`z = S(u, v)` to data samples.

    Parameters
    ----------
    u_vals, v_vals : 1D arrays of length M, each in [0,1]
        The 'x' and 'y' coordinates (parameters).
    z_vals : 1D array of length M
        The 'z' (height) values.

    Returns
    -------
    c : 2D NumPy array of shape (4,4)
        The 16 fitted scalar coefficients c_{i,j}.
        So the fitted surface is:
            z = sum_{i=0..3} sum_{j=0..3} c[i,j]*B_i(u)*B_j(v).

    P : np.ndarray of shape (4, 4, 3)
        Control net points ``(i/3, j/3, c[i, j])`` composing the bicubic patch.
    """
    M = len(u_vals)
    assert len(v_vals) == M and len(z_vals) == M
    
    # Build the "design" matrix A (size Mx16)
    # A[k, index(i,j)] = B_i(u_k) * B_j(v_k),  for i,j in {0..3}.
    A = np.zeros((M, 16))
    for k in range(M):
        u, v = u_vals[k], v_vals[k]
        col = 0
        for i in range(4):
            Bi = bernstein_cubic(i, u)
            for j in range(4):
                Bj = bernstein_cubic(j, v)
                A[k, col] = Bi * Bj
                col += 1
    
    # Right-hand side is just the z-coordinates
    b = z_vals.reshape(-1, 1)  # shape (M,1)
    
    # Solve in a least squares sense for the 16 unknown coefficients
    c_flat, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    # c_flat is a 16x1 vector
    
    # Reshape into (4,4)
    c = c_flat.reshape(4, 4)
    
    # Build the 3D control points: (i/3, j/3, c[i,j])
    P = np.zeros((4,4,3))
    for i in range(4):
        for j in range(4):
            P[i, j, 0] = i / 3.0
            P[i, j, 1] = j / 3.0
            P[i, j, 2] = c[i, j]
    
    return c, P
    
def flatten_data(data):
    r"""Flatten an array of arrays and record segment lengths.

    Parameters
    ----------
    data : array-like
        An array of arrays, where each sub-array has shape (Li, 2).

    Returns
    -------
    tuple of (np.ndarray, tuple)
        The flattened array of shape ``(N, 2)`` and the tuple with each
        sub-array length.
    """
    flattened_array = np.vstack(data).astype('float64')
    lengths = tuple(len(sub_array) for sub_array in data)
    return flattened_array, lengths

def reconstruct_data(flattened_array, lengths):
    r"""Reconstruct nested arrays using flattened data and stored lengths.

    Parameters
    ----------
    flattened_array : numpy.ndarray
        The concatenated array of shape (N, 2).
    lengths : tuple
        A tuple containing the lengths of each sub-array.

    Returns
    -------
    reconstructed : np.ndarray
        Object-dtyped numpy array containing the recovered sub-arrays.
    """
    reconstructed = []
    idx = 0
    for L in lengths:
        sub_array = flattened_array[idx:idx+L]
        reconstructed.append(sub_array)
        idx += L
    return np.array(reconstructed, dtype=object)
def mesh_to_pickable(mesh):
    r"""Convert an :mod:`open3d` triangle mesh into NumPy arrays.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Open3D triangular mesh.

    Returns
    -------
    tuple
        Tuple containing both vertices and triangles as NumPy arrays.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_and_triangles = (vertices, triangles)
    return vertices_and_triangles

def pickable_to_mesh(data):
    r"""Create an Open3D triangle mesh object from vertex and triangle data.

    Parameters
    ----------
    data : tuple
        A tuple containing vertices (numpy.ndarray) and triangles (numpy.ndarray).

    Returns
    -------
    tuple
        A tuple containing the legacy mesh and its associated raycasting scene.
    """
    vertices, triangles = data
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_triangle_normals()
    mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh2)
    return (mesh, scene)

def max_value_for_envelope(env):
    r"""Return the time where the contact curve reaches length ``2 * R``.

    Parameters
    ----------
    env : pmb.Envelope
        Envelope object describing the contact curve.

    Returns
    -------
    float
        Time value for which the circle is completely outside the contact region.
    """
    def curve_tangent(t):
        v = approx_fprime(t, env.curve_flat).flatten()
        return v
    curv_tang_vec = np.vectorize(curve_tangent, signature='()->(n)')
    def norm_tangent(t):
        return np.linalg.norm(curve_tangent(t))
    norm_tangent_vect = np.vectorize(norm_tangent)
    # length_at_t = np.vectorize(lambda t : integral(norm_tangent_vect, 0, t)[0])
    length_at_t_from_1 = np.vectorize(lambda t : integral(norm_tangent_vect, 1, t)[0])
    f_to_solve_bis = np.vectorize(lambda t: length_at_t_from_1(t) - 2*env.R)
    t0 = 1
    while np.abs(f_to_solve_bis(t0))>1e-5:
        t0 = t0 - f_to_solve_bis(t0)/norm_tangent(t0)
    return t0

def tiempos_porcentuales(curva):
    r"""Return the percentual arc-length parameter for each polyline point."""
    longitud_de_curva_porcentual = np.array([pmb.length_3D_curve(np.array(curva[0:n], dtype=float))
                                            for n in range(1,1+len(curva))])
    l = longitud_de_curva_porcentual[-1]
    return longitud_de_curva_porcentual/l

def concatenate_meshes(list_of_meshes, list_of_simplices):
    r"""Concatenate multiple triangular meshes into a single mesh.

    This function adjusts the indices of simplices for each mesh based on the
    cumulative count of vertices from all previously concatenated meshes to
    ensure the simplices refer to the correct vertices in the resulting single
    mesh array.

    Parameters
    ----------
    list_of_meshes : list of ndarray
        Each element is a numpy array of vertices (3D points) for one mesh.
    list_of_simplices : list of ndarray
        Each element is a numpy array of simplices (triangles) for one mesh.
        Each simplex is represented as indices pointing to vertices in the corresponding mesh.

    Returns
    -------
    o3d.geometry.TriangleMesh
        An object containing the vertices and triangles of the concatenated mesh.

    Notes
    -----
    This function does not merge vertices or simplices that are identical across different input meshes.
    All vertices and simplices from each mesh are preserved and adjusted as needed.
    """
    # Calculate cumulative vertex counts to adjust simplices
    vertex_offsets = np.cumsum([0] + [mesh.shape[0] for mesh in list_of_meshes[:-1]])

    # Adjust simplices with the vertex offsets
    adjusted_simplices = [simplices + offset for simplices, offset in zip(list_of_simplices, vertex_offsets)]

    # Stack all vertices and simplices
    concatenated_meshes = np.vstack(list_of_meshes)
    concatenated_simplices = np.vstack(adjusted_simplices)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(concatenated_meshes.astype(np.float32))
    new_mesh.triangles = o3d.utility.Vector3iVector(concatenated_simplices)

    return new_mesh

def angles_times_and_simplices_all_equal(resolution):
    r"""Compute the Delaunay triangulation of ``[0, 2π] x [0, 1]``.

    The rectangular domain is discretized using ``resolution`` samples per axis
    and triangulated via :class:`scipy.spatial.Delaunay`.

    Parameters
    ----------
    resolution : int
        The number of points in each dimension of the grid. Higher values lead to finer 
        discretization and more detailed triangulation.

    Returns
    -------
    tuple
        ``(angles, times, simplices)`` arrays describing the grid and its
        triangulation.
    """
    angles = np.linspace(0, 2*np.pi, resolution)
    times = np.linspace(0,1,resolution)
    angles1,times1=np.meshgrid(angles,times)
    angles1=angles1.flatten()
    times1=times1.flatten()
    points2D=np.vstack([angles1, times1]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices
    return (angles, times, simplices)

def curves_to_same_orientation(set_of_curves):
    r"""Reorient all curves in the given set to share the same direction.

    This function ensures that all curves in the set follow the same orientation direction as the first curve
    based on their tangent directions. The orientation is determined by the direction from the first to the
    second point of each curve. If a curve's orientation is opposite to the first curve, it is reversed.

    Parameters
    ----------
    set_of_curves : list of np.ndarray
        A list of curves, where each curve is represented as a numpy array of points.

    Returns
    -------
    list of np.ndarray
        Curves aligned with the direction of the first entry in ``set_of_curves``.
    """
    set_of_curves_copy = set_of_curves.copy()
    first_curve = set_of_curves[0]
    first_tangent_direction = first_curve[1] - first_curve[0]
    for i,curva in enumerate(set_of_curves):
        if np.dot(first_tangent_direction, curva[1] - curva[0]) >=0:
            pass
        else:
            set_of_curves_copy[i] = np.flip(curva, axis=0)
    return set_of_curves_copy

#%% class Object
class MachiningParameters:
    r"""Coordinate tool, surface, and sampling parameters for milling studies.

    The class instantiates the analytic surface defined by ``mat_Q``, triangulates both the nominal
    and offset meshes, extracts contact curves via the backend ``G_function``, and constructs the
    corresponding envelopes used for collision detection and error analysis. It centralizes every
    parameter required to rebuild these geometric artefacts so downstream routines can focus on
    simulation, visualization, or verification tasks.

    Attributes
    ----------
    R : float
        Radius of the toroidal cutter's medial circle.
    m : float
        Offset distance (second torus radius) applied when generating the offset surface.
    mat_Q : np.ndarray
        ``4x4`` matrix describing the quadric surface that ``pmb.Surface`` evaluates.
    matrices : np.ndarray
        Stack of three ``4x4`` matrices used respectively for the G-function, ``phi`` rotation, and ``theta`` tilt.
    number_of_paths : int
        Requested number of level sets sampled from the G-function. Multiple components can increase
        the actual number of contact curves.
    h : float
        Safety margin applied when computing minimum/maximum values of ``G`` to avoid degenerate level sets.
    surfaces_resolution : int
        Number of samples per parametric direction when triangulating the surface and its offset.
    machining_tolerance : float
        Precision threshold forwarded to routines that rely on geometric tolerances.
    shank_length : float
        Default physical shank length used when creating discrete shanks.
    n_shanks : int
        Default number of shank samples evaluated along each envelope.

    Methods
    -------
    triangulate(...)
        Return vertices, connectivity, and normals for the surface or its offset.
    G_level_sets()
        Sample consistently oriented level sets from the current ``G_function``.
    uv_curves_parametrized()
        Convert the level-set curves into cubic spline parametrizations for ``u`` and ``v``.
    concatenated_envelopes_fun(...)
        Discretize every envelope and merge them into a single mesh, with several trimming options.
    discrete_shanks_vectorized(...)
        Evaluate shank positions across all envelopes in a vectorized fashion.
    error_measure_per_vertex()
        Raycast surface normals against the envelopes to obtain per-vertex machining error.
    """

    def __init__(self, R, m, mat_Q, matrices,
                number_of_paths, h = 0.1, surfaces_resolution = 100,
                machining_tolerance = 5e-4,
                shank_length = 1+np.sqrt(5),
                n_shanks = 10):
        r"""Instantiate the tooling setup, analytic surface, and derived envelopes.

        The constructor builds the ``pmb.Surface`` object from ``mat_Q``, triangulates both the nominal
        surface and its offset, sets up the requested ``G_function``, and immediately generates the
        envelope objects by parametrizing the level-set curves in ``u`` and ``v``. The resulting instance
        is ready to provide meshes, envelopes, and shank positions without further configuration.

        Parameters
        ----------
        R : float
            Radius of the toroidal cutter's medial circle.
        m : float
            Offset distance applied when generating the auxiliary offset surface.
        mat_Q : np.ndarray
            ``4x4`` matrix describing the quadric evaluated by ``pmb.Surface``.
        matrices : array-like of shape (3, 4, 4)
            Transformation matrices used respectively for the G-function, ``phi`` rotation, and ``theta`` tilt.
        number_of_paths : int
            Number of level sets requested from the ``G_function`` when building envelopes.
        h : float, optional
            Positive margin applied when bracketing the range of ``G`` to avoid degenerate level sets.
        surfaces_resolution : int, optional
            Samples per parametric direction used when triangulating the surface and its offset.
        machining_tolerance : float, optional
            Precision threshold forwarded to routines that rely on geometric tolerances.
        shank_length : float, optional
            Default shank length evaluated when calling :meth:`discrete_shanks_vectorized`.
        n_shanks : int, optional
            Default number of shank samples evaluated along each envelope.

        Notes
        -----
        During initialization the method calls :meth:`triangulate` twice (surface and offset), computes the
        ``angles``, ``times``, and ``simplices`` grids, caches the current ``G_function``, and rebuilds the
        envelopes so the instance stays in a consistent state.
        """
            # fixed machining attributes
            
        self.R = R
        self.m = m
        self.h = h
        self.machining_tolerance = machining_tolerance
        self.surfaces_resolution = surfaces_resolution
        # self.number_of_paths = number_of_paths
        self._number_of_paths = number_of_paths
        self.Range = [[0, 1], [0, 1]]
        self.surface = pmb.Surface(mat_Q=mat_Q)
        self.surface_vertices, self.surface_triangles, self.surface_centroids, self.surface_normals, self.vertex_normals = self.triangulate(
                                                                        object = 'surface',
                                                                        surfaces_resolution=surfaces_resolution)
        self.offset_vertices, self.offset_triangles, self.offset_centroids, self.offset_normals, self.vertex_normals = self.triangulate(
                                                                            object = 'offset',
                                                                            surfaces_resolution=surfaces_resolution)
        
        self.angles, self.times, self.simplices = angles_times_and_simplices_all_equal(self.surfaces_resolution)
        # mutable machining attributes

        self._matrices = matrices
        self._G_fun = pmb.G_function(mat=self._matrices[0])

        u_curves_parametrized, v_curves_parametrized = self.uv_curves_parametrized()
        self._envelopes = [pmb.Envelope(R=self.R, surface=self.surface,
                                        u_of_t=u_curve, v_of_t=v_curve,
                                        mat_phi=self._matrices[1], mat_theta=self._matrices[2])
                           for u_curve, v_curve in zip(u_curves_parametrized, v_curves_parametrized)]
    
    @property
    def number_of_paths(self):
        r"""int: Target number of G-function level sets used when generating envelopes."""
        return self._number_of_paths
    @number_of_paths.setter
    def number_of_paths(self, number_of_paths):
        r"""Update the requested number of level sets and refresh the envelopes."""
        self._number_of_paths = number_of_paths
        self._update_envelopes()


    @property
    def matrices(self):
        r"""Get the current set of transformation matrices used in the machining process.

        Returns
        -------
        np.ndarray
            A 3D numpy array of shape (3, 4, 4) containing the transformation matrices.
        """
        return self._matrices
    @matrices.setter
    def matrices(self, matrices):
        r"""Update the transformation matrices and refresh dependent structures.

        Parameters
        ----------
        matrices : np.ndarray or list
            A numpy array or list that can be converted to a numpy array of shape (3, 4, 4).

        Raises
        ------
        ValueError
            If the provided matrices do not conform to the required shape (3, 4, 4).
        """
        matrices = np.array(matrices)
        if matrices.shape != (3,4,4):
            raise ValueError("matrices is not (3,4,4)")
        self._matrices = matrices
        self._update_G_fun()
        self._update_envelopes()

    @property
    def envelopes(self):
        r"""Get the list of envelope objects currently used in machining operations.

        Returns
        -------
        list[pmb.Envelope]
            Envelope instances defining the machining envelopes.
        """
        return self._envelopes
    def _update_envelopes(self):
        r"""Refresh envelope objects using the current matrices and parametrized curves."""
        u_curves_parametrized, v_curves_parametrized = self.uv_curves_parametrized()
        self._envelopes = [pmb.Envelope(R=self.R, surface=self.surface,
                                        u_of_t=u_curve, v_of_t=v_curve,
                                        mat_phi=self._matrices[1], mat_theta=self._matrices[2])
                           for u_curve, v_curve in zip(u_curves_parametrized, v_curves_parametrized)]
    @property
    def G_fun(self):
        r"""Get the current G-function used for generating machining paths.

        Returns
        -------
        pmb.G_function
            Callable generated from the first transformation matrix.
        """
        return self._G_fun
    def _update_G_fun(self):
        r"""Update the G-function using the first transformation matrix."""
        self._G_fun = pmb.G_function(mat=self._matrices[0])

    def triangulate(self, object, surfaces_resolution):
        r"""Generate mesh data for the requested geometric object.

        Parameters
        ----------
        object : str
            Specifies the type of object to triangulate, either 'surface' or 'offset'.
        surfaces_resolution : int
            The resolution at which to generate the point cloud for triangulation.

        Returns
        -------
        tuple
            ``(points, triangles, centroids, triangle_normals, vertex_normals)`` where:
            - ``points`` (np.ndarray): Array of vertices with shape ``(N, 3)``.
            - ``triangles`` (np.ndarray): Triangle indices referencing ``points``.
            - ``centroids`` (np.ndarray): Triangle centroids, one per face.
            - ``triangle_normals`` (np.ndarray): Normals computed by Open3D per triangle.
            - ``vertex_normals`` (np.ndarray): Normals provided directly by the backend sampler.

        Raises
        ------
        ValueError
            If the 'object' parameter is neither 'surface' nor 'offset'.
        """
        if object == 'surface':
            points, tri, vertex_normals = self.surface.point_cloud(n_points_u = surfaces_resolution,
                                                   n_points_v = surfaces_resolution)
        elif object == 'offset':
            points, tri, vertex_normals = self.surface.offset_pointcloud(m = self.m,
                                                     n_points_u = surfaces_resolution,
                                                     n_points_v = surfaces_resolution)
        else:
            raise ValueError('Wrong object input. Object has to be either "surface" or "offset"')
        
        mesh = o3d.geometry.TriangleMesh()
        points = np.array(points).T # it has to be transposed to work nicely
        tri = tri.triangles # actual triangles
        mesh.vertices = o3d.utility.Vector3dVector(points.astype(np.float32))
        mesh.triangles = o3d.utility.Vector3iVector(tri)
        mesh.compute_triangle_normals()
        # Calculate the centroids of each triangle
        centroids = np.mean(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)], axis=1)
        # mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # scene = o3d.t.geometry.RaycastingScene()
        # mesh_id = scene.add_triangles(mesh2)
        return (points.astype(np.float32), tri, centroids, np.asarray(mesh.triangle_normals), vertex_normals.astype(np.float32))
    
    def G_level_sets(self):
        r"""Compute level sets of the G-function with consistent orientation.

        This method minimizes and maximizes the third component of ``G`` (the height) using
        ``scipy.optimize.differential_evolution`` and then samples ``self._number_of_paths``
        evenly spaced height values within that safe range. Each raw level set is re-oriented
        via :func:`curves_to_same_orientation` so downstream routines can assume coherent
        traversal direction.

        Returns
        -------
        list of np.ndarray
            Each array stores the ``(u, v)`` samples of a contact curve.

        Notes
        -----
        The ``delta`` parameter controls the marching step when extracting level sets; smaller values improve
        fidelity at the expense of runtime.
        """

        # get global minimum and maximum of G and move a bit away from it
        minimium_val = differential_evolution(lambda x : self._G_fun.function(x)[2], [[0,1], [0,1]]).fun + self.h
        max_val = -differential_evolution(lambda x : -self._G_fun.function(x)[2], [[0,1], [0,1]]).fun - self.h
        # level sets are defined as a division of the height of G
        zetas = np.linspace(minimium_val, max_val, self._number_of_paths)
        # take the level sets
        every_curve = self._G_fun.level_sets_multiple_z(zetas, delta=0.01) # if delta is too big you could be creating jumps
        # taking delta small minimizes the chance of jumps. Ideally, the smaller the better.
        # make all directions the same
        every_curve = curves_to_same_orientation(every_curve)
        return every_curve
    
    def uv_curves_parametrized(self):
        r"""Parametrize the ``u`` and ``v`` curves derived from the G-function level sets.

        Each level set is converted into a pseudo arc-length parameter ``t`` (ranging from 0 to 1) via
        :func:`tiempos_porcentuales`. Cubic splines are then fitted independently to the ``u`` and ``v``
        components so they can be evaluated at arbitrary ``t`` values when building envelopes.

        Returns
        -------
        list
            ``[u_curves_parametrized, v_curves_parametrized]`` where each entry is a ``np.ndarray`` of
            :class:`scipy.interpolate.CubicSpline` objects, one per level set.
        """
        every_curve = self.G_level_sets()
        matriz_de_tiempos1 = np.array([tiempos_porcentuales(curva) for curva in every_curve], dtype=object)
        u_vals = np.array([curve[:,0] for curve in every_curve], dtype=object)
        v_vals = np.array([curve[:,1] for curve in every_curve], dtype=object)
        u_curves_parametrized = np.array([CubicSpline(tiempo, puntos)
                                          for tiempo, puntos in zip(matriz_de_tiempos1, u_vals)], dtype=object)
        v_curves_parametrized = np.array([CubicSpline(tiempo, puntos)
                                          for tiempo, puntos in zip(matriz_de_tiempos1, v_vals)], dtype=object)
        return [u_curves_parametrized, v_curves_parametrized]
    
    def concatenated_envelopes_fun(self, method = 'by_excess'):
        r"""Concatenate discretized envelopes with different trimming strategies.

        Parameters
        ----------
        method : str, optional
            ``"by_excess"`` extrapolates each contact curve in ``t`` so the resulting mesh extends slightly
            beyond the machining region (default, useful for error metrics). ``"from0to1"`` samples only the
            original ``t`` interval ``[0, 1]``. ``"beauty"`` uses :func:`max_value_for_envelope` to keep a
            small excess suitable for rendering.

        Returns
        -------
        o3d.geometry.TriangleMesh
            Triangle mesh aggregating all envelopes per the selected method.

        Raises
        ------
        ValueError
            If ``method`` is not one of the supported options.

        Notes
        -----
        ``"by_excess"`` extends the envelopes beyond the machining region (useful for error computation),
        ``"from0to1"`` confines envelopes to the machining region, and ``"beauty"`` leaves a small excess for
        visualization purposes.
        """
        if method == 'by_excess':
            # Extend the envelopes by excess so error computations have a bit of slack.
            length_of_contact_curves = np.array([env.length_of_contact_line for env in self.envelopes])
            max_times =1 + 2*self.R/length_of_contact_curves # estimation of excess parameter.
            initial_times = -self.R/length_of_contact_curves
            #If curve is arc length parametrized, it is precisely that
            discretized_envelopes = np.array([
                env.value_at(self.angles,
                             np.linspace(initial_time, max_time, self.surfaces_resolution),
                             method='whole_envelope').reshape(-1,3) 
                for env, max_time, initial_time in zip(self.envelopes, max_times, initial_times)])
            
            # #? we need also the "tapas" of the envelopes for better perfomance (?)
            # discrete_tapas_begining = np.array([ env.value_at(self.angles, initial_time, method='fixed_arc').reshape(-1,3)  for env, initial_time in zip(self.envelopes, max_times, initial_times)])
            
            # discrete_tapas_end = np.array([ env.value_at(self.angles, max_time, method='fixed_arc').reshape(-1,3) 
            #     for env, max_time in zip(self.envelopes, max_times, initial_times)]) #! this might be useless but just in case

            return concatenate_meshes(discretized_envelopes, np.array([self.simplices for i in range(len(self.envelopes))]))
        elif method == 'from0to1':
            # Restrict the envelopes to the original machining locus.
            discretized_envelopes = np.array([
                env.value_at(self.angles, self.times, method='whole_envelope').reshape(-1,3) 
                for env in self.envelopes])
            return concatenate_meshes(discretized_envelopes, np.array([self.simplices for i in range(len(self.envelopes))]))
        elif method == 'beauty':
            # Leave a small excess based on ``max_value_for_envelope`` for rendering purposes.
            max_times = np.array([max_value_for_envelope(env) for env in self.envelopes])
            discretized_envelopes = np.array([
                env.value_at(self.angles,
                             np.linspace(0, max_time, self.surfaces_resolution),
                             method='whole_envelope').reshape(-1,3) 
                for env, max_time in zip(self.envelopes, max_times)])
            return concatenate_meshes(discretized_envelopes, np.array([self.simplices for i in range(len(self.envelopes))]))
        else:
            raise ValueError(' method variable is wrong.')

    def discrete_shanks_vectorized(self, n_shanks = 25, distance = 1+np.sqrt(5)):
        r"""Compute discrete shank positions in a vectorized form for all envelopes.

        This routine samples ``n_shanks`` for each envelope, calls :meth:`pmb.Envelope.shank_at_t` in a vectorized fashion, and reshapes the result so every row contains the  two endpoints of a shank segment in 3D.

        Parameters:
        ----------
        n_shanks : int, optional
            The number of discrete shank positions to calculate along the envelopes, default is 25.
        distance : float, optional
            The length of the shank, default is 1 + sqrt(5).

        Returns:
        -------
        np.ndarray
            Array of shape ``(len(self.envelopes) * n_shanks, 2, 3)`` where the second dimension stores
            ``(shank_bottom, shank_top)``.
        """
        T = np.linspace(0.01,0.99,n_shanks)
        # T = np.linspace(0,1,n_shanks)
        all_shanks = np.array([env.shank_at_t(T, distance = distance) for env in self.envelopes])
        all_shanks = all_shanks.transpose(0, 2, 1, 3)  # Now shape is (8, 25, 2, 3)
        return  all_shanks.reshape(-1, 2, 3)

    def error_measure_per_vertex(self):
        r"""Measure machining error at each surface vertex via raycasting.

        All envelopes are discretized with ``method='by_excess'`` and inserted into an Open3D tensor
        ``RaycastingScene``. Each vertex normal of the nominal surface is used as a ray direction after
        translating the origin by ``-2 * R`` along that normal so the first intersection corresponds to
        the reference envelope.

        Returns
        -------
        np.ndarray
            Signed distances (one per vertex) after subtracting the ``2 * R`` offset. Rays that do not
            intersect the envelopes receive a large sentinel value (``1000``).
        """
        # create the mesh and convert into the t thing
        mesh = self.concatenated_envelopes_fun(method = 'by_excess')
        mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(mesh2)

        # create the first intersections
        # We move the start of the rays down
        # This way, since (unless something horrible happens) the envelopes cannot be further away than 2*R
        # The first intersection should be the one we are looking for
        
        ray_directions  = self.vertex_normals
        ray_origins = self.surface_vertices - 2*self.R*ray_directions
        C = np.hstack((ray_origins, ray_directions))

        # intersect the gauging
        rays = o3d.core.Tensor(C,
                            dtype=o3d.core.Dtype.Float32)

        ans1 = scene.cast_rays(rays)
        A = ans1['t_hit'].numpy()
                
        # t_hit has the distance from ray_origins - 3*R*ray_directions to the intersection.
        # That means that the intersection point is ray_origins + t_hit * ray_directions
        # intersection_points = ray_origins + A.reshape(-1,1) * ray_directions
        # # Now we have to get the signed distance with respect to the surface
        # signs = np.sign(
        #     check_plane_side(point_to_check=intersection_points,
        #         normal_vector=ray_directions,
        #         point_in_plane=machining_object.surface_centroids)
        #                 )
        
        # A = A*signs
        A = A - 2*self.R 
        # There might be some rights that do not hit the mesh
        # For those points, the distance is infinity.
        # We want to correct that to a high value but not too high.
        distances = np.where(np.isfinite(A), A, 1000)
        return distances
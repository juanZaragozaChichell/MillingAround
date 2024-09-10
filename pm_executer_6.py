"""
This library is a 'front-end' to point_milling_N library, in the sense that 
it is used to call the different geometric objects and construct the actual
machining object.

"""

import numpy as np
import open3d as o3d
from scipy.optimize import differential_evolution
from point_milling_6 import *
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.integrate import quad as integral

#%% auxiliar functions

def mesh_to_pickable(mesh):
    '''
    Given an o3d triangle mesh object, returns its vertices and triangles.
    Parameters:
    ----------
        mesh: open3d triangular mesh
    
    Returns:
    -------
        vertices_and_triangles (tuple): tuple containing both vertices and triangles
    '''
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_and_triangles = (vertices, triangles)
    return vertices_and_triangles

def pickable_to_mesh(data):
    """
    Create an Open3D triangle mesh object from vertex and triangle data.

    Parameters:
    ----------
    data : tuple
        A tuple containing vertices (numpy.ndarray) and triangles (numpy.ndarray).

    Returns:
    -------
    tuple
        A tuple containing the Open3D triangle mesh object and its associated raycasting scene.
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
    '''Returns the time value for which the contact curve of env reaches length 2*R
    Expects an Envelope object as input
    
    Parameters:
    ----------
        env: Envelope Object
    
    Returns:
    -------
        t0 (float): time value for which the circle is totally out'''
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
    """
    Returns the percentual value time of each point in a polyline.

    """
    longitud_de_curva_porcentual = np.array([length_3D_curve(np.array(curva[0:n], dtype=float))
                                            for n in range(1,1+len(curva))])
    l = longitud_de_curva_porcentual[-1]
    return longitud_de_curva_porcentual/l

def concatenate_meshes(list_of_meshes, list_of_simplices):
    """
    Concatenate multiple triangular meshes into a single mesh. This function adjusts the indices of simplices
    for each mesh based on the cumulative count of vertices from all previously concatenated meshes to ensure
    the simplices refer to the correct vertices in the resulting single mesh array.

    Args:
        list_of_meshes (list of ndarray): Each element is a numpy array of vertices (3D points) for one mesh.
        list_of_simplices (list of ndarray): Each element is a numpy array of simplices (triangles) for one mesh.
        Each simplex is represented as indices pointing to vertices in the corresponding mesh.

    Returns:
        o3d.geometry.TriangleMesh: An object containing the vertices and triangles of the concatenated mesh.

    Note:
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
    """
    Computes the Delaunay triangulation of the rectangular domain [0, 2π] x [0, 1]. This function discretizes 
    both the interval [0, 2π] and [0, 1] into 'resolution' number of points each, creating a grid. It then performs
    Delaunay triangulation on these points to determine the simplicial structure of the tesselation.

    Args:
        resolution (int): The number of points in each dimension of the grid. Higher values lead to finer 
        discretization and more detailed triangulation.

    Returns:
        tuple: Contains three elements:
            - np.ndarray: The array of angle discretizations from [0, 2π].
            - np.ndarray: The array of time discretizations from [0, 1].
            - np.ndarray: The array of simplices, where each simplex is represented by indices of points 
            forming the vertices of each triangle in the triangulated mesh.

    Example:
        # To create a triangulation with a resolution of 10
        angles, times, simplices = angles_times_and_simplices_all_equal(10)
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
    """
    Reorients all curves in the given set to have the same directionality.

    This function ensures that all curves in the set follow the same orientation direction as the first curve
    based on their tangent directions. The orientation is determined by the direction from the first to the
    second point of each curve. If a curve's orientation is opposite to the first curve, it is reversed.

    Parameters:
    ----------
    set_of_curves : list of np.ndarray
        A list of curves, where each curve is represented as a numpy array of points.

    Returns:
    -------
    list of np.ndarray
        A list of curves all oriented in the same direction as the first curve in the input list.
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
    """
    Manages and encapsulates machining parameters and operations related to surface and offset geometries.

    This class provides a structured approach to handling various geometric calculations and operations
    required in the setup and execution of CNC machining. It integrates multiple functionalities
    including curve parametrization, envelope generation, and machining simulations.
    The class is designed to interact with external libraries and custom algorithms to provide a
    seamless interface for complex machining tasks.

    Attributes:
    ----------
    R : float
        Radius of the medial circle.
    m : int
        Second radius of the toroidal cutter (or offseting distance)
    mat_Q : np.ndarray
        A matrix representing the geometric properties of the surface being machined.
    matrices : list of np.ndarray
        A list containing 3 4x4 numpy arrays that will represent G-function, rotation and tilt, in that order.
    number_of_paths : int
        Number of paths to be generated in the machining process. Enters as actual number of level sets of the G function,
        so more paths could arise.
    h : float, optional
        Hyperparameter for defining the level sets on G. It is expected that h>0.
    surfaces_resolution : int, optional
        Resolution of the generated surfaces which affects the detail level of the meshes, default is 100.
    machining_tolerance : float, optional
        Tolerance level for the machining operations, default is 0.0005.
    shank_length : float, optional
        Length of the shank in the machining tool, default is 1 + sqrt(5).
    n_shanks : int, optional
        Number of shanks to be considered in the simulation, default is 10.

    Methods:
    -------
    __init__(self, R, m, mat_Q, matrices, number_of_paths, h=0.1, surfaces_resolution=100,
             machining_tolerance=0.0005, shank_length=1+np.sqrt(5), n_shanks=10)
        Initializes the MachiningParameters with specified machining parameters and settings.

    envelopes(self)
        Property that retrieves the current list of envelope objects used in the machining simulations.

    matrices(self)
        Property that retrieves the current transformation matrices used in the machining operations.

    matrices(self, matrices)
        Setter method that updates the transformation matrices and refreshes dependent properties and operations.

    G_fun(self)
        Property that retrieves the G-function used for generating the machining paths.

    triangulate(self, object, surfaces_resolution)
        Generates and returns mesh data for either the surface or the offset based on specified parameters.

    G_level_sets(self)
        Computes and returns the level sets of the G-function, which represent distinct machining paths.

    uv_curves_parametrized(self)
        Parametrizes the u and v curves over time based on the computed level sets of the G-function.

    concatenated_envelopes_fun(self, method='by_excess')
        Concatenates envelope data based on specified method criteria for simulation purposes.

    discrete_shanks_vectorized(self, n_shanks=25, distance=1+np.sqrt(5))
        Calculates and returns a vectorized representation of discrete shanks for detailed analysis.

    error_measure_per_triangle_open3d(self)
        Measures and returns the error per triangle using advanced raycasting techniques in Open3D.
    """

    def __init__(self, R, m, mat_Q, matrices,
                number_of_paths, h = 0.1, surfaces_resolution = 100,
                machining_tolerance = 5e-4,
                shank_length = 1+np.sqrt(5),
                n_shanks = 10):
        """
        Initialize the MachiningParameters object with specified machining settings and geometrical data.

        This constructor sets up the machining environment by initializing attributes related to the tool,
        surface properties, and computational settings necessary for generating and manipulating
        machining geometries and their simulations.

        Parameters:
        ----------
        R : float
            Radius of the medial circle.
        m : int
            Parameter defining the offset from the surface. Used in calculating
            the offset mesh where the tool might interact with the material.
        mat_Q : np.ndarray
            A matrix representing the geometric properties of the surface being machined.
        matrices : list of np.ndarray
            List of matrices for various transformations required during machining. The first matrix is used
            for the G-function, the second for rotation, and the third for tilt.
        number_of_paths : int
            Number of level sets to be considered in the G-function. Translates into, at least, number_of_paths
            contact curves, depending on the actual geometry of the G-function.
        h : float, optional
            Parameter to avoid single-point level sets in the level sets computation.
        surfaces_resolution : int, optional
            Resolution of the surfaces, which affects the detail level of the mesh generated for simulation,
            default is 100.
        machining_tolerance : float, optional
            The tolerance level for the machining operations, specified to control the precision of the
            machining process, default is 0.0005.
        shank_length : float, optional
            The length of the shank in the machining setup, default is 1 + sqrt(5).
        n_shanks : int, optional
            The number of shanks considered in the simulation, which can affect the complexity of the
            machining simulation, default is 10.

        Notes:
        -----
        The initialization also sets up the machining geometry by calling the `triangulate` method to create
        meshes for the surface and offset based on the provided `mat_Q` and calculates angles, times, and simplices
        for use in simulations. It establishes envelopes based on the surfaces and transformations provided.
        """
        # fixed machining attributes
        self.R = R
        self.m = m
        self.h = h
        self.machining_tolerance = 5e-4
        self.surfaces_resolution = surfaces_resolution
        # self.number_of_paths = number_of_paths
        self._number_of_paths = number_of_paths
        self.Range = [[0,1], [0,1]]
        self.surface = Surface(mat_Q=mat_Q)
        self.surface_vertices, self.surface_triangles, self.surface_centroids, self.surface_normals = self.triangulate(
                                                                        object = 'surface',
                                                                        surfaces_resolution=surfaces_resolution)
        self.offset_vertices, self.offset_triangles, self.offset_centroids, self.offset_normals = self.triangulate(
                                                                            object = 'offset',
                                                                            surfaces_resolution=surfaces_resolution)
        
        self.angles, self.times, self.simplices = angles_times_and_simplices_all_equal(self.surfaces_resolution)
        # mutable machining attributes

        self._matrices = matrices
        self._G_fun = G_function(mat = self._matrices[0])

        u_curves_parametrized, v_curves_parametrized = self.uv_curves_parametrized()
        self._envelopes = [Envelope(R = self.R, surface = self.surface,
                                u_of_t = u_curve, v_of_t = v_curve,
                                mat_phi = self._matrices[1], mat_theta = self._matrices[2])
                                for u_curve, v_curve in zip(u_curves_parametrized, v_curves_parametrized)]
    
    @property
    def number_of_paths(self):
        return self._number_of_paths
    @number_of_paths.setter
    def number_of_paths(self, number_of_paths):
        self._number_of_paths = number_of_paths
        self._update_envelopes()


    @property
    def matrices(self):
        """
        Gets the current set of transformation matrices used in the machining operations.

        Returns:
        -------
        np.ndarray
            A 3D numpy array of shape (3, 4, 4) containing the transformation matrices.
        """
        return self._matrices
    @matrices.setter
    def matrices(self, matrices):
        """
        Sets and updates the transformation matrices used for the machining operations. This setter also
        triggers updates to the G-function and envelopes based on the new matrices.

        Parameters:
        ----------
        matrices : np.ndarray or list
            A numpy array or list that can be converted to a numpy array of shape (3, 4, 4).

        Raises:
        -------
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
        """
        Gets the list of envelope objects currently used in the machining operations.

        Returns:
        -------
        list of Envelope
            A list of Envelope objects defining the machining envelopes.
        """
        return self._envelopes
    def _update_envelopes(self):
        """
        Updates the envelopes based on the current transformation matrices and the parametrized curves
        for the machining operations.
        """
        u_curves_parametrized, v_curves_parametrized = self.uv_curves_parametrized()
        self._envelopes = [Envelope(R = self.R, surface = self.surface,
                                u_of_t = u_curve, v_of_t = v_curve,
                                mat_phi = self._matrices[1], mat_theta = self._matrices[2])
                                for u_curve, v_curve in zip(u_curves_parametrized, v_curves_parametrized)]
    @property
    def G_fun(self):
        """
        Gets the current G-function used for generating machining paths.

        Returns:
        -------
        function
            The G-function generated based on the first transformation matrix.
        """
        return self._G_fun
    def _update_G_fun(self):
        """
        Updates the G-function based on the first transformation matrix currently set in the matrices attribute.
        """
        self._G_fun = G_function(mat = self._matrices[0])

    def triangulate(self, object, surfaces_resolution):
        """
        Generates mesh data for specified geometric objects based on current machining parameters.

        Parameters:
        ----------
        object : str
            Specifies the type of object to triangulate, either 'surface' or 'offset'.
        surfaces_resolution : int
            The resolution at which to generate the point cloud for triangulation.

        Returns:
        -------
        tuple
            A tuple containing:
            - points (np.ndarray): Transposed array of points that constitute the mesh vertices.
            - tri (np.ndarray): Array of triangle indices forming the mesh.
            - centroids (np.ndarray): The centroids of each triangle in the mesh.
            - normals (np.ndarray): The normals of each triangle.

        Raises:
        -------
        ValueError
            If the 'object' parameter is neither 'surface' nor 'offset'.
        """
        if object == 'surface':
            points, tri = self.surface.point_cloud(n_points_u = surfaces_resolution,
                                                   n_points_v = surfaces_resolution)
        elif object == 'offset':
            points, tri = self.surface.offset_pointcloud(m = self.m,
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
        return (points.astype(np.float32), tri, centroids, np.asarray(mesh.triangle_normals))
    
    def G_level_sets(self):
        """
        Computes and returns the level sets of the G-function, ensuring they all share the same orientation.

        This method identifies the level sets of the G-function across the specified range between the minimum
        and maximum values of G, adjusted by a small height parameter. It helps in generating machining paths
        at different levels of the material surface.

        Returns:
        -------
        list of np.ndarray
            A list of arrays where each array represents a level set of the G-function. All arrays are oriented
            to ensure consistency in direction across different levels.

        Notes:
        -----
        The delta parameter in level set computation controls the accuracy of the level sets; smaller values
        lead to more accurate but computationally intensive calculations.
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
        """
        Generates and returns parametric representations of u and v curves based on the computed G-function level sets.

        This function computes the percentual value time of each point in the level set curves and uses cubic spline
        interpolation to parametrize these curves.

        Returns:
        -------
        list
            A list containing two lists: one for u-curves and one for v-curves, each curve represented as a CubicSpline object.
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
        """
        Concatenates and returns envelope meshes based on the specified method.

        Parameters:
        ----------
        method : str
            The method to use for concatenating envelopes. Can be 'by_excess', 'from0to1', or 'beauty'.

        Returns:
        -------
        o3d.geometry.TriangleMesh
            A single TriangleMesh object that represents concatenated envelopes according to the specified method.

        Raises:
        -------
        ValueError
            If the input method is not recognized.

        Notes:
        -----
        The 'by_excess' method extends the envelopes beyond the machining region, which is useful for error computation.
        The 'from0to1' method confines the envelopes strictly within the machining region.
        The 'beauty' method ensures envelopes only slightly exceed the machining region, suitable for visualizations.
        """
        if method == 'by_excess':
            '''Returns concatenated envelopes that leave the region by excess.
            Default for computing errors, since it is easy to implement and will not perturb much the error.'''
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
            '''Returns concatenated envelopes inside the machining region.
            Might be useful at some point.'''
            discretized_envelopes = np.array([
                env.value_at(self.angles, self.times, method='whole_envelope').reshape(-1,3) 
                for env in self.envelopes])
            return concatenate_meshes(discretized_envelopes, np.array([self.simplices for i in range(len(self.envelopes))]))
        elif method == 'beauty':
            '''Returns concatenated envelopes that leave the region but just the enough amount.
            Destined to be used only with rendering purposes since it is slow to compute'''
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
        """
        Computes and returns the shank positions in a vectorized form for all envelopes.

        This method calculates the discrete shank positions at various time steps within each envelope,
        considering a specified length.

        Parameters:
        ----------
        n_shanks : int, optional
            The number of discrete shank positions to calculate along the envelopes, default is 25.
        distance : float, optional
            The length of the shank, default is 1 + sqrt(5).

        Returns:
        -------
        np.ndarray
            A 3D array where each row represents a shank position in 3D space.
        """
        T = np.linspace(0,1,n_shanks)
        all_shanks = np.array([env.shank_at_t(T, distance = distance) for env in self.envelopes])
        all_shanks = all_shanks.transpose(0, 2, 1, 3)  # Now shape is (8, 25, 2, 3)
        return  all_shanks.reshape(-1, 2, 3)

    def error_measure_per_triangle_open3d(self):
        """
        Measures the machining error per triangle using raycasting in Open3D, considering both gauging and undercutting.

        This method uses raycasting to measure the distance from the surface centroids to the nearest mesh intersection
        point, identifying the maximum deviation caused by the machining process.

        This is more safer than error_measure_per_triangle_open3d_bis, but slower.
        Returns:
        -------
        np.ndarray
            An array containing the signed distances from the centroids to the mesh, where positive values indicate
            gauging and negative values indicate undercutting.
        """
        # create the mesh and convert into the t thing
        mesh = self.concatenated_envelopes_fun(method = 'by_excess')
        mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(mesh2)

        # create the first intersections
        triangle_normals = -self.surface_normals
        C = np.hstack((self.surface_centroids, triangle_normals))

        # intersect the gauging
        rays = o3d.core.Tensor(C,
                            dtype=o3d.core.Dtype.Float32)

        ans1 = scene.cast_rays(rays)
        A = ans1['t_hit'].numpy()
        # intersect the undercutting
        triangle_normals = self.surface_normals
        C = np.hstack((self.surface_centroids, triangle_normals))
        rays = o3d.core.Tensor(C,
                            dtype=o3d.core.Dtype.Float32)

        ans2 = scene.cast_rays(rays)
        B = ans2['t_hit'].numpy()
        distances = np.where(np.isfinite(A), -A, B)
        # change the inf values by 1000
        distances = np.where(np.isfinite(distances), distances, 1000*np.ones_like(B))
        return distances

    def error_measure_per_triangle_open3d_bis(self):
        """
        Equivalent to error_measure_per_triangle_open3d, but faster. Care should be taken when very
        complex geometries are used since it could return wrong results. In those scenarios,
        error_measure_per_triangle_open3d is preferred.

        Returns:
        -------
        np.ndarray
            An array of distances representing the error at each triangle centroid.
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
        ray_origins = self.surface_centroids - 2*self.R*self.surface_normals
        ray_directions = self.surface_normals
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
        #         point_in_plane=self.surface_centroids)
        #                 )
        
        # A = A*signs
        A = A - 2*self.R 
        # There might be some rights that do not hit the mesh
        # For those points, the distance is infinity.
        # We want to correct that to a high value but not too high.
        distances = np.where(np.isfinite(A), A, 1000)
        return distances
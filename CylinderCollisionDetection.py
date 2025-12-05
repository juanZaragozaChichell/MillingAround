r"""
This library is designed for collision detection of cylinders against triangle meshes.
It contains the implementation described in our SPM2024 paper and the Propositions 1&amp;2
that appear there.

Modules:
- numpy: For numerical operations.
- open3d: For 3D geometry operations and visualizations.
"""
import numpy as np
# import trimesh
import open3d as o3d

# index columns for information_matrix
index_of_is_cylinder_colliding = 0
index_of_cylinder_index = 1
index_of_bottom = 2
index_of_top = 3
index_of_list_of_points_in_cylinder = 4
index_of_list_of_footpoints = 5
index_of_list_of_safe_distances = 6
index_of_collision_indices = 7
index_of_list_of_iterations = 8

# Helper class: CollisionInformation
class CollisionInformation:
    r"""Wrapper providing structured access to the information_matrix returned by the cylinder collision routines.

    Parameters
    ----------
    information_matrix : np.ndarray
        The object-typed NumPy array produced by the collision detection routines
        in :class:`MultipleCylinders`.
    
    Attributes
    ----------
    information_matrix : np.ndarray
        Raw table where each row stores the intermediate results for one cylinder.
    is_cylinder_colliding : np.ndarray
        Boolean flag per row indicating whether the sample is currently colliding.
    cylinder_index : np.ndarray
        Integer identifiers tying each row back to the original cylinder list.
    bottoms : np.ndarray
        Bottom points of each cylinder segment.
    tops : np.ndarray
        Top points of each cylinder segment.
    points_in_cylinder : np.ndarray
        Sampled points that were inspected along the cylinder axis.
    footpoints : np.ndarray
        Closest mesh points (and associated metadata) returned by the raycasting scene.
    safe_distances : np.ndarray
        Signed safety margins computed for every inspected point.
    collision_indices : np.ndarray
        Integer labels encoding the collision state (-2 undetermined, 0 colliding, etc.).
    iterations : np.ndarray
        Iteration counters produced by the iterative refinement loop.

    Methods
    -------
    colliding_ids()
        Return the IDs of cylinders currently classified as colliding.
    non_colliding_ids()
        Return the IDs of cylinders cleared of collision.
    save(filepath, compressed=True)
        Persist the wrapped matrix to disk for later inspection.
    load(filepath)
        Restore a :class:`CollisionInformation` instance from a ``.npz`` artifact.
    """

    def __init__(self, information_matrix: np.ndarray):
        self.information_matrix     = information_matrix
        self.is_cylinder_colliding  = information_matrix[:, index_of_is_cylinder_colliding]
        self.cylinder_index         = information_matrix[:, index_of_cylinder_index]
        self.bottoms                = information_matrix[:, index_of_bottom]
        self.tops                   = information_matrix[:, index_of_top]
        self.points_in_cylinder     = information_matrix[:, index_of_list_of_points_in_cylinder]
        self.footpoints             = information_matrix[:, index_of_list_of_footpoints]
        self.safe_distances         = information_matrix[:, index_of_list_of_safe_distances]
        self.collision_indices      = information_matrix[:, index_of_collision_indices]
        self.iterations             = information_matrix[:, index_of_list_of_iterations]

    # Convenience helpers
    def colliding_ids(self):
        r"""Return indices of cylinders classified as colliding."""
        return self.cylinder_index[self.is_cylinder_colliding == True]

    def non_colliding_ids(self):
        r"""Return indices of cylinders classified as free of collision."""
        return self.cylinder_index[self.is_cylinder_colliding == False]

    def save(self, filepath, compressed: bool = True):
        r"""Save the underlying information matrix to a NumPy ``.npz`` archive.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file path. If the provided path does not end with
            ``.npz`` the extension will be appended automatically.
        compressed : bool, optional
            Use :func:`numpy.savez_compressed` when *True* (default) for
            smaller files at the cost of a little CPU time. Set to *False*
            to use :func:`numpy.savez` instead.
        """
        if not str(filepath).endswith(".npz"):
            filepath = f"{filepath}.npz"
        saver = np.savez_compressed if compressed else np.savez
        # Information matrix can contain Python objects, so we just pass it
        # directly—NumPy will store it as an ``object`` array.
        saver(filepath, information_matrix=self.information_matrix)

    @classmethod
    def load(cls, filepath):
        r"""Load a :class:`CollisionInformation` instance from a ``.npz`` file.

    This mirrors the ``load`` classmethod used for shank collision
    information and simply reconstructs the wrapper around the stored
    ``information_matrix``.
    """
        data = np.load(filepath, allow_pickle=True)
        information_matrix = data["information_matrix"]
        return cls(information_matrix)

#####################
#  previous functions
#####################
def are_points_on_boundary(A):
    r"""Check whether points lie on the boundary of the unit square.

    The boundary is defined in the :math:`XY` plane by the lines
    ``x = 0``, ``x = 1``, ``y = 0`` or ``y = 1``. The input can be
    either

    - a NumPy array of shape ``(n, 3)``, interpreted as ``n`` 3D points,
    - or a NumPy array of shape ``(3,)``, interpreted as a single 3D
      point.

    For each point, the result is ``True`` if its ``x`` or ``y``
    coordinate is ``0`` or ``1``, and ``False`` otherwise.

    Assumes the mesh is defined above the unit square in the XY plane.

    Parameters
    ----------
    A : numpy.ndarray
        Input array of shape ``(n, 3)`` or ``(3,)``.

    Returns
    -------
    numpy.ndarray or bool
        A boolean array of length ``n`` when ``A`` has shape ``(n, 3)``,
        or a single boolean when ``A`` has shape ``(3,)``.
    """
    # Check if A is a vector (shape (3,)) or an array of shape (n, 3)
    if A.ndim == 1:  # It's a vector
        return (A[0] == 0) or (A[1] == 0) or (A[0] == 1) or (A[1] == 1)
    else:  # It's an array
        return (A[:,0] == 0) | (A[:,1] == 0) | (A[:,0] == 1) | (A[:,1] == 1)

def signs_points_to_surface(points, footpoints, vectors):
    r"""
    Determine on which side of a surface points lie.

    This function computes the sign of the dot product between
    ``points - footpoints`` and the normal vectors at the footpoints.
    It effectively tells whether each point is above, below, or exactly
    on the local tangent plane defined at its corresponding footpoint.

    Vectorized NumPy operations are used to efficiently handle many
    points at once.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n, 3)`` with the coordinates of the points.
    footpoints : np.ndarray
        Array of shape ``(n, 3)`` with the closest points on the surface
        corresponding to each point in ``points``.
    vectors : np.ndarray
        Array of shape ``(n, 3)`` with the surface normal vectors at
        each footpoint.

    Returns
    -------
    np.ndarray
        One-dimensional array of shape ``(n,)`` containing the sign of
        the dot product between ``points - footpoints`` and ``vectors``.
        The sign indicates the relative position of each point:

        - ``+1`` if the point is in the direction of the normal (above the surface),
        - ``-1`` if it is on the opposite side (below the surface),
        - ``0`` if it lies exactly on the surface.
    """
    return np.sign(np.einsum('ij,ij->i', vectors, points-footpoints)) # checks plane side but faster!

def check_plane_side(point_to_check, normal_vector, point_in_plane):
    r"""
    Determine on which side of a plane points lie.

    This function computes the signed scalar projection (dot product) of
    ``point_to_check - point_in_plane`` onto the plane's normal vector.
    The sign of this projection indicates whether each point is on the
    normal side of the plane, on the opposite side, or exactly on the plane.

    Both single-point and multi-point evaluations are supported, as well as
    a single plane or multiple planes at once.

    Parameters
    ----------
    point_to_check : np.ndarray
        Points whose position relative to the plane is to be determined.
        Either shape ``(n, 3)`` for ``n`` points or ``(3,)`` for a single point.
    normal_vector : np.ndarray
        Normal vector(s) to the plane. Shape ``(3,)`` for a single plane
        or ``(n, 3)`` for multiple planes.
    point_in_plane : np.ndarray
        Point(s) on the plane. Shape ``(3,)`` for a single plane or the
        same shape as ``normal_vector`` when multiple planes are used.

    Returns
    -------
    np.ndarray or float
        Signed distance value(s) from each ``point_to_check`` to the plane
        defined by ``point_in_plane`` and ``normal_vector``. Positive values
        indicate the point is on the side of the normal vector, negative
        values indicate the opposite side, and zero means the point lies
        exactly on the plane.

    Notes
    -----
    - The implementation is vectorized to efficiently handle multiple
    points and multiple planes at once.
    - This is useful in computational geometry tasks such as rendering,
    collision detection, and any application where relative position to
    a plane must be quantified.
    """
    # evaluates <v, p-a>, where v is the normal vector to a plane
    # a is a point in the plane and p is the point we want to check
    # since we want for multiple points, matrix implementation follows
    if normal_vector.shape == (3,):
        return np.dot(normal_vector, (point_to_check - point_in_plane).T)
    else:
        return np.array([np.dot(a,b) for a,b in zip(normal_vector , point_to_check - point_in_plane)])

class MultipleCylinders:
    r"""
    Manages collision detection among multiple cylinders against a given mesh.

    Parameters:
    ----------
    set_of_cylinders : np.ndarray
        An array of cylinders where each cylinder is represented by two points representing their medial axis (start and end), shape (N, 2, 3).
    mesh : open3d.geometry.TriangleMesh
        The mesh object which may interact with the cylinders.
    scene : open3d.t.geometry.RaycastingScene
        A scene object from Open3D that supports efficient raycasting operations.
    R : float
        The radius of the cylinders, used in collision computations.

    Attributes:
    ----------
    bottoms : np.ndarray
        Starting points of each cylinder.
    tops : np.ndarray
        Ending points of each cylinder.
    axis_vector : np.ndarray
        The direction vector from the top to the bottom of each cylinder.
    L : float
        Length of the axis vector (assumed uniform across all cylinders).
    unitary_vectors : np.ndarray
        Normalized axis vectors for each cylinder.
        """

    def __init__(self, set_of_cylinders, mesh, scene, R):
        r"""
        Initializes the Multiplecylinders object with necessary geometry and collision detection settings.

        Parameters:
        ----------
        set_of_cylinders : np.ndarray
            An array of shape (N, 2, 3) where N is the number of cylinders. Each cylinder is represented by two points (bottom and top),
            which define the linear segment of the cylinder in 3D space.
        mesh : open3d.geometry.TriangleMesh
            The mesh against which the cylinders will be checked for collisions.
        scene : open3d.t.geometry.RaycastingScene
            A pre-configured scene from Open3D that includes the mesh. This scene is used for efficient raycasting operations
            necessary for collision detection.
        R : float
            The radius of the cylinders, used to determine the collision boundary around each cylinder. This parameter defines
            the safety envelope around the linear segments of the cylinders.

        Attributes:
        ----------
        bottoms : np.ndarray
            The starting points (bottoms) of each cylinder, derived directly from `set_of_cylinders`.
        tops : np.ndarray
            The ending points (tops) of each cylinder, also derived from `set_of_cylinders`.
        axis_vector : np.ndarray
            Vectors from the tops to the bottoms of each cylinder, indicating the directional axis of the cylinders.
        L : float
            The length of the axis vector from the first cylinder. Used to normalize axis vectors.
        unitary_vectors : np.ndarray
            Normalized axis vectors for each cylinder, calculated as the unit vector along the direction from top to bottom.
        """
        self.set_of_cylinders = set_of_cylinders
        self.mesh = mesh
        self.scene = scene
        self.R = R
        self.bottoms = self.set_of_cylinders[:,0]
        self.tops = self.set_of_cylinders[:,1]
        self.axis_vector = self.bottoms - self.tops
        self.L = np.linalg.norm(self.axis_vector[0])
        self.unitary_vectors = (1/self.L)*self.axis_vector

    def distance_to_axis(self, points, working_indices):
        r"""Compute perpendicular distances from points to cylinder axes.

        Given a set of 3D points and their corresponding cylinder indices,
        this method returns the shortest (orthogonal) distance from each
        point to the axis of its associated cylinder.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n, 3)`` with the 3D points whose distance to a
            cylinder axis is to be computed.
        working_indices : np.ndarray
            Integer array of shape ``(n,)`` giving, for each point in
            ``points``, the index of the cylinder it is associated with. These
            indices are used to select the appropriate cylinder tops and unit
            direction vectors.

        Returns
        -------
        distances : np.ndarray
            Array of shape ``(n,)`` containing the perpendicular distance from
            each point in ``points`` to the axis of its associated cylinder.
            Distances are computed as the norm of the component of
            ``points - top`` orthogonal to the cylinder’s axis vector.

        Notes
        -----
        The computation uses vector projection to decompose the vector from the
        cylinder top to each point into components parallel and orthogonal to
        the axis direction, and then takes the norm of the orthogonal
        component.
        """
        working_tops = self.tops[working_indices] # select approriate tops
        working_unitary_vectors = self.unitary_vectors[working_indices] # select appropriate unitary vectors
        # compute the points in the cylinder that are perpendicular to the input points
        parameters_in_cylinder = np.einsum('ij,ij->i', working_unitary_vectors, points - working_tops) # the parameter on the cylinder is given by <footpoint -top, unit_vector>
        points_in_cylinder  = working_tops + (parameters_in_cylinder*working_unitary_vectors.T).T # point = top + paremeter*unit_vector
        #finally, compute the distances
        distances = np.linalg.norm(points_in_cylinder - points, axis=1)
        return distances

    def compute_safe_spheres(self, list_of_points, working_indices):
        r"""
        Computes safety spheres around given points to determine potential collision states with a mesh.

        Collision detection is based on the definition given in our SPM2024 paper and the Propositions 1&amp;2

        Parameters:
        ----------
        list_of_points : list or np.ndarray
            Points to evaluate for potential collisions, expected to be an array-like structure of 3D points.
        working_indices : list or np.ndarray
            Indices of cylinders corresponding to each point in `list_of_points`. This maps each point to a specific
            cylinder for assessing its relationship with the mesh in the context of that cylinder's position and orientation.

        Returns:
        -------
        collision_index : np.ndarray
            An array indicating the collision status for each point:
            - -2: Undetermined (initial state).
            - 1: Non-colliding (safe).
            - 0: Colliding.
            - -1: Footpoint below the tool cutter (considered safe).
        list_of_points : np.ndarray
            The original list of points passed to the function, formatted as a NumPy array.
        footpoints : np.ndarray
            The nearest points on the mesh to each point in `list_of_points`.
        safe_distances : np.ndarray
            Calculated safe distances from each point to its corresponding footpoint, adjusted for the cylinder's radius.

        Notes:
        -----
        - The method uses the scene's raycasting capabilities to find the nearest mesh points (footpoints) for the given points.
        - It calculates whether these points are on the boundary of the mesh, above or below the tool's cutting plane,
        and their perpendicular distance to the cylinder axis to assess collision risks.
        """
        
        collision_index = -2*np.ones(len(list_of_points)).astype(int) # set the indicies to undetermined by default
        list_of_points = np.array(list_of_points) # convert into numpy array

        # get footpoints and the triangles they lie on
        footpoints_triangles_and_such = self.scene.compute_closest_points(
            list_of_points.astype(np.float32))
        footpoints = footpoints_triangles_and_such['points'].numpy()
        triangles = footpoints_triangles_and_such['primitive_ids'].numpy()

        # compute distance between the points and the footpoints
        distances = np.linalg.norm(footpoints - list_of_points, axis = 1)

        # checkplane side for the footpoints
        footpoints_tool_side = np.einsum('ij,ij->i', 
                                         -self.axis_vector[working_indices], 
                                         footpoints-self.bottoms[working_indices]) # checks plane side but faster!
        
        # if footpoints_tool_side is negative, then the footpoint lies below the tool cutting plane (hence non colliding)
        collision_index[footpoints_tool_side <= 1e-6] = -1 # label those as non colliding 
        # 1e-6 is used insted of 0 or the sake of numerical rounding

        # get sign of the distance from the points to the surface
        signs_point_to_surface = signs_points_to_surface(list_of_points,
                                                            footpoints, 
                                                            np.asarray(self.mesh.triangle_normals)[triangles])
        
        are_footpoints_on_boundary = are_points_on_boundary(footpoints) # check if a footpoint is on the boundary
        # if the point has footpoint on the boundary and is labeled to be below tool the surface, change its value
        signs_point_to_surface[(signs_point_to_surface < 0) & are_footpoints_on_boundary] *= -1  
        safe_distances = signs_point_to_surface*distances - self.R # get safe distance for the safe balls
        distances_to_cylinders = self.distance_to_axis(footpoints, working_indices) # get distance to the cylinders
        # distances_to_cylinders < self.R - 0.000125 aims to avoid collision detection with footpoints that lie on the cutting tool
        collision_index[np.logical_and(np.logical_or( safe_distances < 0,  distances_to_cylinders < self.R - 0.000125), collision_index != -1)  ] = 0
        # if the footpoint lies on the cutting edge, the tool is non colliding either
        # the previous computations have made sure that the footpoint does not lie inside the cylinder, nor below the cutting plane
        # hence we can check if it is in the cutting edge by measuring the distace with a threshold
        distances_to_bottoms = np.linalg.norm(footpoints - self.bottoms[working_indices], axis = 1)- self.R
        collision_index[np.logical_and(collision_index == -2, np.isclose(distances_to_bottoms, np.zeros(len(distances_to_bottoms)), atol=5e-4))] = -1

        # it could happen that you are moving further away from the bottom. Hence, the point is non colliding
        # we are gonna check this by measuring the distance to the top.
        # That is going to be measured as the distance from the point to the top plus the safe distance
        # since that is what would give the next point for computation
        distance_to_tops = np.linalg.norm(list_of_points - self.tops[working_indices], axis = 1) +  safe_distances
        distance_compared_to_top = distance_to_tops - self.L # measures the excess in case that there is some
        collision_index[np.logical_and(collision_index == -2, distance_compared_to_top > 0)] = -1

        collision_index[collision_index == -2] = 1
        return (collision_index, list_of_points, footpoints, safe_distances)
        
    def get_next_points(self, information_matrix, working_indices):
        r"""Compute the next set of points to evaluate for collision.

        This method advances the collision detection process by moving from the
        last known safe points along each cylinder's axis direction by the last
        computed safe distance.

        Based on Propositions 1 and 2 from our SPM2024 paper.

        Parameters
        ----------
        information_matrix : np.ndarray
            Object-typed array storing per-cylinder data, including the history
            of evaluated points, safe distances, and collision indices. It is
            expected to follow the column layout described in the
            :class:`CollisionInformation` wrapper.
        working_indices : np.ndarray
            One-dimensional array of integer indices identifying the cylinders
            that are still under evaluation in the current iteration.

        Returns
        -------
        np.ndarray
            Array of shape ``(n, 3)`` containing the next points to be
            evaluated. For each cylinder in ``working_indices``, the next point
            is obtained by moving from the last evaluated point along the
            cylinder's unit axis vector by the magnitude of the last safe
            distance.

        Notes
        -----
        - For each cylinder, the last point and last safe distance are read
          from ``information_matrix`` using ``working_indices``.
        - The new point is computed as

          .. math::

             p_{\text{next}} = p_{\text{last}} + d_{\text{safe}} \, \hat{u},

          where :math:`p_{\text{last}}` is the last evaluated point,
          :math:`d_{\text{safe}}` is the last safe distance, and
          :math:`\hat{u}` is the unit axis direction of the cylinder.
        """
        # from the lists of points in the cylinder, get only the last one for each considered cylinder
        list_of_points = np.array(information_matrix[working_indices, index_of_list_of_points_in_cylinder].tolist())[:, -1]
        list_of_vectors = self.unitary_vectors[working_indices]
        list_of_safe_distances = np.array(information_matrix[working_indices, index_of_list_of_safe_distances].tolist())[:, -1]
        return list_of_points + list_of_safe_distances[:, np.newaxis]*list_of_vectors
        

    def collision_detection_no_gaps(self):
        r"""
        Conducts an exhaustive collision detection process without any gaps, iteratively assessing each cylinder's 
        collision state until all cylinders are evaluated or a colliding cylinder is found.

        This method initiates with the assumption that the collision state of each cylinder is unknown ('?').
        Collision detection is based on the our definitions and propositions. The method updates cylinder states iteratively.

        Returns:
        -------
        tuple
            A tuple containing:
            - int: The number of iterations it took to resolve the collision states of all cylinders.
            - np.ndarray: The updated information matrix detailing the current state of each cylinder, including whether 
                        it is colliding, non-colliding, or undetermined.

        Information Matrix Structure:
        -----------------------------
        The information matrix contains several columns that store different types of data for each cylinder:
            0. Collision State ('?', True, False) - Initial unknown state '?', True for colliding, False for non-colliding.
            1. cylinder Index - Numeric index of the cylinder in the set.
            2. cylinder Bottom - 3D coordinates of the bottom point of the cylinder.
            3. cylinder Top - 3D coordinates of the top point of the cylinder.
            4. List of Points - Dynamic list of points checked for collision along the cylinder.
            5. List of Footpoints - Corresponding closest points on the mesh for each point checked.
            6. List of Safe Distances - Computed safe distances for each point to determine if it is within a safe buffer.
            7. Collision Indices - Dynamic list of collision results for each point checked.
            8. List of Iterations - Records the iteration number when each point was checked.

        Process:
        --------
        1. Initializes the information matrix with default values for all cylinders.
        2. Iteratively computes the safe distances and checks for collisions using `compute_safe_spheres`.
        3. Updates the information matrix with new data after each iteration, adjusting points based on their last known safe positions.
        4. Continues until all cylinders are confirmed as non colliding or a collision is found.

        Notes:
        -----
        - The method dynamically adjusts the points of evaluation along the cylinder based on previously 
        calculated safe distances.
        - It terminates when all cylinders are confirmed as non colliding or a collision is found.
        - In the final result, only case where a cylinder's colliding state can be '?' is if there's a collision
        in another cylinder and it has been detected.
        """

        # initialization of the information_matrix
        information_matrix = np.array([ ['?',i ,self.bottoms[i],self.tops[i],  [],[],[],[],[]] for i in range(len(self.tops))], dtype = object) 

        iteracion = 0
        iteration = list(iteracion*np.ones(len(self.tops)).astype(int))
        # indices to work with
        working_indices = information_matrix[information_matrix[:, index_of_is_cylinder_colliding] == '?'][:, index_of_cylinder_index].astype(int) 

        list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances= self.compute_safe_spheres(self.tops, working_indices)
        # place the information in an iterable so that we can easilly change the info in the information matrix
        attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
        # modify the information
        for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_cylinder, index_of_list_of_iterations + 1), attributes):
            # we iterate over the attributes we want to modify
            # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
            information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[:, index_of_attribute] , attribute)]
        # set cylinders to colliding or noncolliding depending on the value of the first top
        for i, val in zip([0, -1], [True, False]):
            information_matrix[np.where(list_of_collision_index.astype(int) == i)[0].astype(int), index_of_is_cylinder_colliding] = val
        
        while np.any(information_matrix[:,index_of_is_cylinder_colliding] == '?') and np.all(information_matrix[:,index_of_is_cylinder_colliding] != True):
            # hte loop iterates while there are cylinders whose collidingness has not been decided yet and there are no colliding cylinders found
            iteracion += 1
            # indices to work with
            #   they are selected to be those as not labeled as False yet. They need to be int.
            working_indices      = information_matrix[information_matrix[:, index_of_is_cylinder_colliding] == '?'][:, index_of_cylinder_index].astype(int) 
            # list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances = self.compute_safe_spheres(list_of_points_prev, working_indices)

            # first of all, set the ones with -1 to False
            list_of_collision_index = list_of_collision_index.astype(int)
            if list_of_collision_index[list_of_collision_index == -1].tolist():
                information_matrix[np.array(working_indices)[list_of_collision_index == -1].astype(int), index_of_is_cylinder_colliding] = False
            
            if list_of_collision_index[list_of_collision_index == 0].tolist():
                information_matrix[working_indices[list_of_collision_index == 0], index_of_is_cylinder_colliding] = True
            
            # add now the information to every index
            iteration = list(iteracion*np.ones(len(list_of_points)).astype(int))
            attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
            # modify the information
            for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_cylinder, index_of_list_of_iterations + 1), attributes):
                # we iterate over the attributes we want to modify
                # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
                information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[working_indices, index_of_attribute] , attribute)]
            
        collision_information = CollisionInformation(information_matrix)
        return (iteracion, collision_information)
    
    def collision_detection_no_gaps_detect_all_cylinders(self):
        r"""
        Conducts an exhaustive collision detection process without any gaps, iteratively assessing each cylinder's 
        collision state until all cylinders are evaluated.

        This method initiates with the assumption that the collision state of each cylinder is unknown ('?').
        Collision detection is based on the our definitions and propositions. The method updates cylinder states
        iteratively.

        Returns:
        -------
        tuple
            A tuple containing:
            - int: The number of iterations it took to resolve the collision states of all cylinders.
            - np.ndarray: The updated information matrix detailing the current state of each cylinder, including whether 
                        it is colliding, non-colliding, or undetermined.

        Information Matrix Structure:
        -----------------------------
        The information matrix contains several columns that store different types of data for each cylinder:
            0. Collision State ('?', True, False) - Initial unknown state '?', True for colliding, False for non-colliding.
            1. cylinder Index - Numeric index of the cylinder in the set.
            2. cylinder Bottom - 3D coordinates of the bottom point of the cylinder.
            3. cylinder Top - 3D coordinates of the top point of the cylinder.
            4. List of Points - Dynamic list of points checked for collision along the cylinder.
            5. List of Footpoints - Corresponding closest points on the mesh for each point checked.
            6. List of Safe Distances - Computed safe distances for each point to determine if it is within a safe buffer.
            7. Collision Indices - Dynamic list of collision results for each point checked.
            8. List of Iterations - Records the iteration number when each point was checked.

        Process:
        --------
        1. Initializes the information matrix with default values for all cylinders.
        2. Iteratively computes the safe distances and checks for collisions using `compute_safe_spheres`.
        3. Updates the information matrix with new data after each iteration, adjusting points based on their last known safe positions.
        4. Continues until all cylinders are labeled as (non)colliding.

        Notes:
        -----
        - The method dynamically adjusts the points of evaluation along the cylinder based on previously 
        calculated safe distances.
        - It terminates when all cylinders are labeled as (non)colliding.
        - In the final result, all cylinders should be 'True' or 'False'.
        """

        # initialization of the information_matrix
        information_matrix = np.array([ ['?',i ,self.bottoms[i],self.tops[i],  [],[],[],[],[]] for i in range(len(self.tops))], dtype = object) 

        iteracion = 0
        iteration = list(iteracion*np.ones(len(self.tops)).astype(int))
        # indices to work with
        working_indices = information_matrix[information_matrix[:, index_of_is_cylinder_colliding] == '?'][:, index_of_cylinder_index].astype(int) 

        list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances= self.compute_safe_spheres(self.tops, working_indices)
        # place the information in an iterable so that we can easilly change the info in the information matrix
        attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
        # modify the information
        for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_cylinder, index_of_list_of_iterations + 1), attributes):
            # we iterate over the attributes we want to modify
            # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
            information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[:, index_of_attribute] , attribute)]
        # set cylinders to colliding or noncolliding depending on the value of the first top
        for i, val in zip([0, -1], [True, False]):
            information_matrix[np.where(list_of_collision_index.astype(int) == i)[0].astype(int), index_of_is_cylinder_colliding] = val
        
        while np.any(information_matrix[:,index_of_is_cylinder_colliding] == '?') and iteracion < 20:
            # hte loop iterates while there are cylinders whose collidingness has not been decided yet and there are no colliding cylinders found
            iteracion += 1
            # indices to work with
            #   they are selected to be those as not labeled as False yet. They need to be int.
            working_indices      = information_matrix[information_matrix[:, index_of_is_cylinder_colliding] == '?'][:, index_of_cylinder_index].astype(int) 
            # list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances = self.compute_safe_spheres(list_of_points_prev, working_indices)

            # first of all, set the ones with -1 to False
            list_of_collision_index = list_of_collision_index.astype(int)
            if list_of_collision_index[list_of_collision_index == -1].tolist():
                information_matrix[np.array(working_indices)[list_of_collision_index == -1].astype(int), index_of_is_cylinder_colliding] = False
            
            if list_of_collision_index[list_of_collision_index == 0].tolist():
                information_matrix[working_indices[list_of_collision_index == 0], index_of_is_cylinder_colliding] = True
            
            # add now the information to every index
            iteration = list(iteracion*np.ones(len(list_of_points)).astype(int))
            attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
            # modify the information
            for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_cylinder, index_of_list_of_iterations + 1), attributes):
                # we iterate over the attributes we want to modify
                # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
                information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[working_indices, index_of_attribute] , attribute)]
        # if there is still a cylinder labeled as '?' after all the evaluations, 
        # probably the position is not safe enough so we label it as colliding
        mask = np.array([row[0] == '?' for row in information_matrix])
        # Replace '?' with True in the first element of the selected rows
        information_matrix[mask, 0] = True
        collision_information = CollisionInformation(information_matrix)
        return (iteracion, collision_information)
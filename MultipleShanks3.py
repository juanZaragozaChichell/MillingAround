"""
This library is designed for collision detection of shanks in toroidal cutting and flat-end milling scenarios.
It utilizes computational geometry and raycasting techniques to evaluate potential collisions between shanks
and machined surfaces, helping to ensure the safety and efficiency of CNC machining processes.

Modules:
- numpy: For numerical operations.
- open3d: For 3D geometry operations and visualizations.
- time: For performance timing.
"""
import numpy as np
# import trimesh
import open3d as o3d
from time import perf_counter as clock

# index columns for information_matrix
index_of_is_shank_colliding = 0
index_of_shank_index = 1
index_of_bottom = 2
index_of_top = 3
index_of_list_of_points_in_shank = 4
index_of_list_of_footpoints = 5
index_of_list_of_safe_distances = 6
index_of_collision_indices = 7
index_of_list_of_iterations = 8

#####################
#  previous functions
#####################
def are_points_on_boundary(A):
    """
    This function takes a numpy array A of shape (n,3) or a vector of shape (3,) and returns a boolean vector of length n
    or a single boolean value respectively. The value in the ith coordinate is True if A[i][0] == 0 or A[i][1] == 0 or
    A[i][0] == 1 or A[i][1] == 1 for an array, and the condition is evaluated for each element if A is a vector.
    
    Args:
    - A (numpy array): An input array of shape (n,3) or a vector of shape (3,).
    
    Returns:
    - numpy array or boolean: A boolean vector of length n or a single boolean value.
    """
    # Check if A is a vector (shape (3,)) or an array of shape (n, 3)
    if A.ndim == 1:  # It's a vector
        return (A[0] == 0) or (A[1] == 0) or (A[0] == 1) or (A[1] == 1)
    else:  # It's an array
        return (A[:,0] == 0) | (A[:,1] == 0) | (A[:,0] == 1) | (A[:,1] == 1)

def signs_points_to_surface(points, footpoints, vectors):
    """
    Determines the sign of the dot product between vectors and the displacement from footpoints to points,
    effectively checking which side of the surface (defined by the vectors at the footpoints) the points lie on.

    This function uses vectorized operations to efficiently compute whether each point in a set is above, 
    below, or on the plane defined at corresponding footpoints with normal vectors.

    Parameters:
    ----------
    points : np.ndarray
        An array of shape (n, 3) representing the coordinates of n points in space.
    footpoints : np.ndarray
        An array of shape (n, 3) representing the coordinates of n footpoints, each corresponding to one of the points.
        Footpoints are typically the closest points on a surface from each of the points in the `points` array.
    vectors : np.ndarray
        An array of shape (n, 3) representing the normal vectors at each of the footpoints. These vectors define
        the orientation of the surface at the footpoints.

    Returns:
    -------
    np.ndarray
        A 1D array of shape (n,) containing the sign of the dot product between the displacement vectors (points - footpoints)
        and the normal vectors. The sign indicates the relative position of each point with respect to the surface:
        - Positive (+1) means the point is in the direction of the normal vector (above the surface).
        - Negative (-1) means the point is in the opposite direction of the normal vector (below the surface).
        - Zero (0) means the point is exactly on the surface.

    Notes:
    -----
    This function is particularly useful in geometric computations where understanding the position of a point relative to 
    a surface is necessary, such as in collision detection, machining simulations, and more.
    """
    return np.sign(np.einsum('ij,ij->i', vectors, points-footpoints)) # checks plane side but faster!

def check_plane_side(point_to_check, normal_vector, point_in_plane):
    """
    Calculates the signed scalar projection (dot product) of the vector difference between points and a reference point 
    in a plane onto the plane's normal vector. This measure indicates on which side of the plane each point lies.

    This function can handle both single point and multiple points evaluations with respect to a plane defined by 
    a normal vector and a point on the plane.

    Parameters:
    ----------
    point_to_check : np.ndarray
        An array of shape (n, 3) representing n points in space whose positions relative to the plane are to be determined.
        If a single point is provided, it should be of shape (3,).
    normal_vector : np.ndarray
        The normal vector to the plane, of shape (3,) for a single plane, or (n, 3) for multiple planes.
    point_in_plane : np.ndarray
        A point on the plane, of shape (3,) that is used with `normal_vector` to define the plane. If multiple points
        are provided to check against multiple planes, it should match the shape of `normal_vector`.

    Returns:
    -------
    np.ndarray or float
        A value or array of values indicating the signed distance from each `point_to_check` to the plane defined by 
        `point_in_plane` and `normal_vector`. Positive values indicate that the point is on the side of the normal vector,
        negative values indicate the point is on the opposite side, and zero indicates the point lies exactly on the plane.

    Notes:
    -----
    - The function is vectorized to efficiently handle multiple points and multiple planes at once.
    - This function is crucial in computational geometry, particularly in applications like rendering, collision detection,
      and more where the relative spatial relationship to surfaces needs to be quantified.
    """
    # evaluates <v, p-a>, where v is the normal vector to a plane
    # a is a point in the plane and p is the point we want to check
    # since we want for multiple points, matrix implementation follows
    if normal_vector.shape == (3,):
        return np.dot(normal_vector, (point_to_check - point_in_plane).T)
    else:
        return np.array([np.dot(a,b) for a,b in zip(normal_vector , point_to_check - point_in_plane)])

class MultipleShanks:
    """
    Manages collision detection among multiple shanks against a given mesh in milling scenarios.

    Parameters:
    ----------
    set_of_shanks : np.ndarray
        An array of shanks where each shank is represented by two points (start and end), shape (N, 2, 3).
    mesh : open3d.geometry.TriangleMesh
        The mesh object which may interact with the shanks.
    centroids : np.ndarray
        The centroids of the mesh's triangles, used for certain calculations.
    scene : open3d.t.geometry.RaycastingScene
        A scene object from Open3D that supports efficient raycasting operations.
    R : float
        The radius of the shanks, used in collision computations.

    Attributes:
    ----------
    bottoms : np.ndarray
        Starting points of each shank.
    tops : np.ndarray
        Ending points of each shank.
    axis_vector : np.ndarray
        The direction vector from the top to the bottom of each shank.
    L : float
        Length of the axis vector (assumed uniform across all shanks).
    unitary_vectors : np.ndarray
        Normalized axis vectors for each shank.
    """

    def __init__(self, set_of_shanks, mesh, centroids, scene, R):
        """
        Initializes the MultipleShanks object with necessary geometry and collision detection settings.

        Parameters:
        ----------
        set_of_shanks : np.ndarray
            An array of shape (N, 2, 3) where N is the number of shanks. Each shank is represented by two points (bottom and top),
            which define the linear segment of the shank in 3D space.
        mesh : open3d.geometry.TriangleMesh
            The mesh against which the shanks will be checked for collisions. This mesh typically represents the workpiece
            or obstacles in a CNC milling scenario.
        centroids : np.ndarray
            The centroids of each triangle in the mesh. These are used in collision calculations to represent the geometric
            center of mesh facets.
        scene : open3d.t.geometry.RaycastingScene
            A pre-configured scene from Open3D that includes the mesh. This scene is used for efficient raycasting operations
            necessary for collision detection.
        R : float
            The radius of the shanks, used to determine the collision boundary around each shank. This parameter defines
            the safety envelope around the linear segments of the shanks.

        Attributes:
        ----------
        bottoms : np.ndarray
            The starting points (bottoms) of each shank, derived directly from `set_of_shanks`.
        tops : np.ndarray
            The ending points (tops) of each shank, also derived from `set_of_shanks`.
        axis_vector : np.ndarray
            Vectors from the tops to the bottoms of each shank, indicating the directional axis of the shanks.
        L : float
            The length of the axis vector from the first shank. Used to normalize axis vectors.
        unitary_vectors : np.ndarray
            Normalized axis vectors for each shank, calculated as the unit vector along the direction from top to bottom.
        """
        self.set_of_shanks = set_of_shanks
        self.mesh = mesh
        self.scene = scene
        self.centroids = centroids
        self.R = R
        self.bottoms = self.set_of_shanks[:,0]
        self.tops = self.set_of_shanks[:,1]
        self.axis_vector = self.bottoms - self.tops
        self.L = np.linalg.norm(self.axis_vector[0])
        self.unitary_vectors = (1/self.L)*self.axis_vector

    def distance_to_axis(self, points, working_indices):
        """
        Calculates the perpendicular distances from specified points to their respective shank axes.

        This method computes the shortest distance from each point in the provided array to the line
        defined by its corresponding shank axis, utilizing the vector projection formula.
        The computation involves projecting the displacement vector (from the shank's top
        to the point) onto the shank's unitary axis vector.

        Parameters:
        ----------
        points : np.ndarray
            An array of points in 3D space, where each point's distance to a specific shank's axis is to be calculated.
            Shape: (n, 3), where n is the number of points.
        working_indices : np.ndarray
            An array of indices that correspond to the shanks associated with each point in the `points` array.
            These indices are used to select the appropriate shank tops and unitary vectors for the distance calculations.

        Returns:
        -------
        distances : np.ndarray
            An array containing the perpendicular distances from each point in `points` to the axis of the shank associated with it.
            The distances are computed as the length of the vector component orthogonal to the shank's axis.
        
        Example:
        --------
        # Assuming a setup with multiple shanks and a set of points
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        working_indices = np.array([0, 1, 2])  # Points correspond to shanks at these indices
        distances = instance_of_MultipleShanks.distance_to_axis(points, working_indices)
        print(distances)  # Outputs the perpendicular distances to the shank axes

        Notes:
        -----
        The calculation is performed using the formula for the projection of a vector onto another vector, and then
        computing the orthogonal distance from the point to the line defined by the shank's axis.
        """
        working_tops = self.tops[working_indices] # select approriate tops
        working_unitary_vectors = self.unitary_vectors[working_indices] # select appropriate unitary vectors
        # compute the points in the shank that are perpendicular to the input points
        parameters_in_shank = np.einsum('ij,ij->i', working_unitary_vectors, points - working_tops) # the parameter on the shank is given by <footpoint -top, unit_vector>
        points_in_shank  = working_tops + (parameters_in_shank*working_unitary_vectors.T).T # point = top + paremeter*unit_vector
        #finally, compute the distances
        distances = np.linalg.norm(points_in_shank - points, axis=1) # self explanatory
        return distances

    def compute_safe_spheres(self, list_of_points, working_indices):
        """
        Computes safety spheres around given points to determine potential collision states with a mesh.

        Collision detection is based on the definition given in our SPM2024 paper and the Propositions 1&2
        that appear there, as well as the proposition 1 of the new paper.

        Parameters:
        ----------
        list_of_points : list or np.ndarray
            Points to evaluate for potential collisions, expected to be an array-like structure of 3D points.
        working_indices : list or np.ndarray
            Indices of shanks corresponding to each point in `list_of_points`. This maps each point to a specific
            shank for assessing its relationship with the mesh in the context of that shank's position and orientation.

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
            Calculated safe distances from each point to its corresponding footpoint, adjusted for the shank's radius.

        Notes:
        -----
        - The method uses the scene's raycasting capabilities to find the nearest mesh points (footpoints) for the given points.
        - It calculates whether these points are on the boundary of the mesh, above or below the tool's cutting plane,
        and their perpendicular distance to the shank axis to assess collision risks.
        """
        #TODO Add behaviour for when point in the shank goes further than the bottom
        #TODO in this scenario the shank is non colliding.
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
        collision_index[footpoints_tool_side<0] = -1 # label those as non colliding

        # get sign of the distance from the points to the surface
        signs_point_to_surface = signs_points_to_surface(list_of_points,
                                                            footpoints, 
                                                            np.asarray(self.mesh.triangle_normals)[triangles])
        
        are_footpoints_on_boundary = are_points_on_boundary(footpoints) # check if a footpoint is on the boundary
        # if the point has footpoint on the boundary and is labeled to be below tool the surface, change its value
        signs_point_to_surface[(signs_point_to_surface < 0) & are_footpoints_on_boundary] *= -1  
        safe_distances = signs_point_to_surface*distances - self.R # get safe distance for the safe balls
        distances_to_shanks = self.distance_to_axis(footpoints, working_indices) # get distance to the shanks
        # distances_to_shanks < self.R - 0.000125 aims to avoid collision detection with footpoints that lie on the cutting tool
        collision_index[np.logical_and(np.logical_or( safe_distances < 0,  distances_to_shanks < self.R - 0.000125), collision_index != -1)  ] = 0
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
        """
        Computes the next set of points to be evaluated for collision detection based on the latest evaluated points
        and their respective safe distances.

        This method advances the collision detection process by calculating the next points along the shank directions
        from the last known safe points, considering the last computed safe distances.

        Parameters:
        ----------
        information_matrix : np.ndarray
            An array containing various lists of data for each shank, including the history of points checked,
            safe distances calculated, and collision indices.
        working_indices : np.ndarray
            Indices of shanks currently being considered for further collision evaluations.

        Returns:
        -------
        np.ndarray
            An array of the next points to be evaluated. These points are derived by moving along the shank's
            axis vector from the last checked point by the distance defined as safe in the previous evaluation.

        Notes:
        -----
        - The method extracts the last point and the last safe distance for each shank using the provided indices.
        - It then computes the next point by moving along the unit vector of the shank's axis from the last point by
        the magnitude of the safe distance. This approach assumes linear progression along the shank.
        """
        # from the lists of points in the shank, get only the last one for each considered shank
        list_of_points = np.array(information_matrix[working_indices, index_of_list_of_points_in_shank].tolist())[:, -1]
        list_of_vectors = self.unitary_vectors[working_indices]
        list_of_safe_distances = np.array(information_matrix[working_indices, index_of_list_of_safe_distances].tolist())[:, -1]
        return list_of_points + list_of_safe_distances[:, np.newaxis]*list_of_vectors
        

    def collision_detection_no_gaps(self):
        """
        Conducts an exhaustive collision detection process without any gaps, iteratively assessing each shank's 
        collision state until all shanks are evaluated or a colliding shank is found.

        This method initiates with the assumption that the collision state of each shank is unknown ('?').
        Collision detection is based on the our definitions and propositions. The method updates shank states
        iteratively.

        Returns:
        -------
        tuple
            A tuple containing:
            - int: The number of iterations it took to resolve the collision states of all shanks.
            - np.ndarray: The updated information matrix detailing the current state of each shank, including whether 
                        it is colliding, non-colliding, or undetermined.

        Information Matrix Structure:
        -----------------------------
        The information matrix contains several columns that store different types of data for each shank:
            0. Collision State ('?', True, False) - Initial unknown state '?', True for colliding, False for non-colliding.
            1. Shank Index - Numeric index of the shank in the set.
            2. Shank Bottom - 3D coordinates of the bottom point of the shank.
            3. Shank Top - 3D coordinates of the top point of the shank.
            4. List of Points - Dynamic list of points checked for collision along the shank.
            5. List of Footpoints - Corresponding closest points on the mesh for each point checked.
            6. List of Safe Distances - Computed safe distances for each point to determine if it is within a safe buffer.
            7. Collision Indices - Dynamic list of collision results for each point checked.
            8. List of Iterations - Records the iteration number when each point was checked.

        Process:
        --------
        1. Initializes the information matrix with default values for all shanks.
        2. Iteratively computes the safe distances and checks for collisions using `compute_safe_spheres`.
        3. Updates the information matrix with new data after each iteration, adjusting points based on their last known safe positions.
        4. Continues until all shanks are confirmed as non colliding or a collision is found.

        Notes:
        -----
        - The method dynamically adjusts the points of evaluation along the shank based on previously 
          calculated safe distances.
        - It terminates when all shanks are confirmed as non colliding or a collision is found.
        - In the final result, only case where a shank's colliding state can be '?' is if there's a collision
          in another shank and it has been detected.
        """

        # initialization of the information_matrix
        information_matrix = np.array([ ['?',i ,self.bottoms[i],self.tops[i],  [],[],[],[],[]] for i in range(len(self.tops))], dtype = object) 

        iteracion = 0
        iteration = list(iteracion*np.ones(len(self.tops)).astype(int))
        # indices to work with
        working_indices = information_matrix[information_matrix[:, index_of_is_shank_colliding] == '?'][:, index_of_shank_index].astype(int) 

        list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances= self.compute_safe_spheres(self.tops, working_indices)
        # place the information in an iterable so that we can easilly change the info in the information matrix
        attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
        # modify the information
        for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_shank, index_of_list_of_iterations + 1), attributes):
            # we iterate over the attributes we want to modify
            # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
            information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[:, index_of_attribute] , attribute)]
        # set shanks to colliding or noncolliding depending on the value of the first top
        for i, val in zip([0, -1], [True, False]):
            information_matrix[np.where(list_of_collision_index.astype(int) == i)[0].astype(int), index_of_is_shank_colliding] = val
        
        while np.any(information_matrix[:,index_of_is_shank_colliding] == '?') and np.all(information_matrix[:,index_of_is_shank_colliding] != True):
            # hte loop iterates while there are shanks whose collidingness has not been decided yet and there are no colliding shanks found
            iteracion += 1
            # indices to work with
            #   they are selected to be those as not labeled as False yet. They need to be int.
            working_indices      = information_matrix[information_matrix[:, index_of_is_shank_colliding] == '?'][:, index_of_shank_index].astype(int) 
            # list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_points_prev = self.get_next_points(information_matrix=information_matrix, working_indices=working_indices)
            list_of_collision_index, list_of_points, list_of_footpoints, list_of_safe_distances = self.compute_safe_spheres(list_of_points_prev, working_indices)

            # first of all, set the ones with -1 to False
            list_of_collision_index = list_of_collision_index.astype(int)
            if list_of_collision_index[list_of_collision_index == -1].tolist():
                information_matrix[np.array(working_indices)[list_of_collision_index == -1].astype(int), index_of_is_shank_colliding] = False
            
            if list_of_collision_index[list_of_collision_index == 0].tolist():
                information_matrix[working_indices[list_of_collision_index == 0], index_of_is_shank_colliding] = True
            
            # add now the information to every index
            iteration = list(iteracion*np.ones(len(list_of_points)).astype(int))
            attributes = [list_of_points, list_of_footpoints, list_of_safe_distances, list_of_collision_index.astype(int), iteration]
            # modify the information
            for index_of_attribute, attribute in zip(range(index_of_list_of_points_in_shank, index_of_list_of_iterations + 1), attributes):
                # we iterate over the attributes we want to modify
                # what we do is: in the specified indices, the attribute is changed to the refreshed list with the extra information added
                information_matrix[working_indices, index_of_attribute] = [previous_list + [actual_value] for previous_list, actual_value in zip(information_matrix[working_indices, index_of_attribute] , attribute)]
            
        return (iteracion, information_matrix)
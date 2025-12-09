r"""Utility geometries for machining workflows.

This module provides small mesh-generation helpers (cylinders, cones, tori,
ruled surfaces, heatmaps, etc.) built on top of :mod:`trimesh`.  They are used
throughout the point-milling notebooks to export STL assets, create Plotly
visualizations, and reformat backend data for downstream processing.
"""

import numpy as np
import trimesh
from scipy.optimize import approx_fprime
from scipy.spatial import Delaunay
from point_milling_backend import Surface as SurfaceBackend
from plotly.colors import sample_colorscale, get_colorscale, unlabel_rgb
import plotly.graph_objects as go

class Arrow:
    r"""Composite arrow mesh built from a cylinder shaft and cone tip.

    Methods
    -------
    mesh()
        Build the arrow mesh by concatenating shaft and tip.
    mesh_data()
        Return the vertices and faces composing the arrow.
    export_mesh(name)
        Export the arrow mesh to disk.
    """

    def __init__(self, R1, R2, bottom, top, t):
        r"""Initialize the arrow definition.

        Parameters
        ----------
        R1 : float
            Radius of the cylindrical shaft.
        R2 : float
            Radius of the cone tip.
        bottom : array-like of shape (3,)
            Shaft base point.
        top : array-like of shape (3,)
            Cone apex point.
        t : float
            Fraction of the total axis length occupied by the cone (0, 1).
        """
        self.R1 = R1
        self.R2 = R2
        self.bottom = bottom
        self.top = top
        self.t = t

    def mesh(self):
        r"""Build the :class:`trimesh.Trimesh` arrow by concatenating primitives.

        Returns
        -------
        trimesh.Trimesh
            The assembled arrow mesh.
        """
        start_cone = (1-self.t)*self.bottom + self.t*self.top
        Cone = Cone(self.R2, start_cone, self.top)
        Cone_mesh = Cone.mesh(caps=True)
        Cylinder_mesh = Cylinder(bottom=self.bottom, top=start_cone, radius=self.R1).mesh(caps=True)
        mesh = trimesh.util.concatenate([Cone_mesh, Cylinder_mesh])
        return mesh
    
    def mesh_data(self):
        r"""Return the vertices and faces composing the arrow.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Vertex and face arrays.
        """
        mesh = self.mesh()
        return mesh.vertices, mesh.faces
    
    def export_mesh(self, name):
        r"""Export the arrow mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Output file path.
        """
        mesh = self.mesh()
        mesh.export(name)

class Cone:
    r"""Triangulated cone defined by base center, apex, and radius.

    Methods
    -------
    pointcloud()
        Sample circular layers along the cone axis.
    mesh_data()
        Return vertices and triangular faces describing the cone.
    mesh(caps=False)
        Build a :class:`trimesh.Trimesh` of the cone, optionally capped.
    export_mesh(name, caps=False)
        Export the cone mesh to disk.
    """

    def __init__(self, R, bottom, top):
        r"""Store cone geometry.

        Parameters
        ----------
        R : float
            Radius of the circular base.
        bottom : array-like of shape (3,)
            Base center.
        top : array-like of shape (3,)
            Cone apex.
        """
        self.R = R
        self.bottom = bottom
        self.top = top

    def pointcloud(self):
        r"""Sample circular layers along the cone axis.

        Returns
        -------
        np.ndarray
            Array of shape ``(layers, samples, 3)`` storing each ring of points.
        """

        # vectores ortonormales
        vector = self.top - self.bottom
        vector = vector/np.linalg.norm(vector)
        if np.array_equal(vector, np.array([0,0,1])):
            v1 = np.array([1,0,0])
            v2 = np.array([0,1,0])
        else:
            v1 = np.array([-vector[1], vector[0], 0])
            v1 = v1/np.linalg.norm(v1)
            v2 = np.cross(v1, vector)
        # voy a hacer 10 layers de aluras
        num_points = 50
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
        
        # Generate circle points in the plane defined by U and V
        circles = []
        for i, center in enumerate(np.linspace(self.bottom, self.top, 10)):
            circle = [center + (1-i/9)*self.R * np.cos(angle) * v1 + (1-i/9)*self.R * np.sin(angle) * v2 for angle in angles]
            circles.append(circle)
        circles = np.array(circles)
        return circles

    def mesh_data(self):
        r"""Return vertices and triangular faces describing the cone.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened vertices and corresponding face indices.
        """
        points_Cone = self.pointcloud()
        points = points_Cone.reshape(-1,3)
        triangles = []
        
        # Get number of points per circle
        num_points_per_circle = 50
        
        # Triangulate the lateral surface
        for i in range(len(points_Cone) - 1):  # Number of layers of points minus one
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        
        # Convert triangle indices to numpy array
        triangles = np.array(triangles)
        return points, triangles
    
    def mesh(self, caps = False):
        r"""Build a :class:`trimesh.Trimesh` of the cone.

        Parameters
        ----------
        caps : bool, optional
            When ``True`` the base disk is added to close the shape.

        Returns
        -------
        trimesh.Trimesh
            Triangulated cone, optionally capped.
        """
        vertices, faces = self.mesh_data()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if caps == False:
            return mesh
        else:
            tapa_abajo = Disk(center = self.bottom, normal_vector=self.top - self.bottom, R = self.R).mesh_data()
            mesh_tapa_abajo = trimesh.Trimesh(vertices = tapa_abajo[0], faces = tapa_abajo[1])
            concatenated = trimesh.util.concatenate([mesh, mesh_tapa_abajo])
            return concatenated
        
    def export_mesh(self, name, caps = False):
        r"""Export the cone mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Output file path.
        caps : bool, optional
            Mirrors :meth:`mesh` ``caps`` argument.
        """
        mesh = self.mesh(caps = caps)
        mesh.export(name)

class Sphere:
    r"""Thin wrapper around :class:`trimesh.primitives.Sphere`.

    Methods
    -------
    export_mesh(name)
        Export the sphere primitive.
    """

    def __init__(self, center, R):
        r"""Instantiate a sphere.

        Parameters
        ----------
        center : array-like of shape (3,)
            Sphere center.
        R : float
            Sphere radius.
        """
        self.center = center
        self.R = R
        self.mesh_object = trimesh.primitives.Sphere(radius=R, center=center)

    def export_mesh(self, name):
        r"""Export the primitive to ``name``.

        Parameters
        ----------
        name : str or path-like
            Destination file.
        """
        self.mesh_object.export(name)

class ControlNet:
    r"""Utility to visualize Bezier control nets via spheres and cylinders.

    Methods
    -------
    control_points_fun()
        Return the flattened control net as ``(nrows*ncols, 3)`` array.
    spheres_mesh(R)
        Concatenate :class:`Sphere` primitives centered at control points.
    edges(R)
        Create edge cylinders connecting adjacent control points.
    _mesh(spheres_radius, edges_radius)
        Combine spheres and cylinders into a single mesh.
    export_mesh_separately(names, spheres_radius, edges_radius)
        Export spheres and edges meshes independently.
    export_mesh(name, spheres_radius, edges_radius)
        Export combined control net to disk.
    """

    def __init__(self, mat):
        r"""Store control heights and precompute cartesian points.

        Parameters
        ----------
        mat : np.ndarray
            Height matrix describing the control net grid.
        """
        self.mat = mat
        self.control_points = self.control_points_fun()

    def control_points_fun(self):
        r"""Return the flattened control net as ``(nrows*ncols, 3)`` array.

        Returns
        -------
        np.ndarray
            Cartesian coordinates of each control point.
        """
        nrows = len(self.mat)
        ncols = len(self.mat.T)
        control_points = []
        for i in range(nrows):
            for j in range(ncols):
                v = np.array([i/(nrows-1), j/(ncols-1), self.mat[i,j]])
                control_points.append(v)
        control_points = np.array(control_points)
        return control_points
    
    def spheres_mesh(self, R):
        r"""Concatenate :class:`Sphere` primitives centered at control points.

        Parameters
        ----------
        R : float
            Sphere radius.

        Returns
        -------
        trimesh.Trimesh
            Combined mesh of control-point spheres.
        """
        spheres_trimesh = [Sphere(P, R).mesh_object for P in self.control_points]
        mesh = trimesh.util.concatenate(spheres_trimesh)
        return mesh
    
    def edges(self, R):
        r"""Create edge cylinders connecting adjacent control points.

        Parameters
        ----------
        R : float
            Cylinder radius for each edge.

        Returns
        -------
        trimesh.Trimesh
            Mesh containing all edge cylinders.
        """
        nrows = len(self.mat)
        ncols = len(self.mat.T)
        control_net = self.control_points.reshape(nrows, ncols, 3)
        Cylinders = []
        for i in range(nrows):
            for j in range(ncols -1):
                vertices, faces = Cylinder(bottom=control_net[i,j], top=control_net[i,j+1], radius=R).triangulation()
                Cylinder = trimesh.Trimesh(vertices=vertices, faces=faces)
                Cylinders.append(Cylinder)
        
        for j in range(ncols):
            for i in range(nrows -1):
                vertices, faces = Cylinder(bottom=control_net[i,j], top=control_net[i+1,j], radius=R).triangulation()
                Cylinder = trimesh.Trimesh(vertices=vertices, faces=faces)
                Cylinders.append(Cylinder)

        Cylinders_mesh = trimesh.util.concatenate(Cylinders)
        return Cylinders_mesh
    
    def _mesh(self, spheres_radius, edges_radius):
        r"""Combine spheres and cylinders into a single mesh.

        Parameters
        ----------
        spheres_radius : float
            Radius for control-point spheres.
        edges_radius : float
            Radius for edge cylinders.

        Returns
        -------
        trimesh.Trimesh
            Mesh containing both spheres and edges.
        """
        spheres = self.spheres_mesh(spheres_radius)
        edges = self.edges(edges_radius)
        return trimesh.util.concatenate([spheres, edges])

    def export_mesh_separately(self, names, spheres_radius, edges_radius):
        r"""Export spheres and edges meshes independently.

        Parameters
        ----------
        names : Sequence[str]
            Output file paths ``[spheres_path, edges_path]``.
        spheres_radius : float
            Radius passed to :meth:`spheres_mesh`.
        edges_radius : float
            Radius passed to :meth:`edges`.
        """
        self.spheres_mesh(spheres_radius).export(names[0])
        self.edges(edges_radius).export(names[1])

    def export_mesh(self, name, spheres_radius, edges_radius):
        r"""Export combined control net to ``name``.

        Parameters
        ----------
        name : str or path-like
            Output path for the merged mesh.
        spheres_radius : float
            Radius used for spheres.
        edges_radius : float
            Radius used for cylinders.
        """
        mesh = self._mesh(spheres_radius, edges_radius)
        mesh.export(name)

class Surface:
    r"""Convenience wrapper that exposes :class:`point_milling_backend.Surface`.

    Methods
    -------
    generate_mesh_data()
        Return vertices, faces and normals from the backend surface.
    _mesh()
        Build a :class:`trimesh.Trimesh` instance from backend data.
    export_mesh(name)
        Export the baked surface mesh to disk.
    """

    def __init__(self, matrix):
        r"""Create a meshable surface.

        Parameters
        ----------
        matrix : np.ndarray
            ``4x4`` Bezier control net.
        """
        self.matrix = matrix
        self.surface = SurfaceBackend(mat_Q=matrix)
        self.mesh = self._mesh()

    def generate_mesh_data(self):
        r"""Return vertices, faces and normals from the backend surface.

        Returns
        -------
        tuple[list[np.ndarray], matplotlib.tri.Triangulation, np.ndarray]
            Raw data produced by :mod:`point_milling_backend`.
        """
        return self.surface.point_cloud()

    def _mesh(self):
        r"""Build a :class:`trimesh.Trimesh` instance from backend data.

        Returns
        -------
        trimesh.Trimesh
            Mesh representation of the surface.
        """
        vertices, faces, vert_normals = self.generate_mesh_data()
        mesh = trimesh.Trimesh(vertices=np.array(vertices).T, faces = faces.triangles)
        return mesh

    def export_mesh(self, name):
        r"""Export the baked surface mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Destination path.
        """
        self.mesh.export(name)

class Disk:
    r"""Planar disk triangulation defined by center, normal, and radius.

    Methods
    -------
    pointcloud(num_points=50)
        Return circle samples plus center vertex.
    mesh_data(num_points=50)
        Generate vertices and fan faces for the disk.
    export_mesh(name)
        Export the disk mesh to disk.
    """

    def __init__(self, center,normal_vector, R):
        r"""Store disk definition.

        Parameters
        ----------
        center : array-like of shape (3,)
            Disk center.
        normal_vector : array-like of shape (3,)
            Plane normal.
        R : float
            Disk radius.
        """
        self.center = center
        self.normal_vector = normal_vector
        self.R = R
    
    def pointcloud(self, num_points=50):
        r"""Return circle samples plus center vertex.

        Parameters
        ----------
        num_points : int, optional
            Number of samples along the circumference.

        Returns
        -------
        np.ndarray
            Array of sampled points ordered around the perimeter plus center.
        """

        # Normalize the vector V to be the normal vector
        N = self.normal_vector / np.linalg.norm(self.normal_vector)
        
        # Create two orthogonal vectors to N using cross product trick
        # Handle the case where N is aligned with one of the axes
        if (N[0] == 0 and N[1] == 0):
            # N is parallel to the Z axis
            U = np.array([1, 0, 0])
        else:
            U = np.array([-N[1], N[0], 0])
            U = U / np.linalg.norm(U)
        
        V = np.cross(N, U)
        
        # Define the number of points on the circle
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Generate circle points in the plane defined by U and V
        circle_points = np.array([self.center + self.R * np.cos(angle) * U + self.R * np.sin(angle) * V for angle in angles])
        circle_points = np.vstack([circle_points, self.center])
        return circle_points

    def mesh_data(self, num_points=50):
        r"""Generate vertices and fan faces for the disk.

        Returns
        -------
        tuple[np.ndarray, list[list[int]]]
            Vertex array and triangle fan indices.
        """

        # Check that num_points is at least 3 and is an integer, otherwise raise an error
        if not isinstance(num_points, int) or num_points < 3:
            raise ValueError("num_points must be an integer greater than or equal to 3")
        
        # Generate the indices for the triangles
        indices = []
        for i in range(num_points):
            next_i = (i + 1) % num_points
            # Creating triangles using center, current point, and next point
            indices.append([num_points, i, next_i])
        return self.pointcloud(num_points=num_points), indices
    
    def export_mesh(self, name):
        r"""Export the disk mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Output file path.
        """
        vertices, faces = self.mesh_data()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(name)

class PipeSurface:
    r"""Generate tubular meshes around parametric curves.

    Methods
    -------
    tangent_vector(t)
        Approximate curve tangent at parameter ``t``.
    orthonormal_frame(t)
        Return Frenet-like frame ``(T, N, B)`` at ``t``.
    circle(t, resolution=50)
        Return a circle of radius ``R`` orthogonal to the curve at ``t``.
    pointcloud(resolution_circle=50, resolution_time=100)
        Sample the full pipe surface as a structured array.
    mesh_data(resolution_circle=50, resolution_time=100)
        Return vertices and faces describing the tube surface.
    mesh(caps=False, resolution_circle=50, resolution_time=100)
        Build the :class:`trimesh.Trimesh` pipe, optionally capping the ends.
    export_mesh(name, caps=False, resolution_circle=50, resolution_time=100)
        Export a discretized pipe to ``name``.
    """

    def __init__(self, curve, Range, R):
        r"""Store the swept pipe definition.

        Parameters
        ----------
        curve : callable
            Function ``t -> (3,)`` returning curve positions.
        Range : tuple[float, float]
            Parameter domain inclusive.
        R : float
            Pipe radius (must be positive).
        """
        self.curve = curve
        self.Range = Range
        if R <= 0:
            raise ValueError("Radius R must be positive")
        self.R = R

    def tangent_vector(self, t):
        r"""Approximate curve tangent at parameter ``t`` using finite differences.

        Parameters
        ----------
        t : float
            Parameter value.

        Returns
        -------
        np.ndarray
            Tangent vector normalized to unit length.
        """
        curve_flat = lambda t: self.curve(t).flatten()
        tangent = approx_fprime(t, curve_flat, epsilon=1e-6)
        return tangent.flatten()

    def orthonormal_frame(self, t):
        r"""Return Frenet-like frame ``(T, N, B)`` at ``t``.

        Parameters
        ----------
        t : float
            Parameter value inside ``Range``.

        Returns
        -------
        np.ndarray
            Array ``(3, 3)`` whose rows correspond to ``T``, ``N``, ``B``.
        """
        if t < self.Range[0] or t > self.Range[1]:
            raise ValueError("Parameter t is out of range")
        tangent = self.tangent_vector(t)
        tangent = tangent / np.linalg.norm(tangent)
        if np.array_equal(tangent , np.array([0,0,1])):
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])
        else:
            v1 = np.array([-tangent[1], tangent[0], 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(tangent, v1)
        return np.array([tangent, v1, v2])

    def circle(self, t, resolution = 50):
        r"""Return a circle of radius ``R`` orthogonal to the curve at ``t``.

        Parameters
        ----------
        t : float
            Parameter along the guiding curve.
        resolution : int, optional
            Number of angular samples.

        Returns
        -------
        np.ndarray
            Array of size ``(resolution, 3)`` describing the ring.
        """
        if not isinstance(resolution, int) or resolution < 3:
            raise ValueError("resolution must be an integer greater than or equal to 3")
        if t < self.Range[0] or t > self.Range[1]:
            raise ValueError("Parameter t is out of range")

        T, N, B = self.orthonormal_frame(t)
        center = self.curve(t)
        circle = [center + self.R * (np.cos(s) * N + np.sin(s) * B) for s in np.linspace(0, 2 * np.pi, resolution)]
        return np.array(circle)

    def pointcloud(self, resolution_circle = 50, resolution_time = 100):
        r"""Sample the full pipe surface as a structured array.

        Parameters
        ----------
        resolution_circle : int, optional
            Points per circle.
        resolution_time : int, optional
            Number of samples along the path.

        Returns
        -------
        np.ndarray
            Structured array with shape ``(resolution_time, resolution_circle, 3)``.
        """
        if not isinstance(resolution_circle, int) or resolution_circle < 3:
            raise ValueError("resolution_circle must be an integer greater than or equal to 3")
        if not isinstance(resolution_time, int) or resolution_time < 2:
            raise ValueError("resolution_time must be an integer greater than or equal to 2")
        
        return np.array([self.circle(t, resolution = resolution_circle) for t in np.linspace(self.Range[0], self.Range[1], resolution_time)])

    def mesh_data(self, resolution_circle = 50, resolution_time = 100):
        r"""Return vertices/faces describing the tube surface.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened vertices and triangular faces.
        """
        if not isinstance(resolution_circle, int) or resolution_circle < 3:
            raise ValueError("resolution_circle must be an integer greater than or equal to 3")
        if not isinstance(resolution_time, int) or resolution_time < 2:
            raise ValueError("resolution_time must be an integer greater than or equal to 2")
        
        # Generate tube points
        tube_points = self.pointcloud(resolution_circle = resolution_circle, resolution_time = resolution_time)
        points = tube_points.reshape(-1, 3)
        triangles = []
        num_points_per_circle = resolution_circle
        for i in range(len(tube_points) - 1):
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        triangles = np.array(triangles)
        return points, triangles
    
    def mesh(self, caps = False, resolution_circle = 50, resolution_time = 100):
        r"""Build the :class:`trimesh.Trimesh` pipe, optionally capping the ends.

        Returns
        -------
        trimesh.Trimesh
            Triangulated pipe surface.
        """
        if not isinstance(resolution_circle, int) or resolution_circle < 3:
            raise ValueError("resolution_circle must be an integer greater than or equal to 3")
        if not isinstance(resolution_time, int) or resolution_time < 2:
            raise ValueError("resolution_time must be an integer greater than or equal to 2")
        vertices, faces = self.mesh_data(resolution_circle = resolution_circle, resolution_time = resolution_time)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if caps == False:
            return mesh
        else:
            tapa_abajo = Disk(center = self.curva(self.Range[0]), normal_vector=self.tangent_vector(self.Range[0]), R = self.R).triangulation()
            tapa_arriba = Disk(center = self.curva(self.Range[1]), normal_vector=self.tangent_vector(self.Range[1]), R = self.R).triangulation()
            mesh_tapa_abajo = trimesh.Trimesh(vertices = tapa_abajo[0], faces = tapa_abajo[1])
            mesh_tapa_arriba = trimesh.Trimesh(vertices = tapa_arriba[0], faces = tapa_arriba[1])
            concatenated = trimesh.util.concatenate([mesh, mesh_tapa_abajo, mesh_tapa_arriba])
            return concatenated
        
    def export_mesh(self, name, caps = False, resolution_circle = 50, resolution_time = 100):
        r"""Export a discretized pipe to ``name``."""
        if not isinstance(resolution_circle, int) or resolution_circle < 3:
            raise ValueError("resolution_circle must be an integer greater than or equal to 3")
        if not isinstance(resolution_time, int) or resolution_time < 2:
            raise ValueError("resolution_time must be an integer greater than or equal to 2")
        mesh = self.mesh(caps = caps, resolution_circle = resolution_circle, resolution_time = resolution_time)
        mesh.export(name)

class Cylinder:
    r"""Create a cylinder via :mod:`trimesh.creation` aligned with arbitrary endpoints.

    Parameters
    ----------
    bottom : array-like of shape (3,)
        Base point of the cylinder axis.
    top : array-like of shape (3,)
        End point of the cylinder axis.
    radius : float
        Cylinder radius.
    sections : int, optional
        Number of radial segments passed to ``trimesh.creation.cylinder`` (default ``32``).

    Attributes
    ----------
    bottom : np.ndarray
        Stored copy of the base point.
    top : np.ndarray
        Stored copy of the top point.
    radius : float
        Cylinder radius.
    sections : int
        Radial tessellation used for the procedural mesh.

    Methods
    -------
    mesh()
        Return a :class:`trimesh.Trimesh` oriented between ``bottom`` and ``top``.
    mesh_data()
        Return the vertices and faces of the procedurally generated cylinder.
    triangulation(angle_resolution=128, height_resolution=64, caps=False, cap_resolution=None)
        Return a Delaunay triangulation of the cylinder surface.
    delaunay_mesh(angle_resolution=128, height_resolution=64, caps=False, cap_resolution=None)
        Return a :class:`trimesh.Trimesh` generated via Delaunay triangulation.
    export_mesh(name)
        Export the procedurally generated cylinder.
    """

    def __init__(self, bottom, top, radius, sections=64):
        r"""Store endpoints, radius, and tessellation detail."""
        self.bottom = np.asarray(bottom, dtype=float)
        self.top = np.asarray(top, dtype=float)
        self.radius = float(radius)
        self.axis = self.top - self.bottom
        if sections < 3:
            raise ValueError("sections must be >= 3")
        self.sections = int(sections)

    def mesh(self):
        r"""Return a :class:`trimesh.Trimesh` oriented between ``bottom`` and ``top``.

        Returns
        -------
        trimesh.Trimesh
            Procedurally generated cylinder.
        """
        base_mesh = trimesh.creation.cylinder(radius=self.radius, segment=np.array([self.bottom, self.top]), sections=self.sections)
        return base_mesh

    def mesh_data(self):
        r"""Return the vertices and faces of the procedurally generated cylinder.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Vertex and face arrays.
        """
        mesh = self.mesh()
        return mesh.vertices, mesh.faces

    def triangulation(self, angle_resolution=128, height_resolution=64, caps=False, cap_resolution=None):
        r"""Return a Delaunay triangulation of the cylinder surface.

        Parameters
        ----------
        angle_resolution : int, optional
            Number of samples around the circumference (must be >= 3).
        height_resolution : int, optional
            Number of samples along the axis (must be >= 2).
        caps : bool, optional
            When ``True`` add top and bottom disks to close the mesh.
        cap_resolution : int, optional
            Samples for the cap fans; defaults to ``angle_resolution``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Vertices and triangular faces describing the mesh.
        """
        if not isinstance(angle_resolution, int) or angle_resolution < 3:
            raise ValueError("angle_resolution must be an integer greater than or equal to 3")
        if not isinstance(height_resolution, int) or height_resolution < 2:
            raise ValueError("height_resolution must be an integer greater than or equal to 2")
        if cap_resolution is None:
            cap_resolution = angle_resolution
        if not isinstance(cap_resolution, int) or cap_resolution < 3:
            raise ValueError("cap_resolution must be an integer greater than or equal to 3")

        axis_length = np.linalg.norm(self.axis)
        if axis_length == 0:
            raise ValueError("Cylinder axis has zero length")
        axis_dir = self.axis / axis_length
        if np.allclose(axis_dir, np.array([0, 0, 1])):
            u = np.array([1, 0, 0])
        else:
            u = np.array([-axis_dir[1], axis_dir[0], 0])
            u = u / np.linalg.norm(u)
        v = np.cross(axis_dir, u)

        angles = np.linspace(0, 2 * np.pi, angle_resolution, endpoint=False)
        heights = np.linspace(0.0, 1.0, height_resolution)
        angle_grid, height_grid = np.meshgrid(angles, heights)
        params = np.column_stack([angle_grid.ravel(), height_grid.ravel()])

        delaunay = Delaunay(params)
        faces = delaunay.simplices.astype(int)

        seam_faces = []
        for h_idx in range(height_resolution - 1):
            base = h_idx * angle_resolution
            next_base = (h_idx + 1) * angle_resolution
            last = base + angle_resolution - 1
            next_last = next_base + angle_resolution - 1
            seam_faces.append([last, base, next_last])
            seam_faces.append([base, next_base, next_last])
        if seam_faces:
            faces = np.vstack([faces, np.array(seam_faces, dtype=int)])

        center_grid = self.bottom + height_grid[..., None] * self.axis
        offsets = self.radius * (
            np.cos(angle_grid)[..., None] * u + np.sin(angle_grid)[..., None] * v
        )
        vertices = (center_grid + offsets).reshape(-1, 3)

        if caps:
            bottom_vertices, bottom_faces = Disk(center=self.bottom, normal_vector=-axis_dir, R=self.radius).mesh_data(num_points=cap_resolution)
            top_vertices, top_faces = Disk(center=self.top, normal_vector=axis_dir, R=self.radius).mesh_data(num_points=cap_resolution)

            bottom_faces = np.array(bottom_faces, dtype=int) + len(vertices)
            top_faces = np.array(top_faces, dtype=int) + len(vertices) + len(bottom_vertices)

            vertices = np.vstack([vertices, bottom_vertices, top_vertices])
            faces = np.vstack([faces, bottom_faces, top_faces])

        return vertices, faces

    def delaunay_mesh(self, angle_resolution=128, height_resolution=64, caps=False, cap_resolution=None):
        r"""Return a :class:`trimesh.Trimesh` generated via Delaunay triangulation."""
        vertices, faces = self.triangulation(
            angle_resolution=angle_resolution,
            height_resolution=height_resolution,
            caps=caps,
            cap_resolution=cap_resolution,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def export_mesh(self, name):
        r"""Export the procedurally generated cylinder.

        Parameters
        ----------
        name : str or path-like
            Output file path.
        """
        self.mesh().export(name)

class RuledSurface:
    r"""Interpolate between two parametric curves to form a ruled surface.

    Methods
    -------
    point(s, t)
        Evaluate the ruled surface at ``s`` and ``t``.
    pointcloud(s_resolution=100, t_resolution=100)
        Return a structured grid sampling the ruled surface.
    mesh_data(s_resolution=100, t_resolution=100)
        Return vertices and faces describing the ruled surface.
    mesh(s_resolution=100, t_resolution=100)
        Build the :class:`trimesh.Trimesh` for the ruled surface.
    export_mesh(name, s_resolution=100, t_resolution=100)
        Export the ruled surface mesh to ``name``.
    """

    def __init__(self, curve1, curve2):
        r"""Store the bounding curves."""
        self.curve1 = curve1
        self.curve2 = curve2

    def point(self,s,t):
        r"""Evaluate the ruled surface at ``s`` and ``t``.

        Returns
        -------
        np.ndarray
            Point on the surface corresponding to ``(s, t)``.
        """
        extremo1, extremo2 = self.curve1(t), self.curve2(t)
        midpoint = s*extremo1 + (1-s)*extremo2
        return midpoint
    def pointcloud(self, s_resolution = 100, t_resolution = 100):
        r"""Return a structured grid sampling the ruled surface.

        Returns
        -------
        np.ndarray
            Array of shape ``(t_resolution, s_resolution, 3)``.
        """
        if not isinstance(s_resolution, int) or s_resolution < 2:
            raise ValueError("s_resolution must be an integer greater than or equal to 2")
        if not isinstance(t_resolution, int) or t_resolution < 2:
            raise ValueError("t_resolution must be an integer greater than or equal to 2")
        
        return np.array([np.array([self.point(s,t) for s in np.linspace(0,1,s_resolution)]) for t in np.linspace(0,1,t_resolution)])

    def mesh_data(self, s_resolution = 100, t_resolution = 100):
        r"""Return vertices and faces describing the ruled surface.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened vertices plus triangular faces.
        """
        if not isinstance(s_resolution, int) or s_resolution < 2:
            raise ValueError("s_resolution must be an integer greater than or equal to 2")
        if not isinstance(t_resolution, int) or t_resolution < 2:
            raise ValueError("t_resolution must be an integer greater than or equal to 2")
        # Generate cylinder points
        cylinder_points = self.pointcloud(s_resolution = s_resolution, t_resolution = t_resolution)
        
        # Reshape the points into (N, 3) array where N is number of points
        points = cylinder_points.reshape(-1, 3)
        
        # Correctly calculate the circle indices
        # We need to base the triangulation on the first circle of the cylinder
        # first_circle_points = points[:50]  # Assuming each circle has 50 points
        # base_triangulation = Delaunay(first_circle_points[:, :2])  # 2D triangulation based on x, y coordinates
        
        # Create an array to hold the triangles
        triangles = []
        
        # Get number of points per circle
        num_points_per_circle = 100
        
        # Triangulate the lateral surface
        for i in range(len(cylinder_points) - 1):  # Number of layers of points minus one
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        
        # Convert triangle indices to numpy array
        triangles = np.array(triangles)
        
        return points, triangles
    
    def mesh(self, s_resolution = 100, t_resolution = 100):
        r"""Build the :class:`trimesh.Trimesh` for the ruled surface.

        Returns
        -------
        trimesh.Trimesh
            Mesh describing the ruled surface.
        """
        if not isinstance(s_resolution, int) or s_resolution < 2:
            raise ValueError("s_resolution must be an integer greater than or equal to 2")
        if not isinstance(t_resolution, int) or t_resolution < 2:
            raise ValueError("t_resolution must be an integer greater than or equal to 2")
        vertices, faces = self.mesh_data(s_resolution = s_resolution, t_resolution = t_resolution)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def export_mesh(self, name, s_resolution = 100, t_resolution = 100):
        r"""Export the ruled surface mesh to ``name``."""
        if not isinstance(s_resolution, int) or s_resolution < 2:
            raise ValueError("s_resolution must be an integer greater than or equal to 2")
        if not isinstance(t_resolution, int) or t_resolution < 2:
            raise ValueError("t_resolution must be an integer greater than or equal to 2")
        mesh = self.mesh(s_resolution = s_resolution, t_resolution = t_resolution)
        mesh.export(name)

class CircleEnvelope:
    r"""Envelope surface generated by sweeping circles along a path.

    Methods
    -------
    axis_normalized()
        Return normalized axis vectors for each sample.
    triangulation()
        Triangulate the parameter space spanned by angle and time.
    pointcloud(resolution_circle=None)
        Return the sampled envelope points.
    mesh(resolution_circle=None)
        Build a :class:`trimesh.Trimesh` envelope.
    export_mesh(name, resolution_circle=None)
        Export the envelope mesh to ``name``.
    """

    def __init__(self, contact_points, centers,axis, R, angle_resolution = 100):
        r"""Store the envelope sample data."""
        self.contact_points = contact_points
        self.centers = centers
        self.axis = axis
        self.normal_axis = self.axis_normalized()
        self.R = R
        self.angle_resolution = angle_resolution
        self.time_resolution = len(self.contact_points)
        self.simplices = self.triangulation()

    def axis_normalized(self):
        r"""Return normalized axis vectors for each sample.

        Returns
        -------
        np.ndarray
            Array with the same shape as ``axis`` containing unit normals.
        """
        return self.axis/np.linalg.norm(self.axis, axis = 1)[:, None]
    
    def triangulation(self):
        r"""Triangulate the parameter space spanned by angle and time.

        Returns
        -------
        np.ndarray
            Triangle indices over the structured parameter grid.
        """
        angles = np.linspace(0, 2*np.pi, self.angle_resolution)
        times = np.linspace(0,1, self.time_resolution)
        angles1,times1=np.meshgrid(angles,times)
        angles1=angles1.flatten()
        times1=times1.flatten()
        points2D=np.vstack([angles1, times1]).T
        tri = Delaunay(points2D)
        simplices = tri.simplices
        return simplices
    
    def pointcloud(self, resolution_circle = None):
        r"""Return the sampled envelope points.

        Parameters
        ----------
        resolution_circle : int, optional
            Number of samples per circle; defaults to ``angle_resolution``.

        Returns
        -------
        np.ndarray
            Array of shape ``(time_resolution, resolution_circle, 3)``.
        """
        if resolution_circle is None:
            resolution_circle = self.angle_resolution
        V1 = (1/self.R)*(self.contact_points - self.centers) # Normalized vector radius
        V2 = np.cross(self.normal_axis, V1) # the binormal

        # do the points computations
        angles = np.linspace(0,2*np.pi, resolution_circle)
        s = angles
        cosS,sinS = np.cos(s), np.sin(s)
        sinS, cosS = sinS.reshape(1, -1, 1), cosS.reshape(1, -1, 1)
        V1, V2 = V1.reshape(V1.shape[0], 1, V1.shape[1]), V2.reshape(V2.shape[0], 1, V2.shape[1])

        arc_points = self.R * (cosS*V1 + sinS*V2)
        points_envelope = self.centers[:, np.newaxis, :] + arc_points
        return points_envelope
    
    def mesh(self, resolution_circle = None):
        r"""Build a :class:`trimesh.Trimesh` envelope.

        Returns
        -------
        trimesh.Trimesh
            Triangular mesh for the swept envelope.
        """
        vertices = self.pointcloud(resolution_circle=  resolution_circle)
        mesh = trimesh.Trimesh(vertices=vertices.reshape(-1,3), faces=self.simplices)
        return mesh
    
    def export_mesh(self, name, resolution_circle = None):
        r"""Export the envelope mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Destination path.
        resolution_circle : int, optional
            Passed to :meth:`pointcloud`.
        """
        mesh = self.mesh(resolution_circle=resolution_circle)
        mesh.export(name)

class Heatmap:
    r"""Attach scalar data to meshes and visualize/export them.

    Methods
    -------
    scalars_to_colors()
        Convert scalar field to RGBA colors using the configured colorscale.
    _mesh_with_colors()
        Return a copy of the mesh decorated with per-vertex colors.
    export_mesh(name)
        Export the colored mesh to ``name`` (PLY enforced).
    plotly_figure(showlegend=True, name='Heatmap')
        Create a Plotly figure visualizing the colored surface.
    """

    def __init__(self, mesh, array_of_scalars, upper_bound, lower_bound,  colorscale = 'turbo_r'):
        r"""Store data bounds and precompute vertex colors.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Geometry to annotate.
        array_of_scalars : np.ndarray
            Scalar per vertex.
        upper_bound : float
            Upper display bound.
        lower_bound : float
            Lower display bound.
        colorscale : str, optional
            Plotly colorscale name.
        """
        self.mesh = mesh
        self.array_of_scalars = array_of_scalars
        self.colorscale = colorscale
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.rgba_colors = self.scalars_to_colors()
        self.mesh_with_colors = self._mesh_with_colors()
    
    def scalars_to_colors(self):
        r"""Convert scalar field to RGBA colors using the configured colorscale.

        Returns
        -------
        np.ndarray
            RGBA values in ``uint8`` format.
        """

        normalized_values = (self.array_of_scalars - self.lower_bound) / (self.upper_bound - self.lower_bound)
        normalized_values = np.clip(normalized_values, 0, 1)  # Ensure values are within [0, 1]

        # Step 2: Retrieve the colorscale
        colorscale = get_colorscale(self.colorscale)  # Or any other Plotly colorscale

        # Step 3: Sample the colorscale
        colors = sample_colorscale(colorscale, normalized_values, colortype='rgb')

        rgba_colors = []
        for color in colors:
            rgba_colors.append(unlabel_rgb(color))

        rgb_values = np.array(rgba_colors)
        # Step 2: Create an alpha channel (fully opaque)
        alpha_channel = np.full((rgb_values.shape[0], 1), 255, dtype=np.uint8)

        # Step 3: Combine RGB and alpha to get RGBA
        rgba_values = np.hstack((rgb_values, alpha_channel))
        return rgba_values

    def _mesh_with_colors(self):
        r"""Return a copy of the mesh decorated with per-vertex colors.

        Returns
        -------
        trimesh.Trimesh
            Mesh with ``vertex_colors`` assigned.
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        normals = self.mesh.vertex_normals
        vertex_colors = self.rgba_colors
        
        mesh_with_colors = trimesh.Trimesh(vertices=vertices,
                                            faces=faces,
                                            vertex_normals=normals,
                                            vertex_colors=vertex_colors)
        return mesh_with_colors

    def export_mesh(self, name):
        r"""Export the colored mesh to ``name`` (PLY enforced).

        Parameters
        ----------
        name : str or path-like
            Output path ending with ``.ply``.
        """
        if not name.lower().endswith('.ply'):
            raise ValueError("Error: The file must have a '.ply' extension, otherwise there will be no colors!")
        mesh = self.mesh_with_colors 
        mesh.export(name)

    def plotly_figure(self, showlegend = True, name = 'Heatmap'):
        r"""Create a Plotly figure visualizing the colored surface.

        Parameters
        ----------
        showlegend : bool, optional
            Whether to display legend entries.
        name : str, optional
            Trace label.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive heatmap figure.
        """

        fig = go.Figure()
        cmin, cmax =  self.lower_bound, self.upper_bound

        # make undercutting and overcutting points be the maximum value that we allow
        distances_adjusted_to_tolerances = self.array_of_scalars.copy()
        distances_adjusted_to_tolerances[distances_adjusted_to_tolerances > cmax] = cmax # set undercut
        distances_adjusted_to_tolerances[distances_adjusted_to_tolerances < cmin] = cmin # set overcut
        
        x,y,z = self.mesh.vertices.T
        i,j,k = self.mesh.faces.T
        # plot the heatmap
        fig.add_mesh3d(x = x, y = y, z = z,
                       i=i, j=j, k=k,
                       colorscale = self.colorscale,
                       cmin=cmin,
                       cmax=cmax,
                       intensity=distances_adjusted_to_tolerances,
                       name=name,
                       showscale=True,
                       showlegend=showlegend
        )

        fig.update_layout(
            showlegend = True,
            scene=dict(
                aspectmode='data'),
                width = 900,
                height = 750
            )
        return fig

class Torus:
    r"""Procedurally create a torus oriented along an arbitrary axis.

    Methods
    -------
    mesh()
        Build the rotated torus as a :class:`trimesh.Trimesh`.
    export_mesh(name)
        Export the torus mesh to disk.
    """

    def __init__(self, center, axis, R_major, R_minor):
        r"""Store the torus parameters."""
        self.center = center
        self.axis = axis
        self.R_major = R_major
        self.R_minor = R_minor

    def mesh(self):
        r"""Build the rotated torus as a :class:`trimesh.Trimesh`.

        Returns
        -------
        trimesh.Trimesh
            Torus mesh aligned to ``axis`` and centered at ``center``.
        """
        segment_length = np.linalg.norm(self.axis)
        if segment_length == 0:
            raise ValueError("The axis vector cannot be zero")
        unit_segment_vector = self.axis / segment_length

        # Create the torus aligned with the Z axis at the origin
        torus = trimesh.creation.torus(major_radius=self.R_major, minor_radius=self.R_minor)

        # Compute rotation matrix to align Z axis to unit_segment_vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, unit_segment_vector)
        rotation_angle = np.arccos(np.dot(z_axis, unit_segment_vector))

        if np.linalg.norm(rotation_axis) != 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3]
        else:
            rotation_matrix = np.eye(3)  # No rotation needed

        # Apply rotation and translation to the torus
        torus.apply_transform(trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis, point=[0, 0, 0]))
        torus.apply_translation(self.center)

        return torus
    
    def export_mesh(self, name):
        r"""Export the torus mesh to ``name``.

        Parameters
        ----------
        name : str or path-like
            Destination path.
        """
        mesh = self.mesh()
        mesh.export(name)

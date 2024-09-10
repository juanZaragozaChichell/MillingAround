import numpy as np
import trimesh
from scipy.optimize import approx_fprime
# from scipy.spatial import Delaunay
from point_milling_6 import Surface

class Flecha:
    def __init__(self, R1, R2, bottom, top, t):
        self.R1 = R1 # radio del cilindro
        self.R2 = R2 # radio del cono
        self.bottom = bottom
        self.top = top
        self.t = t

    def mesh(self):
        start_cone = (1-self.t)*self.bottom + self.t*self.top
        cono = Cono(self.R2, start_cone, self.top)
        cono_mesh = cono.mesh(tapas=True)
        cilindro_mesh = Cilindro(bottom=self.bottom, top=start_cone, radius=self.R1).mesh(tapas=True)
        mesh = trimesh.util.concatenate([cono_mesh, cilindro_mesh])
        return mesh
    def pointcloud(self):
        mesh = self.mesh()
        vertices = mesh.vertices
        return vertices
    def triangulation(self):
        mesh = self.mesh()
        return mesh.vertices, mesh.faces
    def to_Rhino(self, name):
        mesh = self.mesh()
        mesh.export(name)


class Cono:
    def __init__(self, R, bottom, top):
        self.R = R
        self.bottom = bottom
        self.top = top

    def pointcloud(self):
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
        circulos = []
        for i, centro in enumerate(np.linspace(self.bottom, self.top, 10)):
            circulo = [centro + (1-i/9)*self.R * np.cos(angle) * v1 + (1-i/9)*self.R * np.sin(angle) * v2 for angle in angles]
            circulos.append(circulo)
        circulos = np.array(circulos)
        return circulos

    def triangulation(self):
        puntos_cono = self.pointcloud()
        puntos = puntos_cono.reshape(-1,3)
        triangles = []
        
        # Get number of points per circle
        num_points_per_circle = 50
        
        # Triangulate the lateral surface
        for i in range(len(puntos_cono) - 1):  # Number of layers of points minus one
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        
        # Convert triangle indices to numpy array
        triangles = np.array(triangles)
        
        return puntos, triangles
    def mesh(self, tapas = False):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if tapas == False:
            return mesh
        else:
            tapa_abajo = Disco(centro = self.bottom, normal_vector=self.top - self.bottom, R = self.R).triangulation()
            mesh_tapa_abajo = trimesh.Trimesh(vertices = tapa_abajo[0], faces = tapa_abajo[1])
            concatenated = trimesh.util.concatenate([mesh, mesh_tapa_abajo])
            return concatenated
        
    def to_Rhino(self, name, tapas = False):
        mesh = self.mesh(tapas = tapas)
        mesh.export(name)

class Esfera:
    def __init__(self, centro, R):
        self.centro = centro
        self.R = R
        self.mesh_object = trimesh.primitives.Sphere(radius=R, center=centro)
    def triangulation(self):
        vertices, faces = self.mesh_object.vertices, self.mesh_object.faces
        return vertices, faces
    def to_Rhino(self, name):
        self.mesh_object.export(name)

class ControlNet:
    def __init__(self, mat):
        self.mat = mat
        self.control_points = self.control_points_fun()
    def control_points_fun(self):
        nrows = len(self.mat)
        ncols = len(self.mat.T)
        control_points = []
        for i in range(nrows):
            for j in range(ncols):
                v = (1/3)*np.array([i,j, 3*self.mat[i,j]])
                control_points.append(v)
        control_points = np.array(control_points)
        return control_points
    
    def esferas_mesh(self, R):
        esferas_trimesh = [Esfera(P, R).mesh_object for P in self.control_points]
        mesh = trimesh.util.concatenate(esferas_trimesh)
        return mesh
    
    def aristas(self, R):
        nrows = len(self.mat)
        ncols = len(self.mat.T)
        control_net = self.control_points.reshape(nrows, ncols, 3)
        cilindros = []
        for i in range(nrows):
            for j in range(ncols -1):
                vertices, faces = Cilindro(bottom=control_net[i,j], top=control_net[i,j+1], radius=R).triangulation()
                cilindro = trimesh.Trimesh(vertices=vertices, faces=faces)
                cilindros.append(cilindro)
        
        for j in range(ncols):
            for i in range(nrows -1):
                vertices, faces = Cilindro(bottom=control_net[i,j], top=control_net[i+1,j], radius=R).triangulation()
                cilindro = trimesh.Trimesh(vertices=vertices, faces=faces)
                cilindros.append(cilindro)

        cilindros_mesh = trimesh.util.concatenate(cilindros)
        return cilindros_mesh
    
    def to_Rhino(self, names, R):
        self.esferas_mesh(R).export(names[0])
        self.aristas(0.7*R).export(names[1])

class Superficie:
    def __init__(self, matriz):
        self.matriz = matriz
        self.surface = Surface(mat_Q=matriz)
    
    def triangulation(self):
        return self.surface.point_cloud()
    
    def to_Rhino(self, name):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=np.array(vertices).T, faces = faces.triangles)
        mesh.export(name)

class Disco:
    def __init__(self, centro,normal_vector, R):
        self.centro = centro
        self.normal_vector = normal_vector
        self.R = R
    def pointcloud(self):

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
        num_points = 50
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Generate circle points in the plane defined by U and V
        circle_points = np.array([self.centro + self.R * np.cos(angle) * U + self.R * np.sin(angle) * V for angle in angles])
        circle_points = np.vstack([circle_points, self.centro])
        return circle_points

    def triangulation(self):
        num_points = 50
        indices = []
        for i in range(num_points):
            next_i = (i + 1) % num_points
            # Creating triangles using center, current point, and next point
            indices.append([num_points, i, next_i])
        return self.pointcloud(), indices
    
    def to_Rhino(self, name):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(name)

class Tubo:
    def __init__(self, curva, Range, R):
        self.curva = curva
        self.Range = Range
        self.R = R

    def tangent_vector(self, t):
        curva_flat = lambda t: self.curva(t).flatten()
        tangent = approx_fprime(t, curva_flat, epsilon=1e-6)
        return tangent.flatten()

    def orthonormal_frame(self, t):
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

    def circle(self, t):
        T, N, B = self.orthonormal_frame(t)
        centro = self.curva(t)
        circulo = [centro + self.R * (np.cos(s) * N + np.sin(s) * B) for s in np.linspace(0, 2 * np.pi, 50)]
        return np.array(circulo)

    def pointcloud(self):
        return np.array([self.circle(t) for t in np.linspace(self.Range[0], self.Range[1], 50)])

    def triangulation(self):
        tube_points = self.pointcloud()
        points = tube_points.reshape(-1, 3)
        triangles = []
        num_points_per_circle = 50
        for i in range(len(tube_points) - 1):
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        triangles = np.array(triangles)
        return points, triangles
    def mesh(self, tapas = False):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if tapas == False:
            return mesh
        else:
            tapa_abajo = Disco(centro = self.curva(self.Range[0]), normal_vector=self.tangent_vector(self.Range[0]), R = self.R).triangulation()
            tapa_arriba = Disco(centro = self.curva(self.Range[1]), normal_vector=self.tangent_vector(self.Range[1]), R = self.R).triangulation()
            mesh_tapa_abajo = trimesh.Trimesh(vertices = tapa_abajo[0], faces = tapa_abajo[1])
            mesh_tapa_arriba = trimesh.Trimesh(vertices = tapa_arriba[0], faces = tapa_arriba[1])
            concatenated = trimesh.util.concatenate([mesh, mesh_tapa_abajo, mesh_tapa_arriba])
            return concatenated
        
    def to_Rhino(self, name, tapas = False):
        mesh = self.mesh(tapas = tapas)
        mesh.export(name)

class Cilindro:
    def __init__(self, bottom, top, radius):
        self.bottom = bottom
        self.top = top
        self.R = radius
    
    def pointcloud(self):
        translation_vector = self.top - self.bottom
        unit_trans_vect = translation_vector / np.linalg.norm(translation_vector)
        if np.array_equal(unit_trans_vect , np.array([0,0,1])):
            v1 = np.array([1,0,0])
            v2 = np.array([0,1,0])
        else:
            v1 = np.array([-unit_trans_vect[1], unit_trans_vect[0], 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(v1, unit_trans_vect)
        
        centro = self.bottom
        circulo = centro + self.R * np.array([np.cos(s) * v1 + np.sin(s) * v2 for s in np.linspace(0, 2 * np.pi, 50)])
        cylinder = np.array([i * translation_vector + circulo for i in np.linspace(0, 1, 50)])
        return cylinder

    def triangulation(self):
        # Generate cylinder points
        cylinder_points = self.pointcloud()
        
        # Reshape the points into (N, 3) array where N is number of points
        points = cylinder_points.reshape(-1, 3)
        
        # Correctly calculate the circle indices
        # We need to base the triangulation on the first circle of the cylinder
        # first_circle_points = points[:50]  # Assuming each circle has 50 points
        # base_triangulation = Delaunay(first_circle_points[:, :2])  # 2D triangulation based on x, y coordinates
        
        # Create an array to hold the triangles
        triangles = []
        
        # Get number of points per circle
        num_points_per_circle = 50
        
        # Triangulate the lateral surface
        for i in range(len(cylinder_points) - 1):  # Number of layers of points minus one
            for j in range(num_points_per_circle):
                next_j = (j + 1) % num_points_per_circle
                triangles.append([i * num_points_per_circle + j, i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
                triangles.append([i * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + next_j, (i + 1) * num_points_per_circle + j])
        
        # Convert triangle indices to numpy array
        triangles = np.array(triangles)
        
        return points, triangles
    def mesh(self, tapas = False):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if tapas == False:
            return mesh
        else:
            tapa_abajo = Disco(centro = self.bottom, normal_vector=self.top - self.bottom, R = self.R).triangulation()
            tapa_arriba = Disco(centro = self.top, normal_vector=self.top - self.bottom, R = self.R).triangulation()
            mesh_tapa_abajo = trimesh.Trimesh(vertices = tapa_abajo[0], faces = tapa_abajo[1])
            mesh_tapa_arriba = trimesh.Trimesh(vertices = tapa_arriba[0], faces = tapa_arriba[1])
            concatenated = trimesh.util.concatenate([mesh, mesh_tapa_abajo, mesh_tapa_arriba])
            return concatenated

    def to_Rhino(self, name, tapas = False):
        mesh = self.mesh(tapas = tapas)
        mesh.export(name)

class RuledSurface:
    def __init__(self, curva1, curva2):
        self.curva1 = curva1
        self.curva2 = curva2
    def punto(self,s,t):
        extremo1, extremo2 = self.curva1(t), self.curva2(t)
        midpoint = s*extremo1 + (1-s)*extremo2
        return midpoint
    def pointcloud(self):
        return np.array([np.array([self.punto(s,t) for s in np.linspace(0,1,100)]) for t in np.linspace(0,1,100)])

    def triangulation(self):
            # Generate cylinder points
            cylinder_points = self.pointcloud()
            
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
    
    def mesh(self):
        vertices, faces = self.triangulation()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def to_Rhino(self, name):
        mesh = self.mesh()
        mesh.export(name)

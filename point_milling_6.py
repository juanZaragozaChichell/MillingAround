""" Point Milling Library

This script provides the essential methods regarding point milling with a
toroidal cutter. Actual stuff is dealt on the offset, allowing us to
simplify both the math and the codigon. 

The script contains multiple classes that intend to conceptualize the
different physical objects that participate on all the process, e.g.:

    1. Surface (Offset surface).
    2. BezierPatch (rotation / tilt functions).
    3. Envelope (envelope of the medial circle of the toroidal cutter).
    4. Intersection (intersection of the envelopes).

Multiple functions that are used for the different geometric computations
are globally defined, e.g.:

    * du_dv
    * normal
    * duu_duv
    * dvv
    * Bernstein

This script requires numpy and scipy.optimize.approx_fprime and 
scipy.optimize.fsolve. This script is, as a matter of fact, intended to
be imported as a module.

This version of the library assumes that the Parametric Domain of the surface
and all is [[0,1], [0,1]] so that things are less complicated
"""

import numpy as np
from scipy.optimize import approx_fprime, fsolve
from skspatial.objects import Plane
import matplotlib.tri as mtri

Pi = np.pi
#%% callable methods

def get_coefficients_of_cubic_at_z(X, z):
    '''Returns the coefficients of a Bezier cubic curve whose control points are given by X
    and we want to solve for z
    
    Parameters
    ----------
    X: array or list of length 4
    z: float
    
    Returns
    -------
    array '''
    q0,q1,q2,q3 = X
    a = q3 +3*(q1 - q2) - q0
    b = 3*(q2 - 2*q1 + q0)
    c = 3*(q1-q0)
    d = q0 - z
    return np.array([a,b,c,d])

def roots_of_cubic_equation(X):
    """
    Computes the real roots of a cubic polynomial defined by its coefficients.

    Args:
        X (array-like): An array or list of coefficients for the cubic polynomial. The coefficients
        are expected in the order of decreasing powers, i.e., [a, b, c, d] for ax^3 + bx^2 + cx + d.

    Returns:
        np.ndarray: An array containing the real roots of the cubic polynomial. Complex roots are ignored,
        and only the real parts of purely real roots are returned.

    Example:
        # To find the roots of the polynomial 3x^3 + 2x^2 - x - 6
        roots = roots_of_cubic_equation([3, 2, -1, -6])
    """
    polyroots = np.poly1d(X).roots # compute roots
    real_roots = polyroots[polyroots.imag == 0].real # get only the real roots and only its real part
    return real_roots


def get_coordinates(V,V1,V2):
    """
    Computes the coordinates of vector V in the basis formed by vectors V1 and V2.
    Assumes that V can be expressed as a linear combination of V1 and V2: V = A*V1 + B*V2.
    This function solves for coefficients A and B.

    Parameters
    ----------
    V : np.ndarray
        The vector(s) whose coordinates are to be determined. Can be a single vector (1D array)
        or a set of vectors (2D array where each row represents a vector).
    V1 : np.ndarray
        The first basis vector (1D array) or set of vectors (2D array).
    V2 : np.ndarray
        The second basis vector (1D array) or set of vectors (2D array).

    Returns
    -------
    (A, B) : tuple
        The coefficients A and B such that V = A*V1 + B*V2. Both A and B are floats if V is a single
        vector (V.ndim == 1). If V is a set of vectors (V.ndim == 2), A and B are arrays where each
        element corresponds to the coefficients for the respective vector in V.

    Notes
    -----
    The function requires that V, V1, and V2 have compatible dimensions. If V.ndim is 2, then V1 and V2
    must also be 2D with the same number of vectors as V or be broadcastable to that shape.
    """
    
    if V.ndim == 1:
        c1,c2 = np.array([V1,V2]).dot(V)
        v12 = np.dot(V1,V1)
        v1v2, v22 = np.array([V1,V2]).dot(V2)
        A = (c2*v1v2 - c1*v22)/(v1v2**2 - v12*v22)
        B = (c1*v1v2 - c2*v12)/(v1v2**2 - v12*v22)
        return (A,B)
    elif V.ndim == 2:
        dot_product_combined_einsum = np.einsum('ij,ij->i', np.concatenate([V1,V2], axis=0), np.tile(V, (2, 1)))
        # Splitting the combined results back into separate results for v1 and v
        split_index = len(dot_product_combined_einsum) // 2
        c1, c2 = dot_product_combined_einsum[:split_index], dot_product_combined_einsum[split_index:]
        v12 = np.einsum('ij,ij->i', V1,V1)
        dot_product_combined_einsum = np.einsum('ij,ij->i', np.concatenate([V1,V2], axis=0), np.tile(V2, (2, 1)))
        v1v2, v22 = dot_product_combined_einsum[:split_index], dot_product_combined_einsum[split_index:]
        divisor = 1/(v1v2**2 - v12*v22)
        A = divisor*(c2*v1v2 - c1*v22)
        B = divisor*(c1*v1v2 - c2*v12)
        return (A,B)
    else:
        return print('bad format in get_coordinates function.')


def du_dv(surface, X):
    """Gets function and point and returns the vector of the derivatives
    with respect to both variables at the given point.

    Parameters
    ----------
    surface: function
        Function surface: R^2 ---> R^3 which is assumed to be differentiable.
    X: np.ndarray
        Point(s) at which to compute the derivative.
    
    Returns
    -------
    np.ndarray
        If X is a single point, returns the derivative at such point.
        If X is a list of points, returns a matrix in which each row
        is the derivative at the corresponding poing.
    """

    if X.ndim == 1:
        return approx_fprime(X, surface, epsilon = 1e-7).T
    else:
        return np.array([approx_fprime(x, surface, epsilon = 1e-7).T for x in X])

def normal(surface, X):
    """Gets function and point and returns the normal vector to the
    graph of surface at point(s) X. The returned normal vector is unitary
    and positively oriented.

    Parameters
    ----------
    surface: function
        Function surface: R^2 ---> R^3 which is assumed to be differentiable.
    X: np.ndarray
        Point(s) at which to compute the normal vector.
    
    Returns
    -------
    np.ndarray
        If X is a single point, returns the normal vector at such point.
        If X is a list of points, returns a matrix in which each row
        is the unit, positively oriented, normal vector at the corresponding
        point.
    """
    if X.ndim == 1:
        du,dv = du_dv(surface, X)
        v = np.cross(du,dv)
        return v/np.linalg.norm(v)
    else:
        dudv = du_dv(surface, X)
        Du,Dv = dudv[:,0], dudv[:,1]
        cross = np.cross(Du,Dv)
        return cross/np.linalg.norm(cross,axis=1)[:, np.newaxis]

def duu_duv(surface, x):
    """Gets function and point and returns the second derivatives d^2/du^2, d^2/dudv
    of surface at point(s) X. 

    Parameters
    ----------
    surface: function
        Function surface: R^2 ---> R^3 which is assumed to be differentiable.
    X: np.ndarray
        Point(s) at which to compute the second derivatives.
    
    Returns
    -------
    np.ndarray
        Returns the second derivatives d^2/du^2, d^2/dudv at such point.
    """
    return approx_fprime(x, lambda x: approx_fprime(x, surface, epsilon = 1e-7).T[0]).T

def dvv(surface, x):
    """Gets function and point and returns the second derivatives d^2/dv^2
    of surface at point(s) X. 

    Parameters
    ----------
    surface: function
        Function surface: R^2 ---> R^3 which is assumed to be differentiable.
    X: np.ndarray
        Point(s) at which to compute the second derivative.
    
    Returns
    -------
    np.ndarray
        Returns the d^2/dv^2 at such point.
    """
    return approx_fprime(x, lambda x: approx_fprime(x, surface, epsilon = 1e-7).T[1]).T[1]

#!DEPRACATED
# def projection_on_vector(vector_on_projection, vector_to_project):
#     return (np.dot(vector_on_projection, vector_to_project)/(np.linalg.norm(vector_on_projection)**2))*vector_on_projection

def length_3D_curve(curve_array_like):
    """
    Calculates the total length of a polyline. The curve is represented as a sequence of points in 3D space.

    Parameters
    ----------
    curve_array_like : array-like
        An array-like structure (e.g., a list or numpy array) containing points along the curve. Each point
        should be an array or tuple of three numbers, representing the x, y, and z coordinates.

    Returns
    -------
    float
        The total length of the curve, computed as the sum of the Euclidean distances between consecutive points.

    Example
    -------
    # Example of a curve consisting of four points in 3D space
    curve = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    length = length_3D_curve(curve)
    print(length)  # Output will be the total length of the curve

    Note:
        There is no assumption on the dimensionality of the curve, so all dimensions are accepted as correct input.
    """
    _ = np.diff(curve_array_like, axis = 0)
    return np.sum(np.linalg.norm(_,axis=1))

def check_plane_side(point_to_check, normal_vector, point_in_plane):
    """
    Determines the position of one or more points relative to corresponding planes in 3D space. Each plane is 
    defined by a normal vector and a point on the plane. The function evaluates the dot product between each 
    plane's normal vector and the vector from the plane's point to the point to be checked. 

    Parameters
    ----------
    point_to_check : np.ndarray
        An array of points to be checked. This can be a single point (as a 1D array of shape (3,)) or multiple 
        points (as a 2D array where each row represents a point in 3D space).
    normal_vector : np.ndarray
        The normal vectors of the planes. Should be a single vector (as a 1D array of shape (3,)) if checking
        one point or multiple vectors (as a 2D array with each row as a vector) corresponding to each point if 
        multiple points are checked.
    point_in_plane : np.ndarray
        Points that lie on each plane. Should be a single point (as a 1D array of shape (3,)) if checking one 
        point or multiple points (as a 2D array where each row represents a point) corresponding to each 
        normal vector if multiple points are checked.

    Returns
    -------
    np.ndarray
        An array of values resulting from the dot product operation. Positive values indicate the point is in 
        the direction of the normal vector (one side of the plane), negative values indicate it is in the 
        opposite direction, and zero indicates the point lies on the plane. If multiple points are checked, 
        the output will be an array with a dot product result for each point.

    Example
    -------
    # Define multiple planes with their normal vectors and known points
    normals = np.array([[0, 0, 1], [0, 1, 0]])
    points_in_plane = np.array([[0, 0, 0], [0, 0, 5]])

    # Define multiple points to check
    points_to_check = np.array([[0, 0, 5], [0, 5, 0]])

    # Determine the position of each point relative to their corresponding planes
    positions = check_plane_side(points_to_check, normals, points_in_plane)
    print(positions)  # Output will show positions relative to each plane
    """
    if normal_vector.shape == (3,):
        return np.dot(normal_vector, (point_to_check - point_in_plane).T)
    else:
        return np.array([np.dot(a,b) for a,b in zip(normal_vector , point_to_check - point_in_plane)])

#%% rotation (tilt) function class
class BezierPatch:
    """
    A class to create and evaluate a bicubic Bézier surface patch, assuming the definition domain is 
    the unit square [[0,1], [0,1]]. This class is used to compute the surface point and derivatives 
    at given parameter values.

    Attributes
    ----------
    control_net : np.ndarray
        A 4x4 numpy array representing the control net for the Bézier patch. Each entry in the array 
        corresponds to a control point in 3D space, influencing the shape of the bicubic patch.

    Methods
    -------
    Bezier(X)
        Evaluates the Bézier surface at the specified parameter values.
    DuDvfun(X)
        Computes the partial derivatives of the Bézier surface at the specified parameter values.
    """
    def __init__(self, control_net : np.array):
        """
        Initializes the BezierPatch with the specified control net.

        Parameters
        ----------
        control_net : np.ndarray
            A 4x4 numpy array of control points defining the Bézier patch.
        """

        self.control_net = control_net

    def Bezier(self,X : np.ndarray):
        """
        Evaluates the Bézier bicubic surface for the given parameters u and v.

        Parameters
        ----------
        X : np.ndarray
            A 2xN numpy array where each column represents a pair (u, v) of parameters, with 
            u and v in the range [0, 1].
            #! Bezier(np.array([[uuuuuuu], [vvvvvvv]]))

        Returns
        -------
        np.ndarray
            A 3xN numpy array where each column is a point [u, v, z] on the Bézier surface, 
            computed using the bicubic formula and the control net.
        """
        X = np.array(X)
        if X.shape[0] != 2:
            error_msg = 'Bad formatting for Bezier input value X.shape = ' + str(X.shape) + '\t Shape of X has to be (2,N).'
            return print(error_msg.center(130, '*'))
        
        return np.array([X[0], X[1], (-1 + X[0])**3 * (-1 + X[1])**3 * self.control_net[0, 0] -
        3 * (-1 + X[0])**3 * (-1 + X[1])**2 * X[1] * self.control_net[0, 1] +
        3 * (-1 + X[0])**3 * (-1 + X[1]) * X[1]**2 * self.control_net[0, 2] -
        (-1 + X[0])**3 * X[1]**3 * self.control_net[0, 3] -
        3 * (-1 + X[0])**2 * X[0] * (-1 + X[1])**3 * self.control_net[1, 0] +
        9 * (-1 + X[0])**2 * X[0] * (-1 + X[1])**2 * X[1] * self.control_net[1, 1] -
        9 * (-1 + X[0])**2 * X[0] * (-1 + X[1]) * X[1]**2 * self.control_net[1, 2] +
        3 * (-1 + X[0])**2 * X[0] * X[1]**3 * self.control_net[1, 3] +
        3 * (-1 + X[0]) * X[0]**2 * (-1 + X[1])**3 * self.control_net[2, 0] -
        9 * (-1 + X[0]) * X[0]**2 * (-1 + X[1])**2 * X[1] * self.control_net[2, 1] +
        9 * (-1 + X[0]) * X[0]**2 * (-1 + X[1]) * X[1]**2 * self.control_net[2, 2] -
        3 * (-1 + X[0]) * X[0]**2 * X[1]**3 * self.control_net[2, 3] -
        X[0]**3 * (-1 + X[1])**3 * self.control_net[3, 0] +
        3 * X[0]**3 * (-1 + X[1])**2 * X[1] * self.control_net[3, 1] -
        3 * X[0]**3 * (-1 + X[1]) * X[1]**2 * self.control_net[3, 2] +
        X[0]**3 * X[1]**3 * self.control_net[3, 3]])
    
    def DuDvfun(self, X):
        """
        Computes the derivatives of the Bézier surface with respect to u and v at the given parameters.

        Parameters
        ----------
        X : np.ndarray
            A 2xN numpy array where each column is a pair of parameters (u, v).

        Returns
        -------
        np.ndarray
            A 2xN numpy array where each row contains the derivatives with respect to u and v respectively.
            This can be used to compute the tangent vectors and normal vector at the surface point.
        """
        u_val, v_val = X.T
        u2, u3 = u_val**2, u_val**3
        v2, v3 = v_val**2, v_val**3
        oneMinusU, oneMinusU2, oneMinusU3 = (1 - u_val), (1 - u_val)**2, (1 - u_val)**3
        oneMinusV, oneMinusV2, oneMinusV3 = (1 - v_val), (1 - v_val)**2, (1 - v_val)**3
        du = (-3 * oneMinusU2 * oneMinusV3 * self.control_net[0, 0] -
                9 * oneMinusU2 * oneMinusV2 * v_val * self.control_net[0, 1] -
                9 * oneMinusU2 * oneMinusV * v2 * self.control_net[0, 2] -
                3 * oneMinusU2 * v3 * self.control_net[0, 3] +
                3 * oneMinusU2 * oneMinusV3 * self.control_net[1, 0] -
                6 * oneMinusU * u_val * oneMinusV3 * self.control_net[1, 0] +
                9 * oneMinusU2 * oneMinusV2 * v_val * self.control_net[1, 1] -
                18 * oneMinusU * u_val * oneMinusV2 * v_val * self.control_net[1, 1] +
                9 * oneMinusU2 * oneMinusV * v2 * self.control_net[1, 2] -
                18 * oneMinusU * u_val * oneMinusV * v2 * self.control_net[1, 2] +
                3 * oneMinusU2 * v3 * self.control_net[1, 3] -
                6 * oneMinusU * u_val * v3 * self.control_net[1, 3] +
                6 * oneMinusU * u_val * oneMinusV3 * self.control_net[2, 0] -
                3 * u2 * oneMinusV3 * self.control_net[2, 0] +
                18 * oneMinusU * u_val * oneMinusV2 * v_val * self.control_net[2, 1] -
                9 * u2 * oneMinusV2 * v_val * self.control_net[2, 1] +
                18 * oneMinusU * u_val * oneMinusV * v2 * self.control_net[2, 2] -
                9 * u2 * oneMinusV * v2 * self.control_net[2, 2] +
                6 * oneMinusU * u_val * v3 * self.control_net[2, 3] -
                3 * u2 * v3 * self.control_net[2, 3] +
                3 * u2 * oneMinusV3 * self.control_net[3, 0] +
                9 * u2 * oneMinusV2 * v_val * self.control_net[3, 1] +
                9 * u2 * oneMinusV * v2 * self.control_net[3, 2] +
                3 * u2 * v3 * self.control_net[3, 3])
        dv = (-3 * oneMinusU3 * oneMinusV2 * self.control_net[0, 0] +
                3 * oneMinusU3 * oneMinusV2 * self.control_net[0, 1] -
                6 * oneMinusU3 * oneMinusV * v_val * self.control_net[0, 1] +
                6 * oneMinusU3 * oneMinusV * v_val * self.control_net[0, 2] -
                3 * oneMinusU3 * v2 * self.control_net[0, 2] +
                3 * oneMinusU3 * v2 * self.control_net[0, 3] -
                9 * oneMinusU2 * u_val * oneMinusV2 * self.control_net[1, 0] +
                9 * oneMinusU2 * u_val * oneMinusV2 * self.control_net[1, 1] -
                18 * oneMinusU2 * u_val * oneMinusV * v_val * self.control_net[1, 1] +
                18 * oneMinusU2 * u_val * oneMinusV * v_val * self.control_net[1, 2] -
                9 * oneMinusU2 * u_val * v2 * self.control_net[1, 2] +
                9 * oneMinusU2 * u_val * v2 * self.control_net[1, 3] -
                9 * oneMinusU * u2 * oneMinusV2 * self.control_net[2, 0] +
                9 * oneMinusU * u2 * oneMinusV2 * self.control_net[2, 1] -
                18 * oneMinusU * u2 * oneMinusV * v_val * self.control_net[2, 1] +
                18 * oneMinusU * u2 * oneMinusV * v_val * self.control_net[2, 2] -
                9 * oneMinusU * u2 * v2 * self.control_net[2, 2] +
                9 * oneMinusU * u2 * v2 * self.control_net[2, 3] -
                3 * u3 * oneMinusV2 * self.control_net[3, 0] +
                3 * u3 * oneMinusV2 * self.control_net[3, 1] -
                6 * u3 * oneMinusV * v_val * self.control_net[3, 1] +
                6 * u3 * oneMinusV * v_val * self.control_net[3, 2] -
                3 * u3 * v2 * self.control_net[3, 2] +
                3 * u3 * v2 * self.control_net[3, 3])
        if np.isscalar(u_val):
            return np.array([du, dv])
        else:
            # DU =  np.array([np.ones_like(u_val), np.zeros_like(u_val), du]).T
            # DV = np.array([np.zeros_like(u_val), np.ones_like(u_val), dv]).T
            return  np.stack([du, dv], axis=1)
        
    def du_dv(self, X):
        """
        Computes the differential of the Bézier surface, providing the tangent vectors at the point.

        Parameters
        ----------
        X : np.ndarray
            A 2xN numpy array of (u, v) parameters.

        Returns
        -------
        np.ndarray
            If X represents a single point, returns a 2x3 array of tangent vectors.
            If X represents multiple points, returns an Nx2x3 array of tangent vectors for each point.
        """
        du, dv = self.DuDvfun(X).T
        
        if X.ndim == 1:
            return np.array([[1, 0, du], [0, 1, dv]])
        else:
            size = X.shape[0]
            DU =  np.array([np.ones(size), np.zeros(size), du]).T
            DV = np.array([np.zeros(size), np.ones(size), dv]).T
            return  np.stack((DU, DV), axis=1)
    def du_dv_n(self, X):
        """
        Computes the differential of the Bézier surface and the unit normal vector at the point.

        Parameters
        ----------
        X : np.ndarray
            A 2xN numpy array of (u, v) parameters.

        Returns
        -------
        np.ndarray
            If X represents a single point, returns a 3x3 array including tangent vectors and the normal vector.
            If X represents multiple points, returns an Nx3x3 array including tangent vectors and the normal vectors for each point.
        """
        du, dv = self.DuDvfun(X).T
        
        if X.ndim == 1:
            n = np.array([-du, -dv, 1])
            n = n/np.linalg.norm(n)
            return np.array([[1, 0, du], [0, 1, dv], n])
        else:
            size = X.shape[0]
            DU =  np.array([np.ones(size), np.zeros(size), du]).T
            DV = np.array([np.zeros(size), np.ones(size), dv]).T
            N = np.array([-du, -dv, np.ones(size)]).T
            N = N/np.linalg.norm(N, axis = 1)[:, np.newaxis]
            return  np.stack((DU, DV, N), axis=1)

    #! I DON'T THINK I'M ACTUALLY USING THIS SO LET'S BETTER STOP IT HERE
    # def duu(self, X):
    #     u,v = X.T
    #     return (6 * (1 - u) * (1 - v)**3 * Q[0, 0] +
    #             18 * (1 - u) * (1 - v)**2 * v * Q[0, 1] +
    #             18 * (1 - u) * (1 - v) * v**2 * Q[0, 2] +
    #             6 * (1 - u) * v**3 * Q[0, 3] +
    #             (-12 * (1 - u) + 6 * u) * (1 - v)**3 * Q[1, 0] +
    #             (-36 * (1 - u) + 18 * u) * (1 - v)**2 * v * Q[1, 1] +
    #             (-36 * (1 - u) + 18 * u) * (1 - v) * v**2 * Q[1, 2] +
    #             (-12 * (1 - u) + 6 * u) * v**3 * Q[1, 3] +
    #             (6 * (1 - u) - 12 * u) * (1 - v)**3 * Q[2, 0] +
    #             (18 * (1 - u) - 36 * u) * (1 - v)**2 * v * Q[2, 1] +
    #             (18 * (1 - u) - 36 * u) * (1 - v) * v**2 * Q[2, 2] +
    #             (6 * (1 - u) - 12 * u) * v**3 * Q[2, 3] +
    #             6 * u * (1 - v)**3 * Q[3, 0] +
    #             18 * u * (1 - v)**2 * v * Q[3, 1] +
    #             18 * u * (1 - v) * v**2 * Q[3, 2] +
    #             6 * u * v**3 * Q[3, 3])

    #! probably needs fixing now but I don't really worry about it
    # def plot(self):
    #     """Gives image of the Bézier patch, control net and control polygon."""
        
    #     urange, vrange = [[0,1], [0,1]]
    #     u = np.linspace(urange[0], urange[1], 240)
    #     v = np.linspace(vrange[0], vrange[1], 240)
    #     z = self.Bezier(u,v)
    #     fig = go.Figure()                                         

    #     puntos = np.array([  [  [el_i, el_j, self.control_net[i][j]]  for j, el_j in enumerate(np.linspace(vrange[0],vrange[1],4))]
    #                              for i, el_i in enumerate(np.linspace(urange[0],urange[1],4)) ])
    #     X = puntos[:,:,0].flatten()
    #     Y = puntos[:,:,1].flatten()
    #     Z = puntos[:,:,2].flatten()
    #     # the control net
    #     fig.add_scatter3d(x = X,y = Y, z = Z.T, mode = 'markers', marker=dict(
    #                 size=8,
    #                 color = 'rgb(0,0,255)',
    #                 opacity=1
    #             ))
    #     for fila in puntos:
    #         fig.add_scatter3d(x = fila[:,0], y = fila[:,1], z = fila[:,2], mode='lines', showlegend = False, marker=dict(
    #                 size=4,
    #                 color = 'rgb(0,0,0)',
    #                 opacity=1
    #             ))
        
    #     _ = np.array([fila.T for fila in puntos])
    #     __ = _.T
    #     __ = np.array([elemento.T for elemento in __])
    #     for fila in __:
    #         fig.add_scatter3d(x = fila[:,0], y = fila[:,1], z = fila[:,2], mode='lines', showlegend = False, marker=dict(
    #                 size=4,
    #                 color = 'rgb(0,0,0)',
    #                 opacity=1
    #             ))
    #     x_range, y_range, z_range = [ [min(_), max(_)] for _ in [X,Y,Z] ]
    #     x_len, y_len, z_len = [ _[1] - _[0] for _ in [x_range, y_range, z_range]]
        
    #     fig.add_surface(x=u, y=v, z=z.T, colorscale = 'blues', opacity = 0.85)    
    #     fig.update_layout(                                           
    #                     scene = dict(
    #                         xaxis = dict(nticks=5, range = x_range,),
    #                                     yaxis = dict(nticks=5, range = y_range,),
    #                                     zaxis = dict(nticks=5, range = z_range,)
    #                                     ,aspectratio=dict(x=x_len, y=y_len, z=z_len)
    #                                     ),
    #                     width=700,
    #                     height = 700
    #                     )

    #     fig.update_traces(showscale = False, selector = dict(type='surface'))
    #     fig.update_layout(title='Rotation Function', showlegend = True,scene=dict(
    #     xaxis=dict(showgrid=True, visible=True),
    #     yaxis=dict(showgrid=True, visible=True),
    #     zaxis=dict(showgrid=True, visible=True),
    #     aspectmode='data'
    # ))
    #     return fig

#%% surface class
class Surface:
    """
    A class that encapsulates a differentiable surface defined over R^2 ---> R^3, 
    using a Bézier bicubic patch representation for practical computation. This surface is primarily 
    defined by a control net which determines its shape, and provides functionalities to compute 
    various geometric properties such as derivatives, normals, and point clouds.

    Attributes
    ----------
    function : callable
        A function that maps R^2 to R^3, representing the surface. It is expected to be differentiable.
    du_dv_n : callable
        A function that computes the derivatives of the surface with respect to the parameters, 
        and also computes the normal vector at a given point on the surface.
    normal : callable
        A function that computes and returns the normal vector of the surface at a given point.

    Parameters
    ----------
    mat_Q : np.ndarray
        A 4x4 numpy array of control points that define the Bézier patch of the surface.
    
    Methods
    -------
    normal_vector(X)
        Computes and returns the normal vector at the point X on the surface.
    point_cloud(n_points_u, n_points_v)
        Generates a point cloud for the surface using a mesh grid of points defined over the domain.
    offset_pointcloud(m, n_points_u, n_points_v)
        Generates an offset point cloud for the surface, moving each point by a scalar multiple of the normal vector.
    """

    def __init__(self, mat_Q):
        """
        Initializes the Surface object with the specified Bézier patch control net.

        Parameters
        ----------
        mat_Q : np.ndarray
            A 4x4 array representing the control net for the Bézier patch.
        """
        self.surface = BezierPatch(control_net = mat_Q)
        self.function = self.surface.Bezier
        self.du_dv_n = self.surface.du_dv_n
        self.normal = self.normal_vector
        #! phi and theta must be defined outside or even inside the envelope class since it is an attribute of such.
        # self.phi = BezierPatch(control_net = mat_phi).Bezier
        # if mat_theta is not False:
        #     self.theta = BezierPatch(control_net = mat_theta).Bezier
        # else:
        #     self.theta = False
        self.Range = [[0,1], [0,1]] # it can be done nicer, but let's keep it simple for now
    
    def normal_vector(self, X):
        """
        Computes and returns the normal vector at the specified point on the surface.

        Parameters
        ----------
        X : np.ndarray
            A 1D or 2D array of parameter coordinates at which to compute the normal vector.

        Returns
        -------
        np.ndarray
            The normal vector(s) at the specified points.
        
        Raises
        ------
        ValueError
            If the input vector X has incorrect dimensions.
        """
        if X.ndim == 1:
            normal_vector = self.du_dv_n(X)[2]
        elif X.ndim == 2:
            normal_vector = self.du_dv_n(X)[:, 2]
        else:
            raise ValueError('Input vector X has bad dimension.')
        return normal_vector
    def point_cloud(self, n_points_u = 50, n_points_v = 50):
        """
        Generates a point cloud representing the surface.

        Parameters
        ----------
        n_points_u : int, optional
            Number of points along the u-parameter, default is 50.
        n_points_v : int, optional
            Number of points along the v-parameter, default is 50.

        Returns
        -------
        tuple
            A list containing the point coordinates and a triangulation object for plotting.
        """
        u, v = np.linspace(self.Range[0][0], self.Range[0][1], n_points_u), np.linspace(self.Range[1][0], self.Range[1][1], n_points_v)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        x,y,z = self.function([u,v])
        tri = mtri.Triangulation(u, v)
        return [[x,y,z], tri]
    
    def offset_pointcloud(self, m, n_points_u = 50, n_points_v = 50):
        """
        Generates an offset point cloud, where each point on the original surface is moved 
        along the normal vector by a distance 'm' downwards.

        Parameters
        ----------
        m : float
            The distance by which each point is offset along the normal vector.
        n_points_u : int, optional
            Number of points along the u-parameter, default is 50.
        n_points_v : int, optional
            Number of points along the v-parameter, default is 50.

        Returns
        -------
        list
            A list containing the offset point coordinates and a triangulation object for plotting.
        """
        u, v = np.linspace(self.Range[0][0], self.Range[0][1], n_points_u), np.linspace(self.Range[1][0], self.Range[1][1], n_points_v)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        offset = lambda X: self.function(X).T - m * self.normal(X.T)
        x,y,z = offset(np.array([u,v])).T
        tri = mtri.Triangulation(u, v)
        return [[x,y,z], tri]


#%% envelope class
class Envelope:
    """
    Represents the envelope of a medial circle in motion over a milled surface. This class models the behavior
    of a toolpath envelope in machining processes with flat-end cutting or toroidal cutters, where the tool's kinematics
    are described by a combination of translational and rotational movements.

    Attributes
    ----------
    R : float
        Radius of the circle representing the milling tool.
    surface : Surface
        The surface object representing the material being milled.
    u_of_t : function
        Function describing the u parameter of the toolpath as a function of time t, where t ranges in [0,1].
    v_of_t : function
        Function describing the v parameter of the toolpath as a function of time t, similar to u_of_t.
    curve : function
        Represents the spatial curve of the toolpath, computed as the surface function evaluated at (u(t), v(t)).
    curve_flat : function
        Same as curve, but ensures the output is flattened to a vector form, useful for differential operations.
    phi_of_t : function
        Function describing the rotation angle phi as a function of the parametric coordinates u(t) and v(t).
    theta_of_t : function
        Function describing the tilt angle theta as a function of the parametric coordinates u(t) and v(t).

    Methods
    -------
    tangent(t)
        Computes the normalized tangent vector of the toolpath at a given time t.
    binormal(t)
        Computes the normalized binormal vector of the toolpath at a given time t.
    curvature_ratio(X, direction)
        Computes the curvature ratio of the surface at point X in the given direction.
    set_theta_function()
        Sets or adjusts the theta function based on the milling context; defaults to Meusnier's angle if not specified.
    value_at(s, t, method='point')
        Evaluates the envelope's position at given parametric values s and t using specified methods.
    value_at_for_diff(X)
        Utility function to facilitate differential calculations at parametric values X.
    point_projection(punto, initial_parameter, delta=1e-1, tolerance=1e-4)
        Projects a point onto the toolpath, adjusting for the initial guess of parametric coordinates.
    shank_at_t(t, distance=1+np.sqrt(5))
        Computes the shank position and its end position based on the toolpath's geometry at time t.
    cilindro_at_t(t, distance=1+np.sqrt(5))
        Constructs a cylindrical representation of the toolpath for visualization and analysis.
    """
    
    def __init__(self, R,surface,u_of_t, v_of_t, mat_phi, mat_theta = False ):
        """
        Initializes the Envelope object with the given attributes.

        Parameters
        ----------
        R : float
            Radius of the milling tool.
        surface : Surface
            The milled surface represented by a Surface object.
        u_of_t : function
            Parametric function for u(t), where t ranges in [0,1].
        v_of_t : function
            Parametric function for v(t), similar to u_of_t.
        mat_phi : np.ndarray
            A 4x4 control matrix for the phi angle Bézier patch.
        mat_theta : np.ndarray or bool
            A 4x4 control matrix for the theta angle Bézier patch or False if default theta is used.
        """
        self.R = R
        self.surface = surface
        self.u_of_t = u_of_t
        self.v_of_t = v_of_t
        self.phi = BezierPatch(control_net = mat_phi).Bezier
        if mat_theta is not False:
            self.theta = BezierPatch(control_net = mat_theta).Bezier
        else:
            self.theta = False
        self.curve = lambda t: np.array(self.surface.function(np.array([self.u_of_t(t), self.v_of_t(t)])))
        self.curve_flat = lambda t: np.array(self.surface.function(np.array([self.u_of_t(t), self.v_of_t(t)]))).flatten() #! used for differentiation
        self.phi_of_t = lambda t: self.phi(np.array([self.u_of_t(t), self.v_of_t(t)]))[2]
        self.theta_of_t = self.set_theta_function()
        self.length_of_contact_line = length_3D_curve(np.array([self.curve(t) for t in np.linspace(0,1,20)]))
        
    def tangent(self,t):
        """
        Computes and returns the normalized tangent vector at a specific or multiple time points along the toolpath.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s) for which to compute the tangent vector(s). Can be a single float or an array of floats.

        Returns
        -------
        np.ndarray
            Normalized tangent vector(s) at the specified time point(s).
        """

        if type(t) != np.ndarray:
            return (approx_fprime(t, self.curve_flat)/np.linalg.norm(approx_fprime(t, self.curve_flat))).flatten()
        else:
            tangents = np.array([approx_fprime(tt, self.curve_flat).flatten() for tt in t])
            return tangents/np.linalg.norm(tangents,axis=1)[:, np.newaxis]

    def binormal(self, t):
        """
        Computes and returns the normalized binormal vector at a specific or multiple time points along the toolpath.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s) for which to compute the binormal vector(s). Can be a single float or an array of floats.

        Returns
        -------
        np.ndarray
            Normalized binormal vector(s) at the specified time point(s).
        """

        if type(t) == float:
            return np.cross(self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)])))
        elif type(t) == np.ndarray and t.ndim == 1:
            #! the surface accepts 
            UV = np.array([self.u_of_t(t), self.v_of_t(t)]).T
            return np.cross(self.tangent(t), self.surface.normal(UV))
        else:
            return print('Bad input t')
            
    def curvature_ratio(self, X, direction):
        """Curvature ratio of surface at point U and direction given.
        
        Parameters
        ----------
            X : ndarray of size (2,)
                Point in the uv-parametric space at which to compute the curvature ratio.
            direction : ndarray of size (3,)
                3D vector that in which to compute the curvature ratio of surface at point X.
        Returns
        ------
            I/II : float or numpy
                Curvature ratio of surface at point U in the given direction.
        """
        if X.ndim == 1:
            du,dv = self.surface.du_dv(X)
            E,F,G = np.linalg.norm(du)**2, np.dot(du,dv), np.linalg.norm(dv)**2
            #n = normal(U)
            n = np.cross(du,dv)
            n = n/np.linalg.norm(n)
            duu, duv = duu_duv(self.surface.function, X)
            dvV = dvv(self.surface.function, X)
            L,M,N = np.array([duu, duv, dvV]).dot(n)
            x,y= get_coordinates(direction, du,dv)
            I = E*x**2 + F*y*x + G*y**2
            II = L*x**2 + M*y*x + N*y**2
            return II/I
        elif X.ndim == 2:
            dudv = self.surface.surface.du_dv(X.T)
            du, dv = dudv[:,0], dudv[:,1]
            E,F,G = np.linalg.norm(du, axis = 1)**2, np.einsum('ij,ij->i', du, dv), np.linalg.norm(dv, axis = 1)**2
            n = np.cross(du,dv)
            n /= np.linalg.norm(n,axis=1)[:, np.newaxis] # recall n here is a matrix actually
            duuduv = np.array([ approx_fprime(a, lambda x: approx_fprime(x, self.surface.function, epsilon = 1e-7).T[0]).T for a in X.T])
            duu, duv = duuduv[:,0], duuduv[:,1]
            dvV    = np.array([ approx_fprime(a, lambda x: approx_fprime(x, self.surface.function, epsilon = 1e-7).T[1]).T[1] for a in X.T])
            L, M, N = np.einsum('ij,ij->i', duu, n), np.einsum('ij,ij->i', duv, n), np.einsum('ij,ij->i', dvV, n)
            x,y = get_coordinates(direction, du, dv)
            I = E*x**2 + F*y*x + G*y**2
            II = L*x**2 + M*y*x + N*y**2
            return II/I
    
    def set_theta_function(self):
        """
        Defines or adjusts the theta function for the envelope, using the Meusnier's angle calculation if no
        specific theta function is provided.

        Returns
        -------
        function
            A function that computes theta(t) for the parametric curve based on current settings.
        """
        if self.theta == False:
            #this sets the Meusnier angle
            #[:, np.newaxis] ensures that we can multiply element wise
            def meusnier_angle(t):
                if np.isscalar(t):
                    t = np.array([t])
                    return np.arcsin(self.R * self.curvature_ratio(np.array([self.u_of_t(t), self.v_of_t(t)]),np.cos(self.phi_of_t(t))[:, np.newaxis]*self.tangent(t) + np.sin(self.phi_of_t(t))[:, np.newaxis]*self.binormal(t) ))[0]
                return np.arcsin(self.R * self.curvature_ratio(np.array([self.u_of_t(t), self.v_of_t(t)]),np.cos(self.phi_of_t(t))[:, np.newaxis]*self.tangent(t) + np.sin(self.phi_of_t(t))[:, np.newaxis]*self.binormal(t) ))

            return meusnier_angle
        else:
            return lambda t: self.theta(np.array([self.u_of_t(t), self.v_of_t(t)]))[2]
    def value_at(self, s,t, method = 'point'):
        """
        Computes the value of the envelope at given parametric values s and t using the specified computational method.

        Parameters
        ----------
        s : float or np.ndarray
            Arc-length parameter(s) along the curve, typically ranging from -π/2 to π/2.
        t : float or np.ndarray
            Time parameter(s) along the curve, typically ranging from 0 to 1.
        method : str
            Specifies the computational method to use ('point', 'arc_trace', 'fixed_arc', 'whole_envelope', or 'curve_trace').

        Returns
        -------
        np.ndarray
            The computed 3D point(s) on the envelope corresponding to the input parameters.
        """
        # Math: E(s,t) = O(t) + R(sin(s)\mathbb{d}(t) - cos(s)\mathbb{\rho}(t))
        if  method == 'point':
            # 
            # both are scalars, i.e., we want a single point of the envelope
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            cosF, sinF, cosT, sinT, cosS, sinS = np.array([[np.cos(angle), np.sin(angle)]
                                                            for angle in [fi, teta, s]]).flatten()
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]))
            T = tangent*((-1 + cosS)*cosT*sinF + cosF*sinS) 
            B = np.cross(tangent,normal)*(cosF*(cosT - cosS*cosT) + sinF*sinS) 
            N = normal*(sinT*(1-cosS))
            return self.curve(t) + self.R*(T + B + N)
        elif method == 'arc_trace':
            # 
            # only s is a scalar an t is an array, i.e., trace of an arc value
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            cosS,sinS = np.cos(s), np.sin(s)
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]).T)
            cosS, sinS = cosS*np.ones_like(cosF), sinS*np.ones_like(cosF) # so that everything has the same dimensionality
            T = tangent*((-1 + cosS)*cosT*sinF + cosF*sinS)[:, np.newaxis] # [:, np.newaxis] gives element-wise multiplication
            B = np.cross(tangent,normal)*(cosF*(cosT - cosS*cosT) + sinF*sinS)[:, np.newaxis]
            N = normal*(sinT*(1-cosS))[:, np.newaxis]
            return self.curve(t).T + self.R*(T + B + N)
        elif method == 'fixed_arc':
            # 
            # only t is a scalar an s is an array, i.e., an arc at fixed time
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            cosS,sinS = np.cos(s), np.sin(s)
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]))
            # print('nrma tangente = ',np.linalg.norm(tangent))
            # print('nrma normal = ',np.linalg.norm(normal))
            binormal  = np.cross(tangent, normal)
            D = sinF*binormal + cosF*tangent
            Rho = sinT*normal + cosT*(-sinF*tangent + cosF*binormal)
            center = self.curve(t)
            arc_points = self.R * (D*sinS[:, np.newaxis] + Rho*(1-cosS[:, np.newaxis]))
            return center + arc_points
        elif method == 'whole_envelope':
            # 
            # in case both are arrays
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            # cosF, sinF, cosT, sinT, cosS, sinS = np.array([[np.cos(angle), np.sin(angle)] for angle in [fi, teta, s]]).flatten()
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            cosS,sinS = np.cos(s), np.sin(s)
            # get the frame at each t value
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]).T)
            # print('norma tangente = ',  np.linalg.norm(tangent, axis = 1))
            # print('norma normal = ',  np.linalg.norm(normal, axis = 1))
            binormal  = np.cross(tangent, normal)
            D = sinF[:, np.newaxis]*binormal + cosF[:, np.newaxis]*tangent
            Rho = sinT[:, np.newaxis]*normal + cosT[:, np.newaxis]*(-sinF[:, np.newaxis]*tangent + cosF[:, np.newaxis]*binormal)
            center = self.curve(t).T 
            sinS, cosS = sinS.reshape(1, -1, 1), cosS.reshape(1, -1, 1)
            D, Rho = D.reshape(D.shape[0], 1, D.shape[1]), Rho.reshape(Rho.shape[0], 1, Rho.shape[1])
            RhocosS = Rho*(1-cosS)
            # Performing element-wise multiplication
            DsinS = D * sinS
            arc_points = self.R * (DsinS + RhocosS)
            return center[:, np.newaxis, :] + arc_points
        elif method == 'curve_trace':
            # both s and t need to be arrays of the same length. We want to return E(s[i], t[i])
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            # cosF, sinF, cosT, sinT, cosS, sinS = np.array([[np.cos(angle), np.sin(angle)] for angle in [fi, teta, s]]).flatten()
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            cosS,sinS = np.cos(s), np.sin(s)

            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]).T)
            binormal  = np.cross(tangent, normal)
            D = sinF[:, np.newaxis]*binormal + cosF[:, np.newaxis]*tangent
            Rho = sinT[:, np.newaxis]*normal + cosT[:, np.newaxis]*(-sinF[:, np.newaxis]*tangent + cosF[:, np.newaxis]*binormal)
            center = self.curve(t).T 
            # sinS, cosS = sinS.reshape(1, -1, 1), cosS.reshape(1, -1, 1)
            # D, Rho = D.reshape(D.shape[0], 1, D.shape[1]), Rho.reshape(Rho.shape[0], 1, Rho.shape[1])
            RhocosS = Rho*(1-cosS[:, np.newaxis])
            # Performing element-wise multiplication
            DsinS = D * sinS[:, np.newaxis]
            arc_points = self.R * (DsinS + RhocosS)
            return center + arc_points
        else:
            warning = 'Bad method string value'
            return print(warning.center)

    def value_at_for_diff(self,X):
        """
        Facilitates differential operations by computing the envelope value at parametric coordinates provided in an array.

        Parameters
        ----------
        X : np.ndarray
            A two-element array containing the s and t parameters.

        Returns
        -------
        np.ndarray
            The 3D point on the envelope corresponding to the given parameters.
        """
        # Wrapper around value_at to support operations like gradient computation.
        s,t = X
        return self.value_at(s,t)
    
    #! DEPRACATED
    def point_projection(self, punto, initial_parameter, delta = 1e-1, tolerancia = 1e-4):
        """
        Projects a point onto the toolpath using iterative adjustment to minimize the distance to the toolpath.

        Parameters
        ----------
        punto : np.ndarray
            The 3D point to project onto the toolpath.
        initial_parameter : np.ndarray
            Initial guess for the parametric coordinates on the toolpath.
        delta : float
            Step size for iterative adjustment.
        tolerance : float
            Tolerance for convergence of the projection.

        Returns
        -------
        list
            A list containing the projected point on the toolpath and the parametric coordinates of this point.
        """
        p = punto
        U = initial_parameter
        q = self.value_at_for_diff(U)
        du,dv = du_dv(self.value_at_for_diff, U)
        n = np.cross(du,dv)
        angle = np.arccos(np.dot(n,p-q)/(np.linalg.norm(p-q)*np.linalg.norm(n)))
        while angle % (0.5*np.pi) > tolerancia:
            plane = Plane(point=q, normal=n)
            p0 = plane.project_point(p)
            x,y,z = np.linalg.solve(np.array([du,dv,n]).T, p0-q) #z will be zero or virtually zero
            U = U + delta*np.array([x,y])
            q = self.value_at_for_diff(U)
            du,dv = du_dv(self.value_at_for_diff, U)
            n = np.cross(du,dv)
            angle = np.arccos(np.dot(n,p-q)/(np.linalg.norm(p-q)*np.linalg.norm(n)))
        return [q, U]

    def shank_at_t(self, t, distance = 1+np.sqrt(5) ):
        """
        Computes the position of the milling shank at a given time t, based on the toolpath and specified distance.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s) for which to compute the shank positions.
        distance : float
            Length of the shank.

        Returns
        -------
        np.ndarray
            Positions of the shank and its endpoint at the given time t.
        """
        if np.isscalar(t):
            contact_point = self.curve(t)

            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]).T)
            
            d_orto = -sinF*tangent + cosF*np.cross(tangent,normal)
            center = contact_point + self.R*(cosT*d_orto + sinT*normal) #R*d_orto_prima
            extreme = center + self.R*distance*(cosT*normal - sinT*d_orto)# R*distance*N_prima

            return np.array([center, extreme])
        else:
            contact_point = self.curve(t).T
            fi, teta = self.phi_of_t(t), self.theta_of_t(t)
            cosF,sinF = np.cos(fi), np.sin(fi)
            cosT,sinT = np.cos(teta), np.sin(teta)
            tangent, normal = self.tangent(t), self.surface.normal(np.array([self.u_of_t(t), self.v_of_t(t)]).T)

            d_orto = -sinF[:, np.newaxis]*tangent + cosF[:, np.newaxis]*np.cross(tangent,normal)
            center = contact_point + self.R*(cosT[:, np.newaxis]*d_orto + sinT[:, np.newaxis]*normal) #R*d_orto_prima
            extreme = center + self.R*distance*(cosT[:, np.newaxis]*normal - sinT[:, np.newaxis]*d_orto)# R*distance*N_prima

            return np.array([center, extreme])


    def cilindro_at_t(self, t, distance = 1+np.sqrt(5)):
        """
        Constructs a cylindrical representation of the shank at time t.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s) for which to compute the cylindrical representation.
        distance : float
            Length of the cylinder

        Returns
        -------
        np.ndarray
            A cylindrical representation of the shank at given time and distance(length).
        """
        shank = self.shank_at_t(t, distance)
        first_circle = np.array([self.value_at(s,t) for s in np.linspace(0,2*Pi,50, endpoint=True)])
        translation_vector = shank[1]-shank[0]
        cylinder = np.array([ i*translation_vector + first_circle for i in np.linspace(0,1,50) ])
        return cylinder

class G_function:
    """
    A class designed for handling Bézier surfaces and conducting detailed analysis on their level sets,
    particularly by computing intersections and tracking contours along specified z-levels within a unit square domain.
    The class utilizes control points to define the Bézier surface, facilitating operations such as gradient
    computation, root finding along the domain borders, and efficient traversal of level sets to map out contours.

    Attributes
    ----------
    mat : np.ndarray
        A matrix representing control points for a Bézier patch. This matrix dictates the shape
        of the surface over which operations like intersection finding and contour tracing are performed.
    function : callable
        A function that computes the values of the Bézier surface for given parametric coordinates. This surface
        function is central to all evaluations and calculations within the class.
    func_grad : callable
        A derivative function that assists in gradient calculations, crucial for methods that involve optimization
        or root finding, such as tracing level sets or finding normal vectors at specific surface points.
    Range : list of list
        The definition domain for the function, specified as a list of two lists, each representing the minimum
        and maximum bounds for the parameters. Typically set to [[0, 1], [0, 1]], covering the full standard
        parametric range of the Bézier surface.

    Methods
    -------
    func_for_grad(X)
        Evaluates the Bézier function, primarily used for gradient calculations within optimization routines.
    border_u0_root_at_level(z)
        Finds intersections of the surface with the plane at z along the u=0 border of the domain.
    border_u1_root_at_level(z)
        Finds intersections of the surface with the plane at z along the u=1 border of the domain.
    border_v0_root_at_level(z)
        Finds intersections of the surface with the plane at z along the v=0 border of the domain.
    border_v1_root_at_level(z)
        Finds intersections of the surface with the plane at z along the v=1 border of the domain.
    all_border_roots(z)
        Aggregates all roots found on the borders for a given z value and filters out those not lying within the domain.
    get_next_initial_condition_and_w(r0, delta, w_last=False)
        Computes the next point and direction for contour tracing starting from a given point using a specified step size.
    is_approximation_in_range(x1)
        Checks if a given point is within the valid parametric domain of the surface.
    get_next_root(z, x1, w)
        Finds the next point on a contour at a given z level starting from an initial point and moving in a specified direction.
    level_sets(z, delta=0.01)
        Traces the complete level curve for a specified z level within the unit square.
    level_sets_multiple_z(list_of_z, delta=0.01)
        Traces level curves for multiple z values and aggregates the results.
    """

    def __init__(self, mat):
        self.mat = mat
        self.function = BezierPatch(self.mat).Bezier
        self.func_grad = self.func_for_grad
        self.Range = [[0,1], [0,1]]

    def func_for_grad(self,X):
        """
        Wrapper function for the Bézier surface to be used in gradient computations.

        Parameters
        ----------
        X : np.ndarray
            The parameter values at which to evaluate the Bézier surface.

        Returns
        -------
        np.ndarray
            The evaluated surface point at the given parameter X.
        """
        return self.function(X)

    # the four following functions can be sinthetised into one single function
    # but it is left like this for readability
    def border_u0_root_at_level(self, z):
        """
        Finds the roots (intersections) of the Bézier surface at the specified level z along the border where u=0.

        Parameters
        ----------
        z : float
            The constant z-value at which to find intersections along the u=0 border.

        Returns
        -------
        np.ndarray
            An array of points on the u=0 border where the surface intersects the plane at level z.
        """
        X = get_coefficients_of_cubic_at_z(self.mat[0], z)
        roots = roots_of_cubic_equation(X)
        return np.array([np.zeros_like(roots), roots]).T

    def border_u1_root_at_level(self, z):
        '''gets intersections for u=1 at level z'''
        X = get_coefficients_of_cubic_at_z(self.mat[-1], z)
        roots = roots_of_cubic_equation(X)
        return np.array([np.ones_like(roots), roots]).T

    def border_v0_root_at_level(self, z):
        '''gets intersections for v=0 at level z'''
        X = get_coefficients_of_cubic_at_z(self.mat[:, 0], z)
        roots = roots_of_cubic_equation(X)
        return np.array([roots, np.zeros_like(roots)]).T

    def border_v1_root_at_level(self, z):
        '''gets intersections for v=1 at level z'''
        X = get_coefficients_of_cubic_at_z(self.mat[:, -1], z)
        roots = roots_of_cubic_equation(X)
        return np.array([roots, np.ones_like(roots)]).T
    
    def all_border_roots(self, z):
        """
        Aggregates all roots found along all borders at the specified z level and filters out those not within the domain.

        Parameters
        ----------
        z : float
            The constant z-value at which to aggregate intersections from all borders.

        Returns
        -------
        np.ndarray
            An array of points where the surface intersects the borders at level z, filtered to ensure they are within the domain.
        """
        # get the roots in the boundary
        roots_u0 = self.border_u0_root_at_level(z)
        roots_u1 = self.border_u1_root_at_level(z)
        roots_v0 = self.border_v0_root_at_level(z)
        roots_v1 = self.border_v1_root_at_level(z)
        #stack them together
        all_roots = np.vstack([roots_u0,roots_u1,roots_v0,roots_v1])
        # there might be some roots outside of the domain of definition of G
        # filter those out
        all_roots = all_roots[(all_roots >= 0).all(axis=1) & (all_roots <= 1).all(axis=1)]
        return all_roots

    def get_next_initial_condition_and_w(self, r0, delta, w_last = False):
        """
        Calculates the next position and direction for tracing the level curve, starting from the initial point r0.

        Parameters
        ----------
        r0 : np.ndarray
            Current position on the level set from which to calculate the next point.
        delta : float
            Step size to move in the direction perpendicular to the gradient.
        w_last : np.ndarray or False
            The previous direction of movement; used to ensure continuity in the direction of travel.

        Returns
        -------
        tuple
            A tuple containing the next point (x1) and the new direction vector (w).
        """
        grad = approx_fprime(xk=r0, f = lambda X: self.func_grad(X)[2])
        grad = grad/np.linalg.norm(grad) # normalized gradient at r0
        w = [-1,1]*np.flip(grad) # get the perpendicular
        # the gradient gives the direction of maximum descendt
        # it's perpendicualr gives the direction of constant value
        x1 = r0 + delta*w # move in w direction
        if w_last is not False:
            # execute if w_last direction has been provided
            if np.dot(w_last, w) < 0: # if w is in the opposite direction of w_last, change it
                w = -w
                x1 = r0 + delta*w
        else:
            # execute only in the first iteration
            if not self.is_approximation_in_range(x1): # check if it is inside
                # if it is not, change direction of w and recompute x1
                w = -w
                w_last = w
                x1 = r0 + delta*w

        return (x1, w)

    def is_approximation_in_range(self, x1):
        """
        Checks whether a given point is within the defined parametric domain of the function.

        Parameters
        ----------
        x1 : np.ndarray
            The point to check.

        Returns
        -------
        bool
            True if the point is within the range, False otherwise.
        """
        # checks if a value lies inside the boundary
        return self.Range[0][0]<=x1[0]<=self.Range[0][1] and self.Range[1][0]<=x1[1]<=self.Range[1][1]

    def get_next_root(self, z, x1, w):
        """
        Computes the next root on the level set specified by z, using a line search strategy along vector w.

        Parameters
        ----------
        z : float
            The z-level at which to find the next root.
        x1 : np.ndarray
            The starting point for the line search.
        w : np.ndarray
            The direction vector along which to search for the next root.

        Returns
        -------
        np.ndarray
            The coordinates of the next root found on the level set.
        """
        L = lambda X: np.dot(w,np.array(X) - x1) # equation of the plane/line with normal direction w that passes through x1
        GZ = lambda X: self.function(X)[2]-z # level curve at level z (equation that defines it)
        sistema = lambda X: np.array([GZ(X), L(X)]) # define system of equations that have to be solved
        val = fsolve(func=sistema, x0 = x1) # next root
        return val
    
    def level_sets(self, z, delta = 0.01):
        """
        Traces complete level curves for a specified z-level within the domain.

        Parameters
        ----------
        z : float
            The z-level at which to trace the level set.
        delta : float
            The step size used for tracing the level curve.

        Returns
        -------
        np.ndarray
            An array containing the traced level curves as sequences of points.
        """
        roots_list = self.all_border_roots(z)
        dict_of_roots = {str(root):'not used' for root in roots_list}
        curves = []
        for r0 in roots_list:
            if dict_of_roots[str(r0)] == 'not used':
                points_on_curve = []
                points_on_curve.append(r0) # get the first root in 
                x1, w = self.get_next_initial_condition_and_w(r0, delta) # get the direction and the next approximation
                val = self.get_next_root(z, x1, w) 
                points_on_curve.append(val)

                while self.is_approximation_in_range(x1):
                    r00 = val
                    x1, w = self.get_next_initial_condition_and_w(r0 = r00, delta = delta, w_last = w) # get the direction and the next approximation
                    val = self.get_next_root(z, x1, w)
                    points_on_curve.append(val)

                if not self.is_approximation_in_range(points_on_curve[-1]):
                    points_on_curve = points_on_curve[:-1] # if we have exit the square, remove the last

                # get closest root to the last value of the curve inside the region
                index_of_closest_root = np.argmin(np.linalg.norm(roots_list - np.array(points_on_curve)[-1], axis=1))
                points_on_curve.append(roots_list[index_of_closest_root]) # append that root to the list
                points_on_curve = np.array(points_on_curve) # transform into numpy array
                # update the used status in the dictionary
                dict_of_roots[str(r0)] = 'used'
                dict_of_roots[str(roots_list[index_of_closest_root])] = 'used'
                # append to the curve
                if (len(points_on_curve) >2) or ((len(points_on_curve) == 2) and (not np.array_equal(points_on_curve[0], points_on_curve[1]))):
                    curves.append(points_on_curve)
                else:
                    pass

        return np.array(curves, dtype=object)

    def level_sets_multiple_z(self, list_of_z, delta = 0.01):
        """
        Computes level sets for multiple values of z and aggregates them into a single array.

        Parameters
        ----------
        list_of_z : list
            A list of z-levels for which to compute the level sets.
        delta : float
            The step size used for tracing the level sets.

        Returns
        -------
        np.array
            An array containing all the level curves traced for the given list of z values.
        """
        list_to_return = []

        for z in list_of_z:
            level_set = self.level_sets(z, delta = delta)

            if level_set.shape[0] == 1:
                level_set = level_set[0]
                list_to_return.append(level_set)
            else:
                for curva in level_set:
                    list_to_return.append(curva) 

        return np.array(list_to_return, dtype=object)

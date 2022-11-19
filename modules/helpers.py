import math
import numpy as np

import matplotlib.pyplot as plt

from scipy.special import comb


class MapVisualization:
    def __init__(self, args, carla_client):
        self.carla_client = carla_client
        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()
        self.fig, self.ax = plt.subplots()

    def destroy(self):
        self.carla_client = None
        self.world = None
        self.map = None

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def draw_line(self, points: list):
        x = []
        y = []
        for p in points:
            x.append(p.x)
            y.append(p.y)
        self.ax.plot(x, y, color='darkslategrey', markersize=2)
        return True

    def draw_spawn_points(self):
        spawn_points = self.map.get_spawn_points()
        for i in range(len(spawn_points)):
            p = spawn_points[i]
            x = p.location.x
            y = p.location.y
            self.ax.text(
                x,
                y,
                str(i),
                fontsize=6,
                color='darkorange',
                va='center',
                ha='center',
                weight='bold',
            )

    def draw_all(self):
        self.draw_roads()
        self.draw_spawn_points()
        self.destroy()

    def draw_roads(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

        for waypoints in set_waypoints:
            # waypoint = waypoints[0]
            road_left_side = [
                self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints
            ]
            road_right_side = [
                self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints
            ]
            # road_points = road_left_side + [x for x in reversed(road_right_side)]
            # self.add_line_strip_marker(points=road_points)

            if len(road_left_side) > 2:
                self.draw_line(points=road_left_side)
            if len(road_right_side) > 2:
                self.draw_line(points=road_right_side)


def find_in_between_angle(v, w):
    theta = math.atan2(np.linalg.det([v[0:2], w[0:2]]), np.dot(v[0:2], w[0:2]))
    return theta


def rotate(points, angle):
    R = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = R.dot(points)
    rotated[0] *= -1
    return rotated


def calc_ego_frame_projection(x, moving_direction):
    theta = find_in_between_angle(moving_direction, np.array([0.0, 1.0, 0.0]))
    projected = rotate(x[0:2].T, theta)
    return projected


def project_to_ego_frame(waypoints, data):
    # Direction vector
    moving_direction = np.array(data['moving_direction'])

    # Origin shift
    shifted_waypoints = np.array(waypoints) - np.array(data['location'])

    # Projected points
    projected_waypoints = np.zeros((len(shifted_waypoints), 2))
    for i, waypoint in enumerate(shifted_waypoints):
        projected_waypoints[i, :] = calc_ego_frame_projection(
            waypoint, moving_direction
        )
    return projected_waypoints


def calc_world_projection(ego_frame_coord, moving_direction, ego_location):
    #  Find the rotation angle such the movement direction is always positive
    theta = find_in_between_angle(moving_direction, np.array([0.0, 1.0, 0.0]))

    # Rotate the pointd back
    ego_frame_coord[0] *= -1
    re_projected = rotate(ego_frame_coord.T, -theta)
    re_projected += ego_location
    return re_projected


def project_to_world_frame(ego_frame_waypoints, data):
    ego_location = np.array(data['location'])
    v_vec = np.array(data['moving_direction'])

    # Projected points
    projected_waypoints = np.zeros((len(ego_frame_waypoints), 2))
    for i, waypoint in enumerate(ego_frame_waypoints):
        projected_waypoints[i, :] = calc_world_projection(
            waypoint, v_vec, ego_location[0:2]
        )
    return projected_waypoints


def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(
            f'There must be at least {degree + 1} points to '
            f'determine the parameters of a degree {degree} curve. '
            f'Got only {len(X)} points.'
        )

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))

    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final) - 1] = [X[len(X) - 1], Y[len(Y) - 1]]
    return np.array(final)

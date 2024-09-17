import numpy as np


class RadialDistortionRemover:

    def __init__(self, fx, fy, cx, cy, k1, k2, t1, t2, k3):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.t1 = t1
        self.t2 = t2
        self.k3 = k3

    def undistort_events(self, events):

        points = events[:, 0:2]

        x_norm = (points[:, 0] - self.cx) / self.fx
        y_norm = (points[:, 1] - self.cy) / self.fy

        # Initial guess for undistorted normalized coordinates
        x_u = x_norm
        y_u = y_norm

        # Iteratively refine undistorted normalized coordinates
        for _ in range(2):  # Adjust the number of iterations as needed
            # Distortion radius
            r2 = x_u ** 2 + y_u ** 2
            r4 = r2 ** 2
            r6 = r2 ** 3

            # Radial distortion
            radial_distortion = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

            # Tangential distortion
            dx = 2 * self.t1 * x_u * y_u + self.t2 * (r2 + 2 * x_u ** 2)
            dy = self.t1 * (r2 + 2 * y_u ** 2) + 2 * self.t2 * x_u * y_u

            # Corrected normalized coordinates
            x_corr_norm = (x_norm - dx) / radial_distortion
            y_corr_norm = (y_norm - dy) / radial_distortion

            # Update undistorted normalized coordinates for next iteration
            x_u = x_corr_norm
            y_u = y_corr_norm

        # Denormalize undistorted coordinates
        undistorted_points = np.column_stack((
            (x_corr_norm * self.fx) + self.cx,
            (y_corr_norm * self.fy) + self.cy,
            events[:, 2],
            events[:, 3]
        ))

        return undistorted_points
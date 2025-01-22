import torch
import matplotlib.pyplot as plt


class semi_sphere_rays():
    def __init__(self):
        self.center = torch.tensor([0, 0, 1.0])

    def create_semi_sphere_spiral(self, N, create_interpolation=True):
        """
        Generate N evenly distributed points on a unit hemisphere using a spiral sampling method.
        """
        indices = torch.arange(0, N)
        indices = indices.float() / 2.0   #semi-sphere

        phi = torch.pi * (5.0 ** 0.5 -1.0)
        lat_grid = torch.acos(1 - 2 * indices / (N - 1))
        lon_grid = indices * phi

        x = torch.sin(lat_grid) * torch.cos(lon_grid)
        y = torch.sin(lat_grid) * torch.sin(lon_grid)
        z = torch.cos(lat_grid)

        # Combine into a single tensor
        vectors = torch.stack((x, y, z), dim=-1)

        self.original_semi_sphere = vectors

        if create_interpolation:
            # flat the semi_sphere, get a square, and use nearest interpolation
            self.resolution_interpolation = (int(N ** 0.5), int(N ** 0.5))
            # Calculate the arc distance from the center (0, 0, 1) for each point
            arc_distances = lat_grid
            radius_max = arc_distances.max()

            # Project points: Use arc distance as radius and atan2(y, x) as angle
            angles = lon_grid
            projected_x = arc_distances * torch.cos(angles)
            projected_y = arc_distances * torch.sin(angles)

            # Grid for interpolation. xy in image direction, so y is from max to min
            grid_x = torch.linspace(projected_x.min(), projected_x.max(), self.resolution_interpolation[0])
            grid_y = torch.linspace(projected_y.max(), projected_y.min(), self.resolution_interpolation[1])
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y)

            # Interpolate using torch (nearest neighbor interpolation for simplicity)
            # Flatten grid coordinates for indexing
            flat_grid_x = grid_x.flatten()
            flat_grid_y = grid_y.flatten()

            # Find nearest neighbors
            dist = torch.sqrt(
                (projected_x[:, None] - flat_grid_x[None, :]) ** 2 + (projected_y[:, None] - flat_grid_y[None, :]) ** 2)
            self.square_interpolation_index = dist.argmin(dim=0)
            self.square_mask = torch.sqrt(flat_grid_x ** 2 + flat_grid_y ** 2) < radius_max

    def rotation_matrix_from_z_to_vector(self, v):
        # Normalize the input vector
        v = v / v.norm(dim=-1, keepdim=True)

        # Z-axis unit vector
        z = torch.tensor([0.0, 0.0, 1.0], device=v.device)

        # Compute the axis of rotation (cross product with z-axis)
        axis = torch.cross(z.expand_as(v), v, dim=-1)
        axis_norm = axis.norm(dim=-1, keepdim=True)
        axis = torch.where(axis_norm > 0, axis / axis_norm, torch.tensor([1.0, 0.0, 0.0], device=v.device).expand_as(v))

        # Compute the skew-symmetric cross-product matrix of the axis
        skew = torch.zeros(*v.shape[:-1], 3, 3, device=v.device)
        skew[..., 0, 1] = -axis[..., 2]
        skew[..., 1, 0] = axis[..., 2]
        skew[..., 0, 2] = axis[..., 1]
        skew[..., 2, 0] = -axis[..., 1]
        skew[..., 1, 2] = -axis[..., 0]
        skew[..., 2, 1] = axis[..., 0]

        # Compute the rotation matrix using Rodrigues' formula
        angle = torch.acos(torch.clamp(v[..., 2], -1.0, 1.0))
        identity = torch.eye(3, device=v.device).expand_as(skew)
        R = identity + skew * torch.sin(angle).unsqueeze(-1).unsqueeze(-1) + \
            torch.matmul(skew, skew) * (1 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1)

        # Handle the special cases
        # Case 1: v is parallel to z-axis (no rotation needed)
        R[v[..., 2] >= 1] = identity[v[..., 2] >= 1]

        # Case 2: v is antiparallel to z-axis (180 degree rotation around x-axis)
        R_antiparallel = -identity.clone()
        R_antiparallel[..., 0, 0] = 1
        R[v[..., 2] <= -1] = R_antiparallel[v[..., 2] <= -1]

        return R

    def rotate_vectors(self, vectors, normal):
        """
        Rotate a set of vectors so that the z-axis aligns with the given normal.

        :param vectors: Tensor of shape (N, M, 3) containing unit vectors.
        :param normal: A target normal vector.
        :return: Rotated vectors.
        """
        # # Normalize the target normal vector and ensure it's a floating point tensor
        # normal = normal.float() / torch.linalg.norm(normal.float())
        #
        # # Find the rotation axis (cross product of z-axis and normal) and angle (arccosine of the dot product of z-axis and normal)
        # z_axis = torch.tensor([0, 0, 1.0], dtype=torch.float)
        # rotation_axis = torch.cross(z_axis, normal)
        # rotation_angle = torch.arccos(torch.dot(z_axis, normal))
        #
        # # Check if the rotation axis is zero (i.e., the normal is in the direction of the z-axis or opposite to it)
        # if torch.linalg.norm(rotation_axis) == 0:
        #     if torch.dot(z_axis, normal) > 0:
        #         # No rotation needed, normal is already in the z-axis direction
        #         return vectors
        #     else:
        #         # 180 degree rotation around any axis perpendicular to z-axis
        #         rotation_axis = torch.tensor([1.0, 0, 0])  # Using x-axis for rotation
        #         rotation_angle = torch.tensor(torch.pi)
        #
        # # Normalize the rotation axis
        # rotation_axis = rotation_axis / torch.linalg.norm(rotation_axis)
        #
        # # Using Rodrigues' rotation formula to create the rotation matrix
        # K = torch.tensor([[0, -rotation_axis[2], rotation_axis[1]],
        #                   [rotation_axis[2], 0, -rotation_axis[0]],
        #                   [-rotation_axis[1], rotation_axis[0], 0]])
        # R = torch.eye(3) + torch.sin(rotation_angle) * K + (1 - torch.cos(rotation_angle)) * (K @ K)

        R = self.rotation_matrix_from_z_to_vector(normal)

        # Rotate all vectors
        rotated_vectors = torch.einsum('ij,kj->ki', R, vectors)

        return rotated_vectors

    def plot_interpolate_semi_sphere(self, output_semi_sphere):
        """
        Perform interpolation on the semi-sphere points using torch and plot the resulting color map.

        :param semi_sphere: Tensor of shape (N, M, 3) containing unit vectors of a semi-sphere.
        """
        nearest = self.square_interpolation_index

        interpolated_output = output_semi_sphere[nearest].view([self.resolution_interpolation[0], self.resolution_interpolation[1], -1])
        interpolated_output = interpolated_output.numpy().transpose([1, 0, 2])

        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(interpolated_output)
        plt.grid(True)
        plt.show()

    # Plotting the 3D scatter plot of the semi-sphere in XYZ space
    def plot_3d_scatter_semi_sphere(self, semi_sphere, color, elev=None, azim=None):
        """
        Plot a 3D scatter plot of the semi-sphere in XYZ space.
        """
        # Extract the x, y, z components
        x = semi_sphere[..., 0].numpy()
        y = semi_sphere[..., 1].numpy()
        z = semi_sphere[..., 2].numpy()
        c = color.numpy().reshape([-1,3])

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=c, cmap='viridis')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Scatter Plot of Semi-Sphere')
        ax.view_init(elev=elev, azim=azim)
        plt.show()


if __name__ == "__main__":
    # Create a semi-sphere of unit vectors
    semi_sphere = semi_sphere_rays()
    semi_sphere.create_semi_sphere_spiral(1000)
    color = (semi_sphere.original_semi_sphere + 1.0) / 2.0
    semi_sphere.plot_3d_scatter_semi_sphere(semi_sphere.original_semi_sphere, color, elev=90, azim=-90)
    new_normal = torch.tensor([0.0, 0.0, 1.0])
    rotated_vectors = semi_sphere.rotate_vectors(semi_sphere.original_semi_sphere, new_normal)
    semi_sphere.plot_3d_scatter_semi_sphere(rotated_vectors, color, elev=None, azim=None)
    semi_sphere.plot_interpolate_semi_sphere(color)



    pass
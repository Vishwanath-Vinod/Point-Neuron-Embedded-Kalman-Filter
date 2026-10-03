import numpy as np

class Grid:
    def __init__(self, positions: np.array):
        self.positions = positions

    def __iter__(self):
        flattened_positions = self.positions.reshape(-1, self.positions.shape[-1])

        for position in flattened_positions:
            yield position

    def __getitem__(self, key):
        position = self.positions[key]
        return position
    
    def __setitem__(self, key, value):
        self.positions[key] = value

    def asarray(self):
        return self.positions.reshape(-1, self.positions.shape[-1])

    def __len__(self):
        return np.prod(self.positions.shape[:-1])
    
class UniformCartesianGrid(Grid):
    """Create an uniform cartesian grid in 2D or 3D for a cuboid shaped room
    """

    def __init__(self, bounds: list[float], n_grid_cells_per_dim: int): 
        """
        bounds: List of floats, each float representing the length of the cuboid shape in that dimension
        n_grid_cells_per_dim: number of grid cells per dimension
        """

        self.bounds = np.array(bounds)
        self.n_grid_cells_per_dim = n_grid_cells_per_dim

        positions = self._create_grid()
        # Save number of points
        self.n_points = positions.shape[0]
        super().__init__(positions)

    def _create_grid(self):
        dim = len(self.bounds)
        grid_shape = [self.n_grid_cells_per_dim] * dim + [dim]
        positions = np.zeros(grid_shape)

        # Cube of side 0.4 centered in the room
        cube_side = 1.0
        room_center = self.bounds / 2
        cube_min = room_center - cube_side / 2
        cube_max = room_center + cube_side / 2

        cell_resolution = cube_side / self.n_grid_cells_per_dim
        start_position = cube_min + cell_resolution / 2

        xrange = np.linspace(start_position[0], cube_max[0] - cell_resolution / 2, self.n_grid_cells_per_dim)
        yrange = np.linspace(start_position[1], cube_max[1] - cell_resolution / 2, self.n_grid_cells_per_dim)

        if dim == 2:
            for i, x in enumerate(xrange):
                for j, y in enumerate(yrange):
                    positions[i, j] = np.array([x, y])
        elif dim == 3:
            zrange = np.linspace(start_position[2], cube_max[2] - cell_resolution / 2, self.n_grid_cells_per_dim)
            for i, x in enumerate(xrange):
                for j, y in enumerate(yrange):
                    for k, z in enumerate(zrange):
                        positions[i, j, k] = np.array([x, y, z])
        return positions

    

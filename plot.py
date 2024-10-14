from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

class Dirichlet_Neumann():
    def __init__(self, room, mesh_size, comm, rank, num_iterations=10, omega=0.8):
        self.mesh_size = mesh_size
        self.room = room
        self.comm = comm
        self.rank = rank
        self.num_iterations = num_iterations
        self.omega = omega
        self.temperature_values = []
        
        # Loop through room dimensions and boundary conditions to build linear system for each room
        for (room_name, room_info) in self.room.items():
            room_dim, bc, adjacents = room_info
            room_width, room_height = room_dim
            _, b = build_linalg(room_width, room_height, bc, self.mesh_size)
            B = np.array(b)
            
            # Reshape temperature values into a 2D array
            Nx = int(room_width / self.mesh_size)
            Ny = int(room_height / self.mesh_size)
            B_reshaped = B.reshape(Ny, Nx)
            
            self.temperature_values.append(B_reshaped)
        print(f'Temperature values stored are: {self.temperature_values}')
            
    def solve(self):
        for num in range(self.num_iterations):
            for i, (room_name, room_info) in enumerate(self.room.items()):
                room_dim, bc, adjacents = room_info
                room_width, room_height = room_dim
                if self.rank == i:
                    room_dim, bc, adjacents = room_info
                    temperature_grid = self.temperature_values[i]
                    self.send_data(temperature_grid, adjacents)
                    u = self.receive_data(temperature_grid, adjacents)
                    print(f'The u received from iteration {i} is {u}')
                    sol = solve_linalg(room_width, room_height, bc, self.mesh_size)
                    sol_relax = self.omega * sol + (1 - self.omega) * sol
                    print(f'The sol_relax for iteration {i} is {sol_relax}')
                
        return sol_relax
    
    def send_data(self, temperature_grid, adjacents):
        for direction, info in adjacents.items():
            adjacent_room, boundary_positions, gamma_type, adjacent_rank = info
            start = boundary_positions['start']
            end = boundary_positions['end']
            #print(f'The adjacent_rank received in the send data is: {adjacent_rank}')
            
           
            if direction == 'right':
                print(f'Rank {self.rank} sending right to rank {adjacent_rank}')
                print(f'The elements sent to right with temperature_grid is: {temperature_grid[:, -1]}')
                self.comm.send(temperature_grid[start:end, -2:], dest = adjacent_rank,  tag = 100 + self.rank)
            elif direction == 'left':
                print(f'Rank {self.rank} sending left to rank {adjacent_rank}')
                print(f'The elements sent to left with temperature_grid is: {temperature_grid[:, 0]}')
                self.comm.send(temperature_grid[start:end, :1], dest = adjacent_rank, tag = 100 + self.rank)
    
    def receive_data(self, temperature_grid, adjacents):
        for direction, info in adjacents.items():
            adjacent_room, boundary_positions, gamma_type, adjacent_rank = info
            start = boundary_positions['start']
            end = boundary_positions['end']
            
            print(f'Rank {self.rank} waiting to receive from rank {adjacent_rank} on direction {direction}')
        
            try:
                sol_new = self.comm.recv(source=adjacent_rank, tag=100 + adjacent_rank)
                print(f'The elements in sol_new of recv data is: {sol_new}')
                print(f"Rank {self.rank} received data from rank {adjacent_rank}")
            except Exception as e:
                print(f"Error receiving data on rank {self.rank}: {e}")
            
            if gamma_type == 'Dirichlet':
                if direction == 'right':
                    temperature_grid[start:end, -1] = sol_new[:, 0]
                    
                elif direction == 'left':
                    temperature_grid[start:end, 0] = sol_new[:, -1]
                    
            if gamma_type == 'Neumann':
                if direction == 'right':
                    flux = (sol_new[:, -1] - sol_new[:, 0] )/ self.mesh_size
                    temperature_grid [start:end, -1] = temperature_grid[start:end, -1] + flux * self.mesh_size
                    
                elif direction == 'left':
                    flux = (sol_new[:,0] - sol_new[:, -1])/ self.mesh_size
                    temperature_grid [start:end, 0] = temperature_grid [start:end, 0] + flux * self.mesh_size
                    
        return temperature_grid
    
                    


def build_linalg(room_width, room_height, bc, mesh_size):
    A = []
    b = []

    Nx = int(room_width / mesh_size)
    Ny = int(room_height / mesh_size)
    
    for j in range(Nx):
        for k in range(Ny):
            N = Nx * Ny
            row = np.zeros(N)
            rhs = 0
            diag = -4

            row, rhs, diag = apply_bc(j-1, k, row, rhs, diag, bc, Ny, mesh_size)
            row, rhs, diag = apply_bc(j+1, k, row, rhs, diag, bc, Ny, mesh_size)
            row, rhs, diag = apply_bc(j, k-1, row, rhs, diag, bc, Ny, mesh_size)
            row, rhs, diag = apply_bc(j, k+1, row, rhs, diag, bc, Ny, mesh_size)

            row[get_index(j, k, Ny)] = diag
            A.append(row)
            b.append(rhs)
    
    print(f'The array A is: {np.array(A)}')
    print(f'The array B is: {np.array(b).reshape(Ny, Nx)}')
    return np.array(A), np.array(b)


def get_index(j, k, Ny):
    return j * Ny + k


def apply_bc(j, k, row, rhs, diag, bc, Ny, mesh_size):
    bound = bc(j, k, mesh_size)
    if bound == 'interior':
        row[get_index(j, k, Ny)] = 1
    elif bound == 'heater':
        rhs -= 40
    elif bound == 'window':
        rhs -= 5
    elif bound == 'wall':
        rhs -= 15
    return row, rhs, diag

def solve_linalg(room_width, room_height, bc, mesh_size):
    A, b = build_linalg(room_width, room_height, bc, mesh_size)
    sol = np.linalg.solve(A, b)
    Nx = int(room_width / mesh_size)
    Ny = int(room_height / mesh_size)
    sol = np.rot90(sol.reshape(Nx, Ny), k=1, axes=(0,1))
    return sol


# MPI Initialization
if __name__ == '__main__':
    dx = 1 / 20

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Boundary conditions for each room
    def bc_1(x, y, mesh_size):
        if x < 0:
            return 'heater'
        if y < 0 or y >= 1/mesh_size:
            return 'wall'
        if x >= 1/mesh_size:
            return 'gamma_1'
        return 'interior'

    def bc_2(x, y, mesh_size):
        if y < 0:
            return 'window'
        if x > 1 and y < 1/mesh_size or x < 0 and y >= 1/mesh_size:
            return 'wall'
        if x >= 1/mesh_size and y >= 1/mesh_size:
            return 'gamma_3'
        if x < 0 and y < 1/mesh_size:
            return 'gamma_1'
        if y >= 2/mesh_size:
            return 'heater'
        return 'interior'

    def bc_3(x, y, mesh_size):
        if x >= 1/mesh_size:
            return 'heater'
        if y < 0 or y >= 1/mesh_size:
            return 'wall'
        if x < 0:
            return 'gamma_3'
        return 'interior'

    # Dictionary of room dimensions and boundary conditions (using unique names for each room)
    room = {
    'Room1': [(1, 1), bc_1, {'right': ('Room2', {'start': 0, 'end': int(1/dx)}, 'Neumann', 1)}],  # Room1 with adjacent Room2 on the right
    'Room2': [(1, 2), bc_2, {'left': ('Room1', {'start': 0, 'end': int(1/dx)}, 'Dirichlet', 0), 'right': ('Room3', {'start': 0, 'end': int(1/dx)}, 'Dirichlet', 2)}],  # Room2 with Room1 on the left and Room3 on the right
    'Room3': [(1, 1), bc_3, {'left': ('Room2', {'start': 0, 'end': int(1/dx)}, 'Neumann', 1)}]  # Room3 with Room2 on the left
}

    # Instantiate the solver
    solver = Dirichlet_Neumann(room, dx, comm, rank)
    val = solver.solve()
    all_vals = comm.gather(val, root=0)

    if rank == 0:
        for i in range(len(all_vals)):
            val_i = all_vals[i]
            print(f'Shape of temperature distribution for Room {i + 1}: {val_i.shape}')
            plt.imshow(val_i, cmap='plasma')
            plt.colorbar(label='Temperature')
            plt.title(f'Temperature Distribution in Room {i + 1}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.savefig(f'Room_{i + 1}_temperature.png')
            plt.clf()  # Clear the figure 
    
    
       

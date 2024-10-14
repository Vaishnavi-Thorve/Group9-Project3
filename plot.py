from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

class Room:
    def __init__(self, dimension, boundary_condition, title):
        self.dimension = dimension
        self.boundary_condition = boundary_condition
        self.title = title
        self.adjacent_rooms = {}  # Dictionary to hold adjacent rooms
        self.temperature = np.full(self.dimension, 15)  # Initialize room temperature array with zeros
        self.boundary_temperatures = {'heater': 40, 'window': 5}  # Dictionary for boundary conditions
        
        self.apply_boundary_conditions()  # Apply boundary conditions during initialization

    def add_adjacent_room(self, direction, room, boundary_positions, gamma_type:None, rank: None):
        self.adjacent_rooms[direction] = {'room': room, 'boundary_positions': boundary_positions, 'gamma_type': gamma_type, 'rank': rank}

    def apply_boundary_conditions(self):
        for i, condition in self.boundary_condition.items():
            # Get the numerical boundary temperature, like 40 for 'heater' or 5 for 'window'
            boundary_temperature = self.boundary_temperatures.get(condition)
            
            # Apply boundary conditions to the temperature array based on the wall direction
            match i:
                case 'left':
                    self.temperature[:, 0] = boundary_temperature  # Apply to the left boundary (1st column)
                case 'right':
                    self.temperature[:, -1] = boundary_temperature  # Apply to the right boundary (last column)
                case 'top':
                    self.temperature[0, :] = boundary_temperature  # Apply to the top boundary (1st row)
                case 'bottom':
                    self.temperature[-1, :] = boundary_temperature  # Apply to the bottom boundary (last row)

class Apartment:
    def __init__(self):
        self.rooms = []

    def add_room(self, room):
        self.rooms.append(room)


class Dirichlet_Neumann:
    def __init__(self, rooms, dx, comm, rank, num_iterations=10, omega=0.8):
        self.rooms = rooms
        self.dx = dx
        self.comm = comm
        self.rank = rank
        self.num_iterations = num_iterations
        self.omega = omega
        self.temperature_values = [room.temperature for room in self.rooms]  

    def solve(self):
        for num in range (self.num_iterations):
            for i, room in enumerate(self.rooms):
                if self.rank == i:
                    temperature_grid = self.temperature_values[i]
                    print(f'Shape of the temperature_grid is: {temperature_grid.shape}')
                    print(f'\nIteration {i}: Solving for room "{room.title}" with rank {self.rank}')
                    self.send_data(room, temperature_grid)
                    u = self.receive_data(room, temperature_grid, dx)
                    print(f'The u received from iteration {i} is {u}')
                    sol = solve_linalg(u, self.dx)
                    sol_relax = self.omega * sol + (1 - self.omega) * sol
                    print(f'The sol_relax for iteration {i} is {sol_relax}')
            
        return sol_relax
            
    def send_data(self, room, temperature_grid):
        for direction, info in room.adjacent_rooms.items():
            adjacent_rank = info['rank']
            #print(f'The adjacent_rank received in the send data is: {adjacent_rank}')
            
           
            if direction == 'right':
                print(f'Rank {self.rank} sending right to rank {adjacent_rank}')
                print(f'The elements sent to right with temperature_grid is: {temperature_grid[:, -1]}')
                self.comm.send(temperature_grid[:, -1], dest = adjacent_rank,  tag = 100 + self.rank)
            if direction == 'left':
                print(f'Rank {self.rank} sending left to rank {adjacent_rank}')
                print(f'The elements sent to left with temperature_grid is: {temperature_grid[:, 0]}')
                self.comm.send(temperature_grid[:, 0], dest = adjacent_rank, tag = 100 + self.rank)
                
    def receive_data(self, room, temperature_grid, dx):
        for direction, info in room.adjacent_rooms.items():
            gamma_type = info['gamma_type']
            adjacent_rank = info['rank']
            print(f'Rank {self.rank} waiting to receive from rank {adjacent_rank} on direction {direction}')
        
            try:
                sol_new = self.comm.recv(source=adjacent_rank, tag=100 + adjacent_rank)
                print(f'The elements in sol_new of recv data is: {sol_new}')
                print(f"Rank {self.rank} received data from rank {adjacent_rank}")
            except Exception as e:
                print(f"Error receiving data on rank {self.rank}: {e}")
            
            if gamma_type == 'Dirichlet':
                if direction == 'right':
                    temperature_grid[:, -1] = sol_new
                    
                elif direction == 'left':
                    temperature_grid[:, 0] = sol_new
                    
            if gamma_type == 'Neumann':
                if direction == 'right':
                    flux = (temperature_grid[:, -2] - temperature_grid[:, -1] )/ dx
                    temperature_grid [:, -1] = flux
                    
                elif direction == 'left':
                    flux = (temperature_grid[:,1] - temperature_grid[:, 0])/ dx
                    temperature_grid [:, 0] = flux
                    
        return temperature_grid   


def build_linalg(u, n, m, dx):
    N = n * m  # Total number of elements
    coeff = 1 / dx**2
    A = np.zeros((N, N))
    
    # Create a mask for the boundary
    boundary_mask = np.zeros((n, m), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_indices = np.flatnonzero(boundary_mask)
    
    b = np.zeros(N)
    b[boundary_indices] = u.flatten()[boundary_indices]  # Flattening u to get a 1D array

    for i in range(N):
        # Diagonal value
        A[i, i] = -4
        
        # Upper diagonal
        if i + 1 < N and (i + 1) % m != 0:
            A[i, i + 1] = 1

        # Lower diagonal
        if i - 1 >= 0 and i % m != 0:
            A[i, i - 1] = 1

        if i + m < N:
            A[i, i + m] = 1

        if i - m >= 0:
            A[i, i - m] = 1

    return csr_matrix(coeff * A), b

def solve_linalg(temperature_grid, dx):
    shape = temperature_grid.shape
    print(f'Shape of the temperature_grid is: {shape}')
    n, m = shape
    
    A, b = build_linalg(temperature_grid, n, m, dx)
    sol = sp.linalg.spsolve(A, b).reshape((n, m))
    return sol

if __name__ == '__main__':
    # Create rooms with boundary conditions
    dx = 1 / 3
    Room1 = Room(dimension=[int(1/dx), int(1/dx)], boundary_condition={'left': 'heater'}, title='Room1')
    Room2 = Room(dimension=[int(2/dx), int(1/dx)], boundary_condition={'top': 'heater', 'bottom': 'window'}, title='Room2')
    Room3 = Room(dimension=[int(1/dx), int(1/dx)], boundary_condition={'right': 'heater'}, title='Room3')

    # Define adjacent rooms
    Room1.add_adjacent_room("right", Room2, boundary_positions={'start': 0, 'end': (int(1/dx) + 1)}, gamma_type = 'Neumann', rank = 1)
    Room2.add_adjacent_room("left", Room1, boundary_positions={'start': (int(1/dx)), 'end': (int(2/dx) + 1)}, gamma_type = 'Dirichlet', rank = 0)
    Room2.add_adjacent_room("right", Room3, boundary_positions={'start': 0, 'end': (int(1/dx) + 1)}, gamma_type = 'Dirichlet', rank =2)
    Room3.add_adjacent_room("left", Room2, boundary_positions={'start': 0, 'end': (int(1/dx) + 1)}, gamma_type = 'Neumann', rank = 1)

    # Create an apartment and add rooms
    apartment = Apartment()
    apartment.add_room(Room1)
    apartment.add_room(Room2)
    apartment.add_room(Room3)


    # Solve the system of equations and display results
    #solver = Dirichlet_Neumann(apartment.rooms, dx)
    #solver.solve()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    solver = Dirichlet_Neumann(apartment.rooms, dx, comm, rank)
    val = solver.solve()
    plt.imshow(val, cmap='plasma')
    plt.colorbar(label='Temperature')
    plt.savefig('temperature.png')
    MPI.Finalize
    
    


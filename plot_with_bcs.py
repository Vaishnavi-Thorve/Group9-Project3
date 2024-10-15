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
        self.temperature = np.full(self.dimension, 15)  # Initialize room temperature array with 15
        self.boundary_temperatures = {'heater': 40, 'window': 5}  # Dictionary for boundary conditions
        self.boundary_mask = np.full((self.dimension[0] + 2, self.dimension[1] + 2), 15) # Initialize room temperature array with wall temp

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
                    self.boundary_mask[:, 0] = boundary_temperature  # Apply to the left boundary (1st column)
                case 'right':
                    self.boundary_mask[:, -1] = boundary_temperature  # Apply to the right boundary (last column)
                case 'top':
                    self.boundary_mask[0, :] = boundary_temperature  # Apply to the top boundary (1st row)
                case 'bottom':
                    self.boundary_mask[-1, :] = boundary_temperature  # Apply to the bottom boundary (last row)
        print(self.boundary_mask)

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
            print(f'\nIteration: {num}')
            for i, room in enumerate(self.rooms):
                if self.rank == i:
                    temperature_grid = self.temperature_values[i]
                    print(f'\nSolving for room "{room.title}" with rank {self.rank}')
                    if num != 0:  u = self.receive_data(room, temperature_grid, self.dx)
                    else: u = temperature_grid

                    sol = solve_linalg(room.boundary_mask, u, self.dx)          # might be an issue here
                    print(f'solution is {sol}')
                    if num == 0: sol_relax = sol
                    else: sol_relax = self.omega * sol + (1 - self.omega) * self.temperature_values[i-1]

                    self.temperature_values[i] = sol_relax
                    temperature_grid = sol_relax
                    print(f'Shape of the temperature_grid is: {temperature_grid.shape}')
                    print(f'The sol_relax for iteration {i} is {sol_relax}')

                    self.send_data(room, temperature_grid)
                    print(f'The u received from iteration {i} is {u}')
        return sol_relax
            
    def send_data(self, room, temperature_grid):
        for direction, info in room.adjacent_rooms.items():
            gamma_type = info['gamma_type']
            adjacent_rank = info['rank']
            boundary_positions = info['boundary_positions']
            start = boundary_positions['start']
            end = boundary_positions['end']
            #print(f'The adjacent_rank received in the send data is: {adjacent_rank}')

            if direction == 'right':
                print(f'Rank {self.rank} sending right to rank {adjacent_rank}')
                print(f'The elements sent to right with temperature_grid is: {temperature_grid[:, -1]}')
                self.comm.send(temperature_grid[start:end, -2:], dest = adjacent_rank,  tag = 100 + self.rank)
            if direction == 'left':
                print(f'Rank {self.rank} sending left to rank {adjacent_rank}')
                print(f'The elements sent to left with temperature_grid is: {temperature_grid[:, 0]}')
                self.comm.send(temperature_grid[start:end, :1], dest = adjacent_rank, tag = 100 + self.rank)
                
    def receive_data(self, room, temperature_grid, dx):
        for direction, info in room.adjacent_rooms.items():
            gamma_type = info['gamma_type']
            adjacent_rank = info['rank']
            boundary_positions = info['boundary_positions']
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
                    room.boundary_mask[start+1:end+1, -1] = sol_new[:, 0]

                elif direction == 'left':
                    room.boundary_mask[start+1:end+1, 0] = sol_new[:, -1]

            if gamma_type == 'Neumann':
                if direction == 'right':
                    flux = (sol_new[:, -1] - sol_new[:, 0] )/ dx
                    print(f'sol:{sol_new}')
                    print(f'flux:{flux}')
                    print(f'boundary:{room.boundary_mask[start+1:end+1, -1]}')
                    print(f'temp:{temperature_grid[start:end,-1]}')
                    room.boundary_mask[start+1:end+1, -1] = temperature_grid[start:end,-1] + flux[1:-1] * dx

                elif direction == 'left':
                    flux = (sol_new[:, 0] - sol_new[:, -1])/ dx
                    print(f'flux:{flux}')
                    print(f'boundary:{room.boundary_mask[start+1:end+1, -1]}')
                    print(f'temp:{temperature_grid[start:end,-1]}')
                    room.boundary_mask[start+1:end+1, 0] = temperature_grid[start:end, 0] + flux[1:-1] * dx
                                                            
        return temperature_grid # Should we be returning this


def build_linalg(boundary_mask, Nx, Ny, dx):
    N = Nx * Ny  # Total number of elements
    coeff = 1 / dx**2
    A = []
    b = []
    

    # Create a mask for the boundary

    for j in range(Nx):
        for k in range(Ny):
            row = np.zeros(N)
            rhs = 0

            # check surrounding pixels and if they are outside the mask just take the boundary condition stored in boundary_mask
            if j-1 < 0: rhs -= boundary_mask[0, k]
            else: row[get_index(j-1, k, Ny)] = 1
            
            if j+1 >= Nx: rhs -= boundary_mask[-1, k]
            else: row[get_index(j+1, k, Ny)] = 1
            
            if k-1 < 0: rhs -= boundary_mask[j, 0]
            else: row[get_index(j, k-1, Ny)] = 1
            
            if k+1 >= Ny: rhs -= boundary_mask[j, -1]
            else: row[get_index(j, k+1, Ny)] = 1

            row[get_index(j, k, Ny)] = -4
            
            A.append(row)
            b.append(rhs)
    print(A)
    print(b)
    return csr_matrix(np.array(A)), np.array(b) # Maybe we need to add the coefficient stuff from the previous implementation

def get_index(j, k, Ny):
    return j * Ny + k

def solve_linalg(boundary_mask, temperature_grid, dx):
    shape = temperature_grid.shape
    print(f'Shape of the temperature_grid is: {shape}')
    n, m = shape

    A, b = build_linalg(boundary_mask, n, m, dx)
    sol = sp.linalg.spsolve(A, b).reshape((n, m))
    return sol

# save matrix A as latex document
def matrixLatex(A):
    latex_str = '\\begin{pmatrix}\n'
    for i in range(0, len(A)):
        therow = ''
        for j in range(0, len(A[i])):
            if j < len(A[i]) - 1:
                therow += str(A[i][j]) + ' & '
            elif i < len(A) - 1:
                therow += str(A[i][j]) + ' \\\\\n'
            else:
                therow += str(A[i][j]) + '\n'
        latex_str += therow
    latex_str += '\\end{pmatrix}'
    return latex_str

def createLatexDocument(matrix_str):
    document = r'''
    \documentclass{article}
    \usepackage{amsmath}  % Required for matrix environments
    \begin{document}

    Here is the matrix:

    \[
    ''' + matrix_str + r'''
    \]

    \end{document}
    '''
    filename = r'A_matrix_output.tex'

    with open(filename, 'w') as f:
        f.write(document, filename)

if __name__ == '__main__':
    # Create rooms with boundary conditions
    dx = 1 / 3
    Room1 = Room(dimension=[int(1/dx), int(1/dx)], boundary_condition={'left': 'heater'}, title='Room1')
    Room2 = Room(dimension=[int(2/dx), int(1/dx)], boundary_condition={'top': 'heater', 'bottom': 'window'}, title='Room2')
    Room3 = Room(dimension=[int(1/dx), int(1/dx)], boundary_condition={'right': 'heater'}, title='Room3')

    # Define adjacent rooms
    Room1.add_adjacent_room("right", Room2, boundary_positions={'start': 0, 'end': (int(1/dx))}, gamma_type = 'Neumann', rank = 1)
    Room2.add_adjacent_room("left", Room1, boundary_positions={'start': (int(1/dx)), 'end': (int(2/dx))}, gamma_type = 'Dirichlet', rank = 0)
    Room2.add_adjacent_room("right", Room3, boundary_positions={'start': 0, 'end': (int(1/dx))}, gamma_type = 'Dirichlet', rank =2)
    Room3.add_adjacent_room("left", Room2, boundary_positions={'start': 0, 'end': (int(1/dx))}, gamma_type = 'Neumann', rank = 1)

    # Create an apartment and add rooms
    apartment = Apartment()
    apartment.add_room(Room1)
    apartment.add_room(Room2)
    apartment.add_room(Room3)
    print('Hello We are rerunning this part')

    # Solve the system of equations and display results
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    solver = Dirichlet_Neumann(apartment.rooms, dx, comm, rank)
    val = solver.solve()
    plt.imshow(val, cmap='plasma')
    plt.colorbar(label='Temperature')
    plt.savefig('temperature.png')
    MPI.Finalize
    
    # plot_rooms(apartment)

    matrix1 = solver.temperature_values[0]
    matrix2 = solver.temperature_values[1]
    matrix3 = solver.temperature_values[2]

    fig = plt.figure(constrained_layout=True, figsize=(30, 20))

    # Create a GridSpec with 2 rows and 3 columns
    # Room 1 (bottom-left), Room 3 (top-right), Room 2 (middle connecting both)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2, 1], height_ratios=[1, 1])
    print('We are plotting')
    # Plot Room 1 (bottom-left)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(matrix1, cmap='plasma')
    ax1.set_title("Room 1 (Bottom-left, 10x10)")
    ax1.axis('off')  # Turn off axis for better visualization

    # Plot Room 2 (middle-right), spanning the second column vertically
    ax2 = fig.add_subplot(gs[:, 1])  # Spanning both rows in the middle column
    ax2.imshow(matrix2, cmap='plasma')
    ax2.set_title("Room 2 (Middle, 10x20)")
    ax2.axis('off')

    # Plot Room 3 (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(matrix3, cmap='plasma')
    ax3.set_title("Room 3 (Top-right, 10x10)")
    ax3.axis('off')


    # Show the final plot
    plt.savefig(f'apartment_{rank}.png')

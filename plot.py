from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def add_adjacent_room(self, direction, room, boundary_positions, gamma_type=None, rank=None):
        # Check if the direction key already exists
        if direction in self.adjacent_rooms:
            # Find a new key with an incrementing suffix (_2, _3, etc.)
            suffix = 2
            new_direction = f"{direction}_{suffix}"
            while new_direction in self.adjacent_rooms:
                suffix += 1
                new_direction = f"{direction}_{suffix}"

            # Add the new room under the unique key
            self.adjacent_rooms[new_direction] = {
                'room': room,
                'boundary_positions': boundary_positions,
                'gamma_type': gamma_type,
                'rank': rank
            }
        else:
            # Add the first room with the original direction key
            self.adjacent_rooms[direction] = {
                'room': room,
                'boundary_positions': boundary_positions,
                'gamma_type': gamma_type,
                'rank': rank
            }

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
        # print(self.boundary_mask)

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
                    if num != 0:  u = self.receive_data(room, temperature_grid, self.dx)
                    else: u = temperature_grid

                    sol = solve_linalg(room.boundary_mask, u, self.dx)
                    if num == 0: sol_relax = sol
                    else: sol_relax = self.omega * sol + (1 - self.omega) * self.temperature_values[i]
                    self.temperature_values[i] = sol_relax
                    temperature_grid = sol_relax
                    self.send_data(room, temperature_grid)

                else: pass
        return locals().get('sol_relax', None)
            
    def send_data(self, room, temperature_grid):
        for direction, info in room.adjacent_rooms.items():
            adjacent_rank = info['rank']
            boundary_positions = info['boundary_positions']
            start = boundary_positions['start']
            end = boundary_positions['end']

            if direction.startswith('right'):
                self.comm.send(temperature_grid[start:end, -2:], dest = adjacent_rank,  tag = 100 + self.rank)

            if direction.startswith('left'):
                self.comm.send(temperature_grid[start:end, :2], dest = adjacent_rank, tag = 100 + self.rank)

    def receive_data(self, room, temperature_grid, dx):
        for direction, info in room.adjacent_rooms.items():
            gamma_type = info['gamma_type']
            adjacent_rank = info['rank']
            boundary_positions = info['boundary_positions']
            start = boundary_positions['start']
            end = boundary_positions['end']

            try:
                sol_new = self.comm.recv(source=adjacent_rank, tag=100 + adjacent_rank)

            except Exception as e:
                print(f"Error receiving data on rank {self.rank}: {e}")
            
            if gamma_type == 'Dirichlet':
                if direction.startswith('right'):
                    room.boundary_mask[start+1:end+1, -1] = sol_new[:, 0]

                elif direction.startswith('left'):
                    room.boundary_mask[start+1:end+1, 0] = sol_new[:, -1]

            if gamma_type == 'Neumann':
                if direction.startswith('right'):
                    flux = (sol_new[:, -1] - sol_new[:, 0])/ dx
                    room.boundary_mask[start+1:end+1, -1] = temperature_grid[start:end,-1] + flux * dx

                elif direction.startswith('left'):
                    flux = (sol_new[:, 0] - sol_new[:, -1])/ dx
                    room.boundary_mask[start+1:end+1, 0] = temperature_grid[start:end, 0] + flux * dx

        return temperature_grid


def build_linalg(boundary_mask, Nx, Ny, dx):
    N = Nx * Ny  # Total number of elements
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
   
    return csr_matrix(np.array(A)), np.array(b)

def get_index(j, k, Ny):
    return j * Ny + k

def solve_linalg(boundary_mask, temperature_grid, dx):
    shape = temperature_grid.shape
    # print(f'Shape of the temperature_grid is: {shape}')
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

def vectorLatex(b):
    latex_str = '\\begin{pmatrix}\n'
    e = '\\begin{pmatrix}\n'
    for i in range(0, len(b)):
        if i != len(b)-1:
            latex_str += str(b[i]) + ' \\\\\n '
            e += 'e' + str(i) + ' \\\\\n '
        else:
            latex_str += str(b[i])
            e += 'e' + str(i)
    latex_str += '\\end{pmatrix}'
    e += '\\end{pmatrix}'
    return latex_str, e

def createLatexDocument(A, b, roomNum):
    matrix_str = matrixLatex(A)
    vector_str, e = vectorLatex(b)
    
    document = r'''
    \documentclass{article}
    \usepackage{amsmath}  % Required for matrix environments
    \begin{document}

    Here is the matrix for the ''' + roomNum + r''' room (at last iteration):

    \[
    ''' + matrix_str + e + '=' + vector_str + r'''
    \]

    \end{document}
    '''
    filename = r'.\latex\A_matrix_output_' + roomNum + r'.tex'

    with open(filename, 'w') as f:
        f.write(document)


if __name__ == '__main__':
    dx = 1 / 50
    # Solve the system of equations and display results
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Create the apartment and add rooms in rank 0
        Room1 = Room(dimension=[int(1 / dx), int(1 / dx)], boundary_condition={'left': 'heater'}, title='Room1')
        Room2 = Room(dimension=[int(2 / dx), int(1 / dx)], boundary_condition={'top': 'heater', 'bottom': 'window'},
                     title='Room2')
        Room3 = Room(dimension=[int(1 / dx), int(1 / dx)], boundary_condition={'right': 'heater'}, title='Room3')
        Room4 = Room(dimension = [int(1/(2*dx)), int(1/(2*dx))], boundary_condition = {'bottom': 'heater'}, title = 'Room4')

        # Define adjacent rooms
        Room1.add_adjacent_room("right", Room2, boundary_positions={'start': 0, 'end': int(1 / dx)},
                                gamma_type='Neumann', rank=1)
        Room2.add_adjacent_room("left", Room1, boundary_positions={'start': int(1 / dx), 'end': int(2 / dx)},
                                gamma_type='Dirichlet', rank=0)
        Room2.add_adjacent_room("right", Room3, boundary_positions={'start': 0, 'end': int(1 / dx)},
                                gamma_type='Dirichlet', rank=2)
        Room2.add_adjacent_room("right", Room4, boundary_positions={'start': (int(1/dx)), 'end': (int(2/dx*3/4))}, gamma_type = 'Dirichlet', rank = 3 )
        Room3.add_adjacent_room("left", Room2, boundary_positions={'start': 0, 'end': int(1 / dx)},
                                gamma_type='Neumann', rank=1)
        Room4.add_adjacent_room("left", Room2, boundary_positions= {'start': 0, 'end': int(1/(2*dx))}, gamma_type = 'Neumann', rank= 1)

        # Create an apartment and add rooms
        apartment = Apartment()
        apartment.add_room(Room1)
        apartment.add_room(Room2)
        apartment.add_room(Room3)
        apartment.add_room(Room4)

    else:
        apartment = None

    # Broadcast apartment to all ranks
    apartment = comm.bcast(apartment, root=0)


    solver = Dirichlet_Neumann(apartment.rooms, dx, comm, rank, 10)
    val = solver.solve()
    temperature_values = comm.gather(val, root=0)
    print('done '+str(rank))

    if rank == 0:

        matrix1 = temperature_values[0]  # Temperature for Room 1
        matrix2 = temperature_values[1]  # Temperature for Room 2
        matrix3 = temperature_values[2]  # Temperature for Room 3
        matrix4 = temperature_values[3]  # Temperature for Room 4

        # Create the figure for plotting
        fig = plt.figure(constrained_layout=True, figsize=(30, 18.3))

        # Create a GridSpec with 3 rows and 3 columns
        # Room 1 (bottom-left), Room 3 (top-right), Room 2 (middle connecting both), Room 4 (below Room 3)
        gs = fig.add_gridspec(4, 6)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        # gs.tight_layout(fig, pad=0)
        print(f'We are plotting, {gs}')

        # Set a consistent color scale for all rooms by determining vmin and vmax from all matrices
        vmin = min(matrix.min() for matrix in temperature_values if matrix is not None)
        vmax = max(matrix.max() for matrix in temperature_values if matrix is not None)

        # Plot Room 1 (bottom-left)
        ax1 = fig.add_subplot(gs[2:, :2])
        img1 = ax1.imshow(matrix1, cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
        ax1.axis('off')

        # Plot Room 2 (middle-right), spanning the second column vertically
        ax2 = fig.add_subplot(gs[:, 2:4])  # Spanning both rows in the middle column
        img2 = ax2.imshow(matrix2, cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
        ax2.axis('off')

        # Plot Room 3 (top-right)
        ax3 = fig.add_subplot(gs[:2, 4:])
        img3 = ax3.imshow(matrix3, cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
        ax3.axis('off')

        # Plot Room 4 (below Room 3, half the width, square)
        ax4 = fig.add_subplot(gs[2, 4:5])
        img4 = ax4.imshow(matrix4, cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
        ax4.axis('off')

        # Add colorbar associated with one of the images (for the entire figure)
        cbar = fig.colorbar(img1, ax=[ax1, ax2, ax3, ax4], orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Temperature', fontsize=18)  # Set label font size
        cbar.ax.tick_params(labelsize=14)  # Set tick font size for the colorbar

        # Show the final plot
        plt.savefig('apartment_plot.png')

    MPI.Finalize()

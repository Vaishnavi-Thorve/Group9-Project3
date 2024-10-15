from math import gamma
# from unicodedata import ucd_3_2_0

from mpi4py import MPI
import numpy as np
from numpy.matlib import zeros
from scipy.linalg import solve
from scipy.sparse import diags
import matplotlib.pyplot as plt


def build_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp):
    A = []
    b = []

    Nx = int(room_width / mesh_size)
    Ny = int(room_height / mesh_size)
    print(f'Nx is {Nx}, Ny is {Ny}')
    for j in range(Nx):
        for k in range(Ny):
            N = Nx * Ny
            row = np.zeros(N)
            rhs = 0
            diag = -4

            print(f'Processing point (j={j}, k={k})')
            row, rhs, diag = apply_bc(j-1, k, row, rhs, diag, bc, gamma_type, gamma_temp)
            row, rhs, diag = apply_bc(j+1, k, row, rhs, diag, bc, gamma_type, gamma_temp)
            row, rhs, diag = apply_bc(j, k-1, row, rhs, diag, bc, gamma_type, gamma_temp)
            row, rhs, diag = apply_bc(j, k+1, row, rhs, diag, bc, gamma_type, gamma_temp)

            row[get_index(j, k, Nx)] = diag
            A.append(row)
            b.append(rhs)

    print("Linear system matrix A and vector b built successfully.")
    return np.array(A), np.array(b)


def get_index(j, k, Nx):
    index = j * Nx + k
    print(f'Index for (j={j}, k={k}) is {index}')
    return index


def apply_bc(j, k, row, rhs, diag, bc, gamma_type, gamma_temp):
    print(f'Applying boundary condition at (j={j}, k={k})')
    bound = bc(j, k)
    print(f'Boundary type: {bound}')
    if bound == 'interior':
        row[get_index(j - 1, k, 3)] = 1
    elif bound == 'heater':
        rhs -= 40
    elif bound == 'window':
        rhs -= 5
    elif bound == 'wall':
        rhs -= 15
    elif bound == 'gamma':
        if gamma_type == 'Dirichlet':
            rhs -= gamma_temp
        elif gamma_type == 'Neumann':
            diag += 1

    return row, rhs, diag


def solve_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp=0):
    print(f'Starting to build linear system for room of size ({room_width}x{room_height})')
    A, b = build_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp)
    print('Linear system matrix A:')
    print(A)
    print('Right-hand side vector b:')
    print(b)
    print('Solving linear system...')
    solution = np.linalg.solve(A, b)
    print('Solution found.')
    return solution

# %%
if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Size of rooms
        Nx1, Ny1 = 1, 1  # Room 1 (1x1)
        Nx2, Ny2 = 1, 2  # Room 2 (1x2)
        Nx3, Ny3 = 1, 1  # Room 3 (1x1)

        dx = 1 / 3

        # Boundary temperatures
        u_wall = 15  # Normal wall
        u_heater = 40  # Heater wall ΓH
        u_window = 5  # Window wall ΓWF

        # Relaxation parameter
        omega = 0.8

        def bc_1(x, y):
            if x < 0:
                return 'heater'
            if y < 0 or y > 1:
                return 'wall'
            if x > 1:
                return 'gamma'
            return 'interior'

        def bc_2(x, y):
            if y < 0:
                return 'window'

            if x > 1 and y < 1 or x < 0 and y > 1:
                return 'wall'

            if x > 1 and y >= 1 or x < 0 and y <= 1:
                return 'gamma'

            if y > 2:
                return 'heater'

            return 'interior'

        def bc_3(x, y):
            if x > 1:
                return 'heater'
            if y < 0 or y > 1:
                return 'wall'
            if x < 0:
                return 'gamma'
            return 'interior'

        print('Starting solver for Room 1...')
        s = solve_linalg(Nx1, Ny1, bc_1, 1/3, 'Neumann')
        print('Solution for Room 1:')
        print(s)
        s_new = s.reshape(3, 3)
        plt.imshow(s_new, cmap='gray')
        plt.savefig("testfig.png")
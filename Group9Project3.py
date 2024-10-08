from math import gamma
# from unicodedata import ucd_3_2_0

from mpi4py import MPI
import numpy as np
from numpy.matlib import zeros
from scipy.linalg import solve
from scipy.sparse import diags
import matplotlib.pyplot as plt



def build_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp_1, gamma_temp_3):
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

            row, rhs, diag = apply_bc(j-1, k, row, rhs, diag, bc, gamma_type, gamma_temp_1, gamma_temp_3, Ny, mesh_size)
            row, rhs, diag = apply_bc(j+1, k, row, rhs, diag, bc, gamma_type, gamma_temp_1, gamma_temp_3, Ny, mesh_size)
            row, rhs, diag = apply_bc(j, k-1, row, rhs, diag, bc, gamma_type, gamma_temp_1, gamma_temp_3, Ny, mesh_size)
            row, rhs, diag = apply_bc(j, k+1, row, rhs, diag, bc, gamma_type, gamma_temp_1, gamma_temp_3, Ny, mesh_size)


            row[get_index(j,k,Ny)] = diag
            A.append(row)
            b.append(rhs)

    return np.array(A), np.array(b)


def get_index(j, k, Ny):
    return j * Ny + k


def apply_bc(j, k, row, rhs, diag, bc, gamma_type, gamma_temp_1, gamma_temp_3, Ny, mesh_size):
    bound = bc(j, k, mesh_size)
    if bound == 'interior':
        row[get_index(j, k, Ny)] = 1
    elif bound == 'heater':
        # rhs -= u_heater
        rhs -= 40
    elif bound == 'window':
        # rhs -= u_window
        rhs -= 5
    elif bound == 'wall':
        # rhs -= u_wall
        rhs -= 15
    elif bound == 'gamma_1' or bound == 'gamma_3':
        if gamma_type == 'Dirichlet':
            # gamma can just be on the left or right side of the room (could still not work)
            if bound == 'gamma_1': rhs -= gamma_temp_1[max(0,j-1)]
            if bound == 'gamma_3': rhs -= gamma_temp_3[max(0,j-1)]
        elif gamma_type == 'Neumann':
            # want in the rhs -= u_NC (which is some sort of flux) * mesh_size
            diag += 1

    return row, rhs, diag

def solve_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp_1 = None, gamma_temp_3 = None):
    A, b = build_linalg(room_width, room_height, bc, mesh_size, gamma_type, gamma_temp_1, gamma_temp_3)

    sol = np.linalg.solve(A, b)
    Nx = int(room_width / mesh_size)
    Ny = int(room_height / mesh_size)
    sol = np.rot90(sol.reshape(Nx, Ny), k=1, axes=(0,1))
    return sol

# %%
if __name__ == '__main__':
    # Size of rooms
    Nx1, Ny1 = 1, 1  # Room 1 (1x1)
    Nx2, Ny2 = 1, 2  # Room 2 (1x2)
    Nx3, Ny3 = 1, 1  # Room 3 (1x1)

    dx = 1 / 20

    # Boundary temperatures
    u_wall = 15  # Normal wall
    u_heater = 40  # Heater wall ΓH
    u_window = 5  # Window wall ΓWF

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Relaxation parameter
    omega = 0.8
    no_iterations = 10


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




    for iteration in range(no_iterations):
        # Solve room 2 with Dirichlet
        if rank == 1:
            if iteration == 0:
                room_2 = solve_linalg(Nx2, Ny2, bc_2, dx, 'Neumann')

            else:
                from_room_1 = comm.recv(source=0, tag=12)
                from_room_3 = comm.recv(source=2, tag=32)

                room_2 = solve_linalg(Nx2, Ny2, bc_2, dx, 'Dirichlet', from_room_1, from_room_3)
            # Communicate the boundary conditions

            comm.send(room_2[0:Ny1-1,0],dest=0, tag=21)  # send to room 1
            comm.send(room_2[0:Ny3-1,-1],dest=2, tag=23)  # send to room 3

        # Solve room 1 and 3 with Neumann
        if rank == 0:
            from_room_2 = comm.recv(source=1,tag=21)  # Receive boundary from room 2

            room_1 = solve_linalg(Nx2, Ny2, bc_2, dx, 'Neumann', from_room_2)

            comm.send(room_1[:,1],dest=1, tag=12)

        if rank == 2:
            from_room_2 = comm.recv(source=1,tag=23)

            room_3 = solve_linalg(Nx3, Ny3, bc_3, dx, 'Neumann', None, from_room_2)

            comm.send(room_3[:,0],dest=1, tag=32)

        if iteration == 0:
            if rank == 1:
                room_2_old = room_2
            if rank == 0:
                room_1_old = room_1
            if rank == 2:
                room_3_old = room_3
            continue

        # Relax
        if rank == 1:
            room_2 = omega * room_2 + (1 - omega) * room_2_old
        elif rank == 0:
            room_1 = omega * room_1 + (1 - omega) * room_1_old
        elif rank == 2:
            room_3 = omega * room_3 + (1 - omega) * room_3_old


    # s = solve_linalg(Nx1, Ny1, bc_1, 1/20, 'Neumann')
    # plt.imshow(s, cmap='plasma')
    # plt.title('Room 1')
    # plt.show()
    #
    # s = solve_linalg(Nx2, Ny2, bc_2, 1/20, 'Neumann')
    # plt.imshow(s, cmap='plasma')
    # plt.title('Room 2 - Neumann')
    # plt.show()
    #
    # g1 = np.full(20,5)
    # g2 = np.full(20, 10)
    # s = solve_linalg(Nx2, Ny2, bc_2, 1/20, 'Dirichlet', g1, g2)
    # plt.imshow(s, cmap='plasma')
    # plt.title('Room 2 - Dirichlet')
    # plt.show()
    #
    #
    # s = solve_linalg(Nx3, Ny3, bc_3, 1/20, 'Neumann')
    # plt.imshow(s, cmap='plasma')
    # plt.title('Room 3')
    # plt.show()


import math
import sys
import numpy as np
import matplotlib.pyplot as plt


def simulate_taylor(length=20.0,
                    width=5.0,
                    depth=1.0,
                    dispersion_coefficient=0.01,
                    advection_coefficient=0.1,
                    injection_point_x=2.0,
                    measure_points=None,
                    mass=1.0,
                    timestep=0.5,
                    spatial_step=0.1,
                    simulation_time=200.0):
    if measure_points is None:
        measure_points = [6.0, 12.0, 18.0]
    initial_c = mass / width / depth / spatial_step
    cd = dispersion_coefficient * timestep / spatial_step / spatial_step
    ca = advection_coefficient * timestep / spatial_step
    c1 = cd * (1 - ca) - ca / 6 * (ca * ca - 3 * ca + 2)
    c2 = cd * (2 - 3 * ca) - ca / 2 * (ca * ca - 2 * ca - 1)
    c3 = cd * (1 - 3 * ca) - ca / 2 * (ca * ca - ca - 2)
    c4 = ca * cd + ca / 6 * (ca * ca - 1)
    print(ca, cd)
    solutions = np.zeros((math.floor(simulation_time / timestep), math.floor(length / spatial_step)))
    solutions[:, math.floor(injection_point_x / spatial_step)] = initial_c

    masses = []
    for i in np.arange(timestep, simulation_time, timestep):
        index = math.floor(i / timestep)
        solutions[index] = solutions[index - 1] \
                           + c1 * np.pad(solutions[index - 1], (0, 1))[1:] \
                           - c2 * solutions[index - 1] + c3 * np.pad(solutions[index - 1], (1, 0))[:-1] \
                           + c4 * np.pad(solutions[index - 1], (2, 0))[:-2]
        solutions[index][-1] = solutions[index - 1][-1]
        solutions[index][0] = 0
        masses.append(np.cumsum(solutions[index])[-1])

    for measure_point in measure_points:
        plt.plot([i[math.floor(measure_point / spatial_step)] for i in solutions])

    plt.xlabel("krok czasowy")
    plt.ylabel("masa znacznika w danym punkcie pomiarowym w czasie")
    plt.show()

    plt.xlabel("krok czasowy")
    plt.ylabel("masa znacznika w całym kanale")
    plt.plot(masses)
    plt.show()


def simulate_taylor_2(length=20.0,
                      width=5.0,
                      depth=1.0,
                      dispersion_coefficient=0.01,
                      advection_coefficient=0.1,
                      injection_point_x=2.0,
                      measure_points=None,
                      mass=1.0,
                      timestep=0.5,
                      spatial_step=0.1,
                      simulation_time=200.0):
    if measure_points is None:
        measure_points = [6.0, 12.0, 18.0]
    initial_c = mass / width / depth / spatial_step
    cd = dispersion_coefficient * timestep / spatial_step / spatial_step
    ca = advection_coefficient * timestep / spatial_step
    print(ca, cd)
    aa = np.zeros((math.floor(length / spatial_step), math.floor(length / spatial_step)))
    bb = np.zeros((math.floor(length / spatial_step), math.floor(length / spatial_step)))

    for i in range(math.floor(length / spatial_step)):
        aa[i][i] = 1 + cd
        bb[i][i] = 1 - cd

        if i < math.floor(length / spatial_step) - 1:
            aa[i][i + 1] = ca/4 - cd/2
            bb[i][i + 1] = cd/2 - ca/4

        if i > 0:
            aa[i][i - 1] = - ca / 4 - cd / 2
            bb[i][i - 1] = cd / 2 + ca / 4

    ab = np.matmul(np.linalg.inv(aa), bb)
    solutions = np.zeros((math.floor(simulation_time / timestep), math.floor(length / spatial_step)))
    solutions[:, math.floor(injection_point_x / spatial_step)] = initial_c

    masses = []
    for i in np.arange(timestep, simulation_time, timestep):
        index = math.floor(i / timestep)
        solutions[index] = np.matmul(ab, solutions[index - 1])
        solutions[index][-1] = solutions[index - 1][-1]
        solutions[index][0] = 0
        masses.append(np.cumsum(solutions[index])[-1])

    for measure_point in measure_points:
        plt.plot([i[math.floor(measure_point / spatial_step)] for i in solutions])

    plt.xlabel("krok czasowy")
    plt.ylabel("masa znacznika w danym punkcie pomiarowym w czasie")
    plt.show()

    plt.xlabel("krok czasowy")
    plt.ylabel("masa znacznika w całym kanale")
    plt.plot(masses)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '1' in sys.argv:
            simulate_taylor()
        if '2' in sys.argv:
            simulate_taylor_2()

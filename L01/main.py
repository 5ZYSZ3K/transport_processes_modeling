# This is a sample Python script.
import math

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy

# get_brown_movements
# A function that takes an integer as a parameter, returns a brown movements tuple of two number arrays: x and y values

def get_brown_movements(number):
    positions = [[0], [0]]
    for i in range(number - 1):
        positions[0].append(positions[0][i] + randn())
        positions[1].append(positions[1][i] + randn())
    return positions


# plot_tuple
# A function that takes a x, y tuple as a parameter and plots it


def plot_tuple(positions):
    plt.plot(positions[0], positions[1])
    plt.show()


# calculate_squared_average
# A function that simply takes two integers as a parameters and returns the sum of both values squared


def calculate_squared_average(x, y):
    return math.pow(x, 2) + math.pow(y, 2)


# get_moves_array
# A function, that takes an integer (n) as a parameter, and returns a n x n array filled with brown moves (see below)


def get_moves_array(number):
    moves = [[None] * number for _ in range(number)]
    for i in range(number):
        for j in range(number):
            x, y = get_next_move()
            moves[i][j] = calculate_squared_average(x, y)
    return moves


# get_next_move
# a function that returns a tuple of two random numbers, x and y (brown move)


def get_next_move():
    return [randn(), randn()]

# get_average_movements
# a function that takes an integer (n) as a parameter
# Returns a tuple: at index 0 there will be an array with indexes
# At the second position array filled with average distance travelled by a brown particle
# (simulating n units of time for n particles)


def get_average_movements(number):
    current_averages = [[], []]
    moves_array = get_moves_array(number)
    curr_array = [0] * number
    for i in range(len(moves_array)):
        curr_array = numpy.add(curr_array, moves_array[i])
        current_averages[0].append(i)
        current_averages[1].append(sum(curr_array) / len(curr_array))
    return current_averages


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    averages = get_average_movements(1000)
    brown_movements = get_brown_movements(1000)
    plot_tuple(averages)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy
import math
import matplotlib.pyplot as plot
import random
import numba

G = 6.67408e-11

class body:
    def __init__(self, position, velocity, mass, name):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.name = name


SUN = body(numpy.array([0.0,0.0,0.0]), numpy.array([0.0,0.0,0.0]), 2e30, "Sun")
MERCURY = body(numpy.array([0.0,5.7e10,0.0]), numpy.array([47000.0,0.0,0.0]), 3.285e23, "Mercury")
VENUS = body(numpy.array([0.0,1.1e11,0.]), numpy.array([35000.0,0.0,0.0]), 4.8e24, "Venus")
EARTH = body(numpy.array([0.0,1.5e11,0.0]), numpy.array([30000.0,0.0,0.0]), 6e24, "Earth")
MARS = body(numpy.array([0.0,2.2e11,0.0]), numpy.array([24000.0,0.0,0.0]), 2.4e24, "Mars")
JUPITER = body(numpy.array([0.0,7.7e11,0.0]), numpy.array([13000.0,0.0,0.0]), 1e28, "Jupiter")
SATURN = body(numpy.array([0.0,1.4e12,0.0]), numpy.array([9000.0,0.0,0.0]), 5.7e26, "Saturn")
URANUS = body(numpy.array([0.0,2.8e12,0.0]), numpy.array([6835.0,0.0,0.0]), 8.7e25, "Uranus")
NEPTUNE = body(numpy.array([0.0,4.5e12,0.0]), numpy.array([5477.0,0.0,0.0]), 1e26, "Neptune")
PLUTO = body(numpy.array([0.0,3.7e12,0.0]), numpy.array([4748.0,0.0,0.0]), 1.3e22, "Pluto")

@numba.jit("Tuple((float64, float64[:]))(float64[:],float64[:])",nopython=True)
def radius(vector1, vector2):
    r_vec = vector1 - vector2
    r_mag = (r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)**0.5
    return r_mag, r_vec

def update_acceleration(bodies, current_index):
    acceleration = numpy.array([0.0,0.0,0.0])
    current_body = bodies[current_index]
    for index, selected_body in enumerate(bodies):
        if index != current_index:
            r, r_vec = radius(selected_body.position, current_body.position)
            scalar = (G * selected_body.mass) / (r**3)
            acceleration += (scalar * r_vec)

    return acceleration

def update_velocity(bodies, dT = 1.0):
    for index, selected_body in enumerate(bodies):
        acceleration = update_acceleration(bodies, index)
        selected_body.velocity += acceleration * dT

def update_position(bodies, dT = 1.0):
    for selected_body in bodies:
        selected_body.position += selected_body.velocity * dT

def update_bodies(bodies, dT = 1.0):
    update_velocity(bodies, dT=dT)
    update_position(bodies, dT=dT)

def simulate(bodies, dT = 1000.0, T = 10000, output_freq = 100):
    trajectory = []

    for current_body in bodies:
        trajectory.append({"x":[],"y":[],"z":[],"name":current_body.name})

    for i in range(1, int(T)):
        update_bodies(bodies, dT=dT)

        if i % output_freq == 0:
            for index, position in enumerate(trajectory):
                position["x"].append(bodies[index].position[0])
                position["y"].append(bodies[index].position[1])
                position["z"].append(bodies[index].position[2])

    return trajectory

def visualise(trajectory):
    fig = plot.figure()
    colours = ["r","b","g","y","m","c"]
    ax = fig.add_subplot(1,1,1, projection="3d")
    max_range = 0

    for index, selected_body in enumerate(trajectory):
        max_x = max(selected_body["x"])
        max_y = max(selected_body["y"])
        max_z = max(selected_body["z"])
        max_dimension = max(max_x, max_y, max_z)
        if max_dimension > max_range:
            max_range = max_dimension

        ax.plot(selected_body["x"], selected_body["y"], selected_body["z"], c = colours[index%len(colours)], label = selected_body["name"])

    ax.set_xlim([-max_range,max_range])
    ax.set_ylim([-max_range,max_range])
    ax.set_zlim([-max_range,max_range])

    plot.show()

if __name__ == "__main__":
    bodies = numpy.array([SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE,PLUTO])
    print(f"Simulating {len(bodies)} bodies.")
    trajectory = simulate(bodies, dT=10000.0, T=160000, output_freq=100)
    print("Visualising.")
    visualise(trajectory)

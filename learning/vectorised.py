import numpy
import math
import random
import numba
import visualisation
import time

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

def convert_bodies(bodies):
    nbodies = len(bodies)
    state = numpy.zeros((nbodies,7), dtype=numpy.float64)
    labels = []

    for a in range(nbodies):
        labels.append(bodies[a].name)
        state[a,0:3] = bodies[a].position
        state[a,3:6] = bodies[a].velocity
        state[a,6] = bodies[a].mass

    return labels, state

@numba.jit("Tuple((float64, float64[:]))(float64[:],float64[:])",nopython=True)
def radius(vector1, vector2):
    r_vec = vector1 - vector2
    r_mag = (r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)**0.5
    return r_mag, r_vec

def partial_step(vector1, vector2, dT):
    return vector1 + (vector2 * dT)

def partial_step_v(position, other_positions, dT):
    position_ex = numpy.repeat(numpy.reshape(position,(1,3)), len(other_positions),0)
    r_vec = position_ex - other_positions
    return dT * r_vec

def update_acceleration(bodies, current_index, dT):
    acceleration = numpy.array([0.0,0.0,0.0])
    current_body = bodies[current_index]
    for index, selected_body in enumerate(bodies):
        if index != current_index:
            r, r_vec = radius(selected_body.position, current_body.position)
            scalar = (G * selected_body.mass) / (r**3)
            acceleration += (scalar * r_vec)

    return acceleration

def update_acceleration_rk4(bodies, current_index, dT):
    acceleration = numpy.array([0.0,0.0,0.0])
    current_body = bodies[current_index]

    k1 = numpy.array([0.0,0.0,0.0])
    k2 = numpy.array([0.0,0.0,0.0])
    k3 = numpy.array([0.0,0.0,0.0])
    k4 = numpy.array([0.0,0.0,0.0])

    temp_position = numpy.array([0.0,0.0,0.0])
    temp_velocity = numpy.array([0.0,0.0,0.0])

    for index, selected_body in enumerate(bodies):
        if index != current_index:
            r, r_vec = radius(selected_body.position, current_body.position)
            scalar = (G * selected_body.mass) / (r**3)

            k1 = scalar * r_vec

            temp_velocity = partial_step(current_body.velocity, k1, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k2 = scalar * (selected_body.position - temp_position)

            temp_velocity = partial_step(current_body.velocity, k2, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k3 = scalar * (selected_body.position - temp_position)

            temp_velocity = partial_step(current_body.velocity, k3, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k4 = scalar * (selected_body.position - temp_position)

            acceleration += (k1 + (k2 * 2) + (k3 * 2) + k4)/6


    return acceleration


def radius_v(position, other_positions):
    position_ex = numpy.repeat(numpy.reshape(position,(1,3)), len(other_positions),0)
    r_vec = position_ex - other_positions
    r =  sum((r_vec * r_vec).transpose())** 0.5
    r = r.reshape(len(r),1)
    return r, r_vec

def update_acceleration_euler_v(state, current_index, dT):
    acceleration = numpy.array([0.0,0.0,0.0])
    current_body = state[current_index]

    other_states = numpy.delete(state,current_index,0)
    r,r_vec = radius_v(current_body[0:3], other_states[:,0:3])
    scalar = numpy.repeat(G * other_states[:,6:7]/(r**3), 3,1)

    acceleration = sum(scalar * r_vec)
    return acceleration

def update_acceleration_rk4_v(state, current_index, dT):
    acceleration = numpy.array([0.0,0.0,0.0])
    current_body = state[current_index]
    nbodies = len(state)

    k1 = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)
    k2 = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)
    k3 = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)
    k4 = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)

    temp_position = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)
    temp_velocity = numpy.zeros((nbodies - 1, 3), dtype=numpy.float64)
    other_states = numpy.delete(state,current_index,0)
    r,r_vec = radius_v(current_body[0:3], other_states[:,0:3])
    scalar = numpy.repeat(G * other_states[:,6:7]/(r**3), 3,1)

    k1 = scalar * r_vec

    temp_velocity = partial_step_v(current_body[3:6], k1, 0.5)
    temp_position = partial_step_v(current_body[0:3], temp_velocity, 0.5 * dT)
    k2 = scalar * (other_states[:,0:3] - temp_position)


    temp_velocity = partial_step_v(current_body[3:6], k2, 0.5)
    temp_position = partial_step_v(current_body[0:3], temp_velocity, 0.5 * dT)
    k3 = scalar * (other_states[:,0:3] - temp_position)


    temp_velocity = partial_step_v(current_body[3:6], k3, 0.5)
    temp_position = partial_step_v(current_body[0:3], temp_velocity, 0.5 * dT)
    k4 = scalar * (other_states[:,0:3] - temp_position)


    acceleration = sum((k1 + (k2 * 2) + (k3 * 2) + k4)/6)

    return acceleration
"""
    for index, selected_body in enumerate(bodies):
        if index != current_index:
            r, r_vec = radius(selected_body.position, current_body.position)
            scalar = (G * selected_body.mass) / (r**3)

            k1 = scalar * r_vec

            temp_velocity = partial_step(current_body.velocity, k1, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k2 = scalar * (selected_body.position - temp_position)

            temp_velocity = partial_step(current_body.velocity, k2, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k3 = scalar * (selected_body.position - temp_position)

            temp_velocity = partial_step(current_body.velocity, k3, 0.5)
            temp_position = partial_step(current_body.position, temp_velocity, 0.5*dT)
            k4 = scalar * (selected_body.position - temp_position)

            acceleration += (k1 + (k2 * 2) + (k3 * 2) + k4)/6
"""


def update_velocity(bodies, dT = 1.0):
    for index, selected_body in enumerate(bodies):
        acceleration = update_acceleration_rk4(bodies, index, dT)
        #acceleration = update_acceleration(bodies, index, dT)
        selected_body.velocity += acceleration * dT

def update_velocity_v(states, dT = 1.0):
    for a in range(len(states)):
        #acceleration = update_acceleration_rk4_v(states, a, dT)
        acceleration = update_acceleration_euler_v(states, a, dT)
        states[a,3:6] += acceleration * dT

def update_position_v(states, dT = 1.0):
    for a in range(len(states)):
        states[a,0:3] += states[a,3:6] * dT

def update_states_v(states, dT=1.0):
    update_velocity_v(states, dT=dT)
    update_position_v(states, dT=dT)

def update_position(bodies, dT = 1.0):
    for selected_body in bodies:
        selected_body.position += selected_body.velocity * dT

def update_bodies(bodies, dT = 1.0):
    update_velocity(bodies, dT=dT)
    update_position(bodies, dT=dT)

def simulate(bodies, dT = 1000.0, T = 10000, output_freq = 100):
    trajectory = []

    labels,state = convert_bodies(bodies)
    print(state)

    for current_body in bodies:
        trajectory.append({"x":[],"y":[],"z":[],"name":current_body.name})

    for i in range(1, int(T)):
        #update_bodies(bodies, dT=dT)
        update_states_v(state, dT=dT)

        if i % output_freq == 0:
            for index, position in enumerate(trajectory):
                """
                position["x"].append(bodies[index].position[0])
                position["y"].append(bodies[index].position[1])
                position["z"].append(bodies[index].position[2])
                """
                position["x"].append(state[index,0])
                position["y"].append(state[index,1])
                position["z"].append(state[index,2])
    return trajectory

if __name__ == "__main__":
    year = 3600.0 * 24 * 365
    dT = 1000.0
    number_steps = math.ceil(year/dT)
    bodies = numpy.array([SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE,PLUTO])
    #bodies = numpy.array([SUN,MERCURY,VENUS,EARTH,MARS])

    print(f"Simulating {len(bodies)} bodies.")
    start = time.time()
    trajectory = simulate(bodies, dT=dT, T=number_steps, output_freq=100)
    stop = time.time()
    elapsed = stop - start
    print(f"Time taken: {elapsed} seconds")
    print("Visualising.")
    visualisation.visualise(trajectory)

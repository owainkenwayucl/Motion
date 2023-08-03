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

#@numba.jit("Tuple((float64, float64[:]))(float64[:],float64[:])",nopython=True)
#@def radius(vector1, vector2):
#    r_vec = vector1 - vector2
#    r_mag = (r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)**0.5
#    return r_mag, r_vec

@numba.jit("float64[:,:](float64[:],int64)", nopython=True)
def re_smear(vector, number):
    r = numpy.zeros((number, len(vector)))
    for a in range(number):
        r[a,:] = vector
    return r

@numba.jit("float64[:,:](float64[:,:],int64)", nopython=True)
def ve_smear(vector, number):
    l = len(vector)
    r = numpy.zeros((l, number))
    for a in range(l):
        for b in range(number):
            r[a,b] = vector[a,0]
    return r

# this function doesn't do what I want!
@numba.jit("float64[:](float64[:,:])", nopython=True)
def lin_sum(vvect):
    l = len(vvect[0])
    r = numpy.zeros(l)
    for a in range(l):
        for b in range(3):
            r[a] += vvect[b,a]
    return r

@numba.jit("float64[:,:](float64[:,:])", nopython=True)
def lin_sum_vert(vvect):
    l = len(vvect[0])
    r = numpy.zeros((l,1))
    for a in range(l):
        for b in range(3):
            r[a,0] += vvect[b,a]
    return r

@numba.jit("float64[:,:](float64[:],float64[:,:],float64)",nopython=True)
def partial_step_v(position, other_positions, dT):
    position_ex = re_smear(position, len(other_positions))
    r_vec = position_ex + (dT *other_positions)
    return r_vec

@numba.jit("Tuple((float64[:,:], float64[:,:]))(float64[:],float64[:,:])",nopython=True)
def radius_v(position, other_positions):
    position_ex = re_smear(position, len(other_positions))
    r_vec = (position_ex - other_positions) * -1
    r = lin_sum_vert((r_vec * r_vec).transpose())** 0.5
    return r, r_vec

@numba.jit("float64[:,:](float64[:,:],int64)", nopython=True)
def remove_row(state, index):
    rows = state.shape[0]
    columns = state.shape[1]

    r = numpy.zeros((rows - 1, columns))

    row = 0
    for a in range(rows - 1):
        if row == index:
            row += 1
        for b in range(columns):
            r[a,b] = state[row, b]

        row += 1

    return r


@numba.jit("float64[:](float64[:,:],int64,float64)",nopython=True)
def update_acceleration_euler_v(state, current_index, dT):
    current_body = state[current_index]
    other_states = remove_row(state, current_index)

    r,r_vec = radius_v(current_body[0:3], other_states[:,0:3])
    scalar = ve_smear(G * other_states[:,6:7]/(r**3),3)

    acceleration = lin_sum(scalar * r_vec)
    return acceleration


@numba.jit("float64[:](float64[:,:],int64,float64)",nopython=True)
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
    other_states = remove_row(state, current_index)
    r,r_vec = radius_v(current_body[0:3], other_states[:,0:3])
    scalar = ve_smear(G * other_states[:,6:7]/(r**3),3)

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


    acceleration = lin_sum((k1 + (k2 * 2) + (k3 * 2) + k4)/6)

    return acceleration

@numba.jit("(float64[:,:],float64)",nopython=True)
def update_velocity_v(states, dT = 1.0):
    for a in range(len(states)):
        acceleration = update_acceleration_rk4_v(states, a, dT)
        #acceleration = update_acceleration_euler_v(states, a, dT)
        states[a,3:6] += acceleration * dT

@numba.jit("(float64[:,:],float64)",nopython=True)
def update_position_v(states, dT = 1.0):
    for a in range(len(states)):
        states[a,0:3] += states[a,3:6] * dT

@numba.jit("(float64[:,:],float64)",nopython=True)
def update_states_v(states, dT=1.0):
    update_velocity_v(states, dT=dT)
    update_position_v(states, dT=dT)

def simulate(bodies, dT = 1000.0, T = 10000, output_freq = 100):
    trajectory = []

    labels,state = convert_bodies(bodies)

    for current_body in bodies:
        trajectory.append({"x":[],"y":[],"z":[],"name":current_body.name})

    for i in range(1, int(T)):
        update_states_v(state, dT=dT)

        if i % output_freq == 0:
            for index, position in enumerate(trajectory):

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

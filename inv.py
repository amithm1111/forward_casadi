from casadi import*
from urdf2casadi import urdfparser as u2c
urdf_path = "/home/msi/Documents/UT/forward_kinematics/2link_robot.urdf"
root_link = "base"
end_link = "endEffector"
robot_parser = u2c.URDFparser()
robot_parser.from_file(urdf_path)
# Also supports .from_server for ros parameter server, or .from_string if you have the URDF as a string.
fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
# print(fk_dict.keys())
# should give ['q', 'upper', 'lower', 'dual_quaternion_fk', 'joint_names', 'T_fk', 'joint_list', 'quaternion_fk']
forward_kinematics = fk_dict["T_fk"]
# print(forward_kinematics([0.2, 0.5]))
q = fk_dict["q"]
# print("Number of joints:", q.size()[0])
j_name = fk_dict["joint_names"]
# print(j_name)
#print(forward_kinematics)

import math
import time
import scipy.io
import numpy as np
from casadi.tools import *
import numpy.matlib

N = 3
xd = 1.27893  # goal states
yd = 1.45585

states = struct_symSX(["x","y"]) 
n_states = states.size # Number of states
x,y = states[...]

controls = struct_symSX(["u1","u2"]) # control vector of the system
n_controls = controls.size
u1,u2 = controls[...]

U = SX.sym("U",n_controls,N) # Decision variables (controls)
P = SX.sym("P",n_states) # parameters which include the initial state
X = SX.sym("X",n_states,(N+1)) # A Matrix that represents the states over the optimization problem.

obj = 0 # Objective function
g = []  # constraints vector
st = X[:,0] # initial state
g.append(st - P[0:n_states]) # initial condition constraints
# compute solution symbolically
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    obj = obj+((st[0]-xd)**2 + (st[1]-yd)**2) # calculate obj
    st_next = X[:,k+1]
    f_value  = forward_kinematics(con)
    st_next_c = f_value[0:2,3]
    g.append(st_next - st_next_c) # compute constraints

# make the decision variables one column vector
OPT_variables = vertcat(reshape(X,n_states*(N+1),1),reshape(U,n_controls*N,1))
g = vertcat(*g)
nlp_prob = {'x': OPT_variables,'f': obj,'g': g, 'p': P}
opts = {"print_time": False,"ipopt.print_level":0,"ipopt.max_iter":150000,"ipopt.acceptable_tol":1e-16,"ipopt.acceptable_obj_change_tol":1e-12,"ipopt.warm_start_init_point":"yes"}

solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# Equality Constraints (Multiple Shooting)
lbg = np.zeros(n_states*(N+1))
ubg = np.zeros(n_states*(N+1))
# State Constraints
lbx = np.zeros((n_states*(N+1)+ n_controls*N,1))
ubx = np.zeros((n_states*(N+1)+ n_controls*N,1))
# Create the indixes list for states and controls
xIndex = np.arange(0, n_states*(N+1), n_states).tolist()
yIndex = np.arange(1, n_states*(N+1), n_states).tolist()

u1Index = np.arange(n_states*(N+1),n_states*(N+1)+ n_controls*N, n_controls).tolist()
u2Index = np.arange(n_states*(N+1)+1,n_states*(N+1)+ n_controls*N, n_controls).tolist()
# Feed Bounds For State Constraints
lbx[xIndex,:] = -2
lbx[yIndex,:] = -2

ubx[xIndex,:] = 2
ubx[yIndex,:] = 2
# Feed Bounds For control Constraints
u1_max = 2*pi
u1_min = -2*pi
u2_max = 2*pi
u2_min = -2*pi

lbx[u1Index,:] = u1_min
lbx[u2Index,:] = u2_min

ubx[u1Index,:] = u1_max
ubx[u2Index,:] = u2_max

x_i = 0.
y_i = 0.

x0 = np.array([x_i,y_i])    # initial condition.
u0 = np.zeros((N,2))  # two control inputs
X0 = np.matlib.repmat(x0, 1, N+1) # initialization of the states decision variables
X0 = X0.T

p = np.copy(x0)
x00 = vertcat(reshape(X0.T,n_states*(N+1),1), reshape(u0.T,n_controls*N,1))
sol = solver(x0= x00, lbx= lbx, ubx= ubx,lbg= lbg, ubg= ubg,p=p)
solution = sol['x'].full()
control = np.copy(solution[n_states*(N+1):])
u = np.copy(reshape(control.T,n_controls,N).T)
print('control',u[0,:])
states = np.copy(solution[0:n_states*(N+1)])
# print('states',states)
# print(forward_kinematics(u[0,:]))
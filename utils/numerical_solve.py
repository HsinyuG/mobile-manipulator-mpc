from sympy import symbols, Eq, solve, sin, cos # not used because closed form solution too difficult to find
import casadi as ca
import numpy as np

target_position = np.array([0.6, 0.1])

# Define the symbolic variables
q1 = ca.MX.sym('q1')
q2 = ca.MX.sym('q2')
q3 = ca.MX.sym('q3')
x = ca.MX.sym('x')
z = ca.MX.sym('z')
a2 = 0.316
a3 = 0.0825
a5 = 0.384
a6 = 0.088
a7 = 0.107

x = a2*ca.sin(q1) - a3*ca.sin(q1)*ca.sin(q2) - a3*ca.cos(q1)*ca.cos(q2) + a3*ca.cos(q1) + a5*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1)) - a6*(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3) - a7*((-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3))
z = a2*ca.cos(q1) + a3*ca.sin(q1)*ca.cos(q2) - a3*ca.sin(q1) - a3*ca.sin(q2)*ca.cos(q1) + a5*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) + a6*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) - a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3) - a7*((ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3))

cost = (x - target_position[0])**2 + (z - target_position[1])**2

ineq_constraints = ca.vertcat(q1, q2, q3)
opts_setting = {
        'ipopt.max_iter': 2000,
        'ipopt.print_level': 0, # 5,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6
}
nlp = {'x': ca.vertcat(q1, q2, q3), 'f': cost, 'g': ineq_constraints} # all values in dict must be ca variables, not numpy or list
solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)

result = solver(x0=[0.0, 0.0, 0.0], lbg=[-ca.pi/2, -ca.pi*3/4, 0], ubg=[ca.pi/2, 0, ca.pi*3/2])
print((result['x']))





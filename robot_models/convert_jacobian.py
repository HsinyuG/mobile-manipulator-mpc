import sympy as sp

def classical_dh_to_matrix(theta, d, a, alpha):
    """
    Convert DH parameters to a transformation matrix.
    """
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def modified_dh_to_matrix(theta, d, a, alpha):
    """
    Convert DH parameters to a transformation matrix.
    """
    return None

def jacobian_from_classical_dh(dh_table, joint_angles):
    """
    Compute the Jacobian matrix for a robotic manipulator given its DH table and joint angles.
    """
    num_joints = len(joint_angles)
    jacobian = sp.zeros(6, num_joints)

    T = sp.eye(4)
    zs = []
    ts = []
    zs.append(T[:3, 2])
    ts.append(T[:3, 3])
    print(0,", t: ", T[:3, 3], end=" ")
    print("z: ", T[:3, 2])
    print(T)
    print()

    for i, (theta, d, a, alpha) in enumerate(dh_table):
        T_i = classical_dh_to_matrix(theta, d, a, alpha)
        T *= T_i

        # Linear velocity component
        ts.append(T[:3, 3]) #.jacobian(joint_angles)
        # print(i+1,", t: ", T[:3, 3], end=" ")

        # Angular velocity component
        zs.append(T[:3, 2]) # .jacobian(joint_angles)
        # print("z: ", T[:3, 2])

        # print(T_i)
        # print(T)
        # print()

    for i in range(num_joints):
        jacobian[:3, i] = zs[i].cross((ts[-1] - ts[i]))
        jacobian[3:, i] = zs[i]
        # print(i)
        # print("zi", zs[i])
        # print("ti", ts[i])
        # print("t3", ts[-1])
        # print()

    return jacobian

# Define symbolic variables
# theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
# d1, d2, d3 = sp.symbols('d1 d2 d3')
# a1, a2, a3 = sp.symbols('a1 a2 a3')
# alpha0, alpha1, alpha2, alpha3 = sp.symbols('alpha0 alpha1 alpha2 alpha3')

# q1, q2, q3 = sp.symbols('q1 q2 q3')
# l1, l2, l3 = sp.symbols('l1 l2 l3')

# # Define the DH table # theta, d, a, alpha
# dh_table_demo = [
#     (q1, l1, 0, sp.pi/2),
#     (q2, 0, l2, 0),
#     (q3, 0, l3, 0)
# ]

q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')
a2, a3, a5, a6, a7 = sp.symbols('a2 a3 a5 a6 a7')

# Define the DH table # theta, d, a, alpha
dh_table_panda = [
    (q1,         0,     0,      -sp.pi/2),
    (q2-sp.pi/2, 0,     a2,     0),
    (sp.pi/2,    0,     a3,     sp.pi),
    (q3,         0,     -a3,    0),
    (sp.pi/2,    0,     a5,     0),
    (q4-sp.pi/2, 0,     a6,     0),
    (-sp.pi/2,   0,     a7,     -sp.pi/2)
]

# Define joint angles
joint_angles = [q1, q2, q3, q4]

# Compute the Jacobian matrix
jacobian_matrix = jacobian_from_classical_dh(dh_table_panda, joint_angles)

# Display the Jacobian matrix
print("Jacobian Matrix:")
print(jacobian_matrix)
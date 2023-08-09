import casadi as cs
from urdf2casadi import urdfparser as u2c
urdf_path = "/home/msi/Documents/UT/forward_kinematics/2link_robot.urdf"
root_link = "base"
end_link = "endEffector"
robot_parser = u2c.URDFparser()
robot_parser.from_file(urdf_path)
# Also supports .from_server for ros parameter server, or .from_string if you have the URDF as a string.
fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
print(fk_dict.keys())
# should give ['q', 'upper', 'lower', 'dual_quaternion_fk', 'joint_names', 'T_fk', 'joint_list', 'quaternion_fk']
forward_kinematics = fk_dict["T_fk"]
print(forward_kinematics([0.6, 0.5]))
q = fk_dict["q"]
print("Number of joints:", q.size()[0])
j_name = fk_dict["joint_names"]
print(j_name)
print(forward_kinematics)


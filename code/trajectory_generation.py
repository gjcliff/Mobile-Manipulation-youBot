"""
Author: Graham Clifford

Drive a Kuka robot around to pick up and move a block.

Plan a trajectory for the end-effector of the youBot mobile
manipulator (a mobile base with four mecanum wheels and a 5R
robot arm), perform odometry as the chassis moves, and perform
feedback control to drive the youBot to pick up a block at a
specified location. Then, carry the block to a desired location
and put it down.

"""

import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=10)


class FinalProject():
    def __init__(self):

        self.chassis_phi = 0
        self.chassis_x = 0
        self.chassis_y = 0

        self.J1 = 0
        self.J2 = 0
        self.J3 = 0
        self.J4 = 0
        self.J5 = 0

        self.W1 = 0
        self.W2 = 0
        self.W3 = 0
        self.W4 = 0

        self.gripper_state = 0

        self.gamma1 = np.pi/4
        self.gamma2 = np.pi/4
        self.gamma3 = -np.pi/4
        self.gamma4 = -np.pi/4

        self.l = 0.47/2
        self.w = 0.3/2
        self.r = 0.0475

        self.H0 = 1/self.r * np.array([[-self.l - self.w, 1, -1],
                                       [self.l + self.w, 1, 1],
                                       [self.l + self.w, 1, -1],
                                       [-self.l - self.w, 1, 1]])

        self.F = np.linalg.pinv(self.H0)
        zeros_row = np.array([0, 0, 0, 0])
        self.F6 = np.vstack((zeros_row, zeros_row, self.F, zeros_row))

        self.Tsb = np.array([[np.cos(self.chassis_phi), -np.sin(self.chassis_phi), 0, self.chassis_x],
                             [np.sin(self.chassis_phi), np.cos(
                                 self.chassis_phi), 0, self.chassis_y],
                             [0, 0, 1, 0.0963],
                             [0, 0, 0, 1]])

        self.Tb0 = np.array([[1, 0, 0, 0.1662],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.0026],
                             [0, 0, 0, 1]])

        self.M0e = np.array([[1, 0, 0, 0.033],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.6546],
                             [0, 0, 0, 1]])

        self.orientation90 = np.array([[0, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [-1, 0, 0, 0],
                                       [0, 0, 0, 1]])

        self.orientation135 = np.array([[-0.70710678, 0, 0.70710678, 0],
                                        [0, 1, 0, 0],
                                        [-0.70710678, 0, -0.70710678, 0],
                                        [0, 0, 0, 1]])

        self.TbeInitial = self.Tb0 @ self.M0e
        self.TseInitial_ref = self.Tsb @ self.Tb0 @ self.M0e @ self.orientation90
        self.TseInitial = np.array([[0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [-1, 0, 0, 0.5],
                                    [0, 0, 0, 1]])

        self.w1 = np.array([0, 0, 1])
        self.v1 = np.array([0, 0.033, 0])
        self.B1 = np.array([0, 0, 1, 0, 0.033, 0])

        self.w2 = np.array([0, -1, 0])
        self.v2 = np.array([-0.5076, 0, 0])
        self.B2 = np.array([0, -1, 0, -0.5076, 0, 0])

        self.w3 = np.array([0, -1, 0])
        self.v3 = np.array([-0.3526, 0, 0])
        self.B3 = np.array([0, -1, 0, -0.3526, 0, 0])

        self.w4 = np.array([0, -1, 0])
        self.v4 = np.array([-0.02176, 0, 0])
        self.B4 = np.array([0, -1, 0, -0.2176, 0, 0])

        self.w5 = np.array([0, 0, 1])
        self.v5 = np.array([0, 0, 0])
        self.B5 = np.array([0, 0, 1, 0, 0, 0])

        self.Blist = np.vstack((self.B1, self.B2, self.B3, self.B4, self.B5))

        self.TscInitial = np.array([[1, 0, 0, 1.0],
                                    [0, 1, 0, 0.0],
                                    [0, 0, 1, 0.025],
                                    [0, 0, 0, 1]])

        self.TscGoal = np.array([[0, 1, 0, 0],
                                 [-1, 0, 0, -1.0],
                                 [0, 0, 1, 0.025],
                                 [0, 0, 0, 1]])

        self.TceStandoff = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0.05],
                                     [0, 0, 0, 1]])

        self.TceGrasp = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        self.max_linear_velocity = 0.5  # m/s
        self.max_angular_velocity = 20.0  # rad/s

        self.start_configuration = np.array([0.707, 0.2, self.chassis_y, self.J1, self.J2, self.J3,
                                             self.J4, self.J5, self.W1, self.W2, self.W3, self.W4, 0])

    def TrajectoryToOutput(self, trajectory, gripper_state):
        output = None
        for step in trajectory:
            if output is None:
                output = np.array([[step[0][0], step[0][1], step[0][2],
                                    step[1][0], step[1][1], step[1][2],
                                    step[2][0], step[2][1], step[2][2],
                                    step[0][3], step[1][3], step[2][3], gripper_state]])
            else:
                output = np.vstack((output, np.array([[step[0][0], step[0][1], step[0][2],
                                                       step[1][0], step[1][1], step[1][2],
                                                       step[2][0], step[2][1], step[2][2],
                                                       step[0][3], step[1][3], step[2][3], gripper_state]])))

        return output

    def TrajectoryGeneration(self, TseInitial, TscInitial, TscFinal, TceGrasp, TceStandoff, k):
        '''
        Generates trajectories for the end-effector.

        Generates trajectories for the end-effector to allow it to travel to a cube in space.
        The duration of a trajectory segment is equal to either the distance the origin of the 
        end-effector frame {e} has to travel divided by the maximum linear velocity
        of the end-effector, or the angle the {e} frame must rotate divided by the maximum angular
        velocity of the end-effector, whichever is greater.

        Args:
        ----
        TseInitial: The initial configuration of the end-effector in the reference trajectory.
        TscInitial: The cube's initial configuration.
        TscFinal: The cube's desired final configuration.
        TceGrasp: The end-effector's configuration relative to the cube when it is grasping the cube.
        TceStandoff: The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube.
        k: The number of trajectory reference configurations per 0.01 seconds.

        Returns:
        -------
        output_trajectories: An length N list of trajectories
        '''

        print("Generating Reference Trajectories.")

        Tf = 10 - 0.625 * 2

        # trajectory 1
        T1Initial = TseInitial
        T1Final = TscInitial @ TceStandoff @ self.orientation135
        traj1 = mr.ScrewTrajectory(
            T1Initial, T1Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj1Output = self.TrajectoryToOutput(traj1, 0)

        # trajectory 2
        T2Initial = T1Final
        T2Final = TscInitial @ TceGrasp @ self.orientation135
        traj2 = mr.ScrewTrajectory(
            T2Initial, T2Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj2Output = self.TrajectoryToOutput(traj2, 0)

        # trajectory 3
        T3Initial = T2Final
        T3Final = T3Initial
        traj3 = mr.ScrewTrajectory(
            T3Initial, T3Final, 0.625, 0.625/(0.01/k), 5)
        traj3Output = self.TrajectoryToOutput(traj3, 1)

        # trajectory 4
        T4Initial = T3Final
        T4Final = TscInitial @ TceStandoff @ self.orientation135
        traj4 = mr.ScrewTrajectory(
            T4Initial, T4Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj4Output = self.TrajectoryToOutput(traj4, 1)

        # trajectory 5
        T5Initial = T4Final
        T5Final = TscFinal @ TceStandoff @ self.orientation135
        traj5 = mr.ScrewTrajectory(
            T5Initial, T5Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj5Output = self.TrajectoryToOutput(traj5, 1)

        # trajectory 6
        T6Initial = T5Final
        T6Final = TscFinal @ TceGrasp @ self.orientation135
        traj6 = mr.ScrewTrajectory(
            T6Initial, T6Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj6Output = self.TrajectoryToOutput(traj6, 1)

        # trajectory 7
        T7Initial = T6Final
        T7Final = T7Initial
        traj7 = mr.ScrewTrajectory(
            T7Initial, T7Final, 0.625, 0.625/(0.01/k), 5)
        traj7Output = self.TrajectoryToOutput(traj7, 0)

        # trajectory 8
        T8Initial = T7Final
        T8Final = TscFinal @ TceStandoff @ self.orientation135
        traj8 = mr.ScrewTrajectory(
            T8Initial, T8Final, Tf/6, (Tf/6)/(0.01/k), 5)
        traj8Output = self.TrajectoryToOutput(traj8, 0)

        output_configurations = np.vstack((traj1Output, traj2Output, traj3Output,
                                           traj4Output, traj5Output, traj6Output,
                                           traj7Output, traj8Output))

        return output_configurations

    def NextState(self, current_configuration, speed_controls, dt, max_angular_velocity):
        '''
        Take an initial robot configuration and simulate the next step in its trajectory.

        Take an initial robot configuration of 12 variables. 3 variables (phi, x, y) for
        the chassis, 5 variables (J1, J2, J3, J4, J5) for the positions of the 5 DOF robot
        arm, and 4 variables (W1, W2, W3, W4) for the positions of the 4 robot wheels. Use
        the initial robot configuration to simulate the next step in the robot's trajcetory.

        Args:
        ----
        current_configuration (1x12 numpy array): The current configuration of the robot
        speed_controls (1x9 numpy array): The current angular velocities of the wheels and\
        robot joints.
        dt (float): The time step from the current configuration to the next configuration.
        max_angular_velocity (float): The max angular velocity of either the wheels or the\
        robot joints.

        Returns:
        -------
        new_configuration (1x12 numpy array): The new robot configuration after dt has passed.

        '''
        # assume current_configuration is 1. q(phi, x, y) of the base, 2. J1, J2, J3, J4, J5, 3. W1, W2, W3, W4

        for i in range(len(speed_controls)):
            if speed_controls[i] > max_angular_velocity:
                speed_controls[i] = max_angular_velocity
            elif speed_controls[i] < -max_angular_velocity:
                speed_controls[i] = -max_angular_velocity

        old_arm_joint_angles = current_configuration[3:8]
        old_wheel_angles = current_configuration[8:12]

        arm_speeds = speed_controls[4:]
        wheel_speeds = speed_controls[:4]

        new_arm_joint_angles = old_arm_joint_angles + arm_speeds * dt
        new_wheel_angles = old_wheel_angles + wheel_speeds * dt

        delta_theta = wheel_speeds * dt
        Vb = np.linalg.pinv(self.H0) @ delta_theta
        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]

        if wbz == 0:
            delta_qb = np.array([0, vbx, vby])
        else:
            delta_qb = np.array([wbz,
                                (vbx * np.sin(wbz) + vby * (np.cos(wbz) - 1))/wbz,
                                (vby * np.sin(wbz) + vbx * (1 - np.cos(wbz)))/wbz])

        phi = current_configuration[0]

        delta_q = np.array([[1, 0, 0],
                            [0, np.cos(phi), -1 * np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)]]) @ delta_qb

        q_new = current_configuration[:3] + delta_q

        new_configuration = np.hstack(
            (q_new, new_arm_joint_angles, new_wheel_angles))

        return new_configuration

    def FeedbackControl(self, X, Xd, Xdnext, Kp, Ki, dt, integral_error):
        """
        Calculate the kinematic task-space feedforward plus feedback control law.

        Calculate the kinematic task-space feedforward plus feedback control law
        iteratively as represented in equation 11.16 in the Modern Robotics Task space.

        Args:
        ----
        X (4x4 np array): The current actual end-effector configuration.
        Xd (4x4 np array): The current end-effector reference configuration.
        Xdnext (4x4 np array): The current end-effector reference configuration future time dt.
        Kp (4x4 np array): The proportional gain.
        Ki (4x4 np array): The integral gain.
        dt (float): The timestep.
        integral error (4x4 np array): The cumulative integral error

        Returns:
        -------

        """
        Vd_se3 = (1/dt) * mr.MatrixLog6(mr.TransInv(Xd) @ Xdnext)
        Vd = mr.se3ToVec(Vd_se3)

        xerr_se3 = mr.MatrixLog6(mr.TransInv(X) @ Xd)
        xerr = mr.se3ToVec(xerr_se3)

        XinvXd = mr.TransInv(X) @ Xd
        AdjXinvXd = mr.Adjoint(XinvXd)

        integral_error += xerr * dt

        V = AdjXinvXd @ Vd + Kp @ xerr + Ki @ integral_error

        return V, integral_error, xerr

    def rowToTrans(self, row):
        trans = np.array([[row[0], row[1], row[2], row[9]],
                          [row[3], row[4], row[5], row[10]],
                          [row[6], row[7], row[8], row[11]],
                          [0, 0, 0, 1]])
        return trans

    def run(self):
        """Find the robot trajectory necessary to pick up the block and move it."""
        Kp = 3.0 * np.identity(6)
        Ki = 0.1 * np.identity(6)

        dt = 0.01
        integral_error = np.array([0., 0., 0., 0., 0., 0.])

        # calculate reference trajectories
        reference_trajectories = self.TrajectoryGeneration(
            self.TseInitial_ref, self.TscInitial, self.TscGoal, self.TceGrasp, self.TceStandoff, 1)

        current_configuration = self.start_configuration

        actual_configuration = [current_configuration]
        xerr_list = []

        print("Generating trajectories using feedforward plus feedback control")

        for i in range(len(reference_trajectories)-1):

            # find the current joint angles of the arm
            thetalist_arm = current_configuration[3:8]

            x = current_configuration[1]
            y = current_configuration[2]
            phi = current_configuration[0]

            Tsb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                            [np.sin(phi), np.cos(phi), 0, y],
                            [0, 0, 1, 0.0963],
                            [0, 0, 0, 1]])

            T0e = mr.FKinBody(self.M0e, self.Blist.T, thetalist_arm)

            Tse_current = Tsb @ self.Tb0 @ T0e

            # find transformation matrix T0e and the adjoint

            V, integral_error, xerr = self.FeedbackControl(
                Tse_current, self.rowToTrans(reference_trajectories[i]), self.rowToTrans(reference_trajectories[i+1]), Kp, Ki, dt, integral_error)

            AdjT0einvTb0inv = mr.Adjoint(
                np.linalg.inv(T0e) @ np.linalg.inv(self.Tb0))

            # find the base and arm jacobians
            Jbase = AdjT0einvTb0inv @ self.F6
            Jarm = mr.JacobianBody(self.Blist.T, thetalist_arm)

            J = np.concatenate((Jbase, Jarm), axis=1)
            speed_controls = np.linalg.pinv(J) @ V

            current_configuration = self.NextState(
                current_configuration[:12], speed_controls, dt, self.max_angular_velocity)

            current_configuration = np.append(
                current_configuration, reference_trajectories[i][12])

            actual_configuration.append(current_configuration)
            xerr_list.append(xerr)

        print("Generating csv data...")
        # np.savetxt("newTask.csv", actual_configuration, delimiter=',')
        np.savetxt("trajectory.csv",
                   xerr_list, delimiter=',')

        xerr_array = np.asarray(xerr_list).T

        plt.figure(dpi=125, facecolor='w')

        x_min = 0
        x_max = 10
        N = 993
        x_domain = np.linspace(x_min, x_max, N)

        plt.figure(figsize=(12, 4))

        plt.plot(x_domain, xerr_array[0])
        plt.plot(x_domain, xerr_array[1])
        plt.plot(x_domain, xerr_array[2])
        plt.plot(x_domain, xerr_array[3])
        plt.plot(x_domain, xerr_array[4])
        plt.plot(x_domain, xerr_array[5])
        plt.title(
            "$\\omega_x$, $\\omega_x$, $\\omega_x$, and $v_x$, $v_y$, and $v_z$")
        plt.legend([r'$\omega_x$', r'$\omega_y$',
                   r'$\omega_z$', r'$v_x$', r'$v_y$', r'$v_z$'])
        plt.xlabel("Time")

        plt.show()


def main():
    final = FinalProject()
    final.run()


if __name__ == "__main__":
    main()

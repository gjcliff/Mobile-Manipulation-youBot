import modern_robotics as mr
import numpy as np

# to run: python3 trajectory_generation.py


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
        self.TseInitial = self.Tsb @ self.Tb0 @ self.M0e @ self.orientation90

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
        self.B4 = np.array([0, -1, 0, -0.02176, 0, 0])

        self.w5 = np.array([0, 0, 1])
        self.v5 = np.array([0, 0, 0])
        self.B5 = np.array([0, 0, 1, 0, 0, 0])

        self.Blist = np.array([self.B1, self.B2, self.B3, self.B4, self.B5])

        self.TscInitial = np.array([[1, 0, 0, 1],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0.025],
                                    [0, 0, 0, 1]])

        self.TscGoal = np.array([[0, 1, 0, 0],
                                 [-1, 0, 0, -1],
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

        '''

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

        traj = np.vstack((traj1Output, traj2Output, traj3Output,
                          traj4Output, traj5Output, traj6Output,
                          traj7Output, traj8Output))

        # np.savetxt("milestone2.csv", traj, delimiter=',')

    def NextState(self, current_configuration, speed_controls, dt, max_angular_velocity):
        '''

        '''
        # assume current_configuration is 1. q(phi, x, y) of the base, 2. J1, J2, J3, J4, J5, 3. W1, W2, W3, W4

        for elem in speed_controls:
            if elem > max_angular_velocity:
                elem = max_angular_velocity

        old_arm_joint_angles = current_configuration[3:8]
        old_wheel_angles = current_configuration[8:]

        new_arm_joint_angles = old_arm_joint_angles + speed_controls[4:] * dt
        new_wheel_angles = old_wheel_angles + speed_controls[:4] * dt

        # print(speed_controls[:4])

        delta_theta = (new_wheel_angles - old_wheel_angles)
        Vb = np.linalg.pinv(self.H0) @ delta_theta
        wbz = Vb[0]
        vbx = Vb[1]
        vby = Vb[2]

        # print(f"wbz: {wbz}, vbx: {vbx}, vby: {vby}")

        if wbz == 0:
            delta_q = np.array([0, vbx, vby])
        else:
            delta_q = np.array(
                [wbz, (vbx * np.sin(wbz) + vby * (np.cos(wbz) - 1))/wbz, (vby * np.sin(wbz) + vbx * (1 - np.cos(wbz)))/wbz])

        q_new = current_configuration[:3] + delta_q

        new_configuration = np.hstack(
            (q_new, new_arm_joint_angles, new_wheel_angles))

        print(new_configuration)

        # print(f"new_configuration: {new_configuration}")

        return new_configuration

    def TestNextState(self, u, v, dt, N):
        start_configuration = np.array([self.chassis_phi, self.chassis_x, self.chassis_y, 1, self.J2, self.J3,
                                        self.J4, self.J5, 1, self.W2, self.W3, self.W4])

        speed_controls = np.hstack((u, v))
        # print(f"speed_Controls: {speed_controls}")

        max_angular_velocity = 1

        current_configuration = start_configuration

        configuration_list = [np.append(current_configuration, 0)]

        for i in range(N):
            current_configuration = self.NextState(current_configuration, speed_controls,
                                                   dt, max_angular_velocity)

            # print(current_configuration)
            configuration_list.append(np.append(current_configuration, 0))

        np.savetxt("NextStateTest.csv", configuration_list, delimiter=',')

    def run(self):
        v = np.array([0, 0, 0, 0, 0])
        u = np.array([10, 10, 10, 10])
        dt = 0.01
        N = 100

        self.TrajectoryGeneration(
            self.TseInitial, self.TscInitial, self.TscGoal, self.TceGrasp, self.TceStandoff, 1)
        self.TestNextState(u, v, dt, N)


def main():
    final = FinalProject()
    final.run()


if __name__ == "__main__":
    main()

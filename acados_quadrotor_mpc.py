from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSimSolver, AcadosSim
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from itertools import product


class QuadrotorMPC:
    def __init__(self, params):
        self.model_name = 'quadrotor_mpc'
        self.mass = params['mass']
        self.arm_length = params['arm_length']
        self.k = params['motor_torque_coefficient']
        self.gravity = 9.80665
        self.Jxx = params['Jxx']
        self.Jyy = params['Jyy']
        self.Jzz = params['Jzz']

    def solve_mpc(self, num_steps, use_cython):
        px = SX.sym('px')
        py = SX.sym('py')
        pz = SX.sym('pz')
        vx = SX.sym('vx')
        vy = SX.sym('vy')
        vz = SX.sym('vz')
        qw = SX.sym('qw')
        qx = SX.sym('qx')
        qy = SX.sym('qy')
        qz = SX.sym('qz')
        wx = SX.sym('wx')
        wy = SX.sym('wy')
        wz = SX.sym('wz')
        f1 = SX.sym('f1')
        f2 = SX.sym('f2')
        f3 = SX.sym('f3')
        f4 = SX.sym('f4')
        dt = SX.sym('dt')

        x = vertcat(px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz)
        u = vertcat(f1,f2,f3,f4,dt)
        R = vertcat(
            horzcat(qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
            horzcat(2*(qx*qy+qw*qz), qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz-qw*qx)),
            horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw**2 - qx**2 - qy**2 + qz**2)
        )
        omega_skew = vertcat(
            horzcat(0, -wx, -wy, -wz),
            horzcat(wx, 0, wz, -wy),
            horzcat(wy, -wz, 0, wx),
            horzcat(wz, wy, -wx, 0)
        )
        q = vertcat(qw,qx,qy,qz)
        qdot = 0.5 * omega_skew @ q
        f_total = vertcat(0,0,f1 + f2 + f3 + f4)
        vdot = -self.gravity*vertcat(0,0,1) + R @ f_total / self.mass

        J = diag(horzcat(self.Jxx,self.Jyy,self.Jzz))
        w = vertcat(wx,wy,wz)
        Mx = sqrt(2) / 2 * self.arm_length * (f1 - f2 - f3 + f4)
        My = sqrt(2) / 2 * self.arm_length * (- f1 - f2 + f3 + f4)
        Mz = self.k * (f1 - f2 + f3 - f4)
        wdot = inv_minor(J) @ (vertcat(Mx,My,Mz) - cross(w, J @ w))

        # Setup explicit ode equations
        pxdot = vx
        pydot = vy
        pzdot = vz
        vxdot = vdot[0]
        vydot = vdot[1]
        vzdot = vdot[2]
        qwdot = qdot[0]
        qxdot = qdot[1]
        qydot = qdot[2]
        qzdot = qdot[3]
        wxdot = wdot[0]
        wydot = wdot[1]
        wzdot = wdot[2]

        # Time-scaled dynamics for time optimal Control
        f_expl = dt*vertcat(pxdot,pydot,pzdot,vxdot,vydot,vzdot,qwdot,qxdot,qydot,qzdot,wxdot,wydot,wzdot)

        model = AcadosModel()

        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name

        ocp = AcadosOcp()

        ocp.model = model

        N = num_steps
        Tf = N
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = N

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        ocp.model.cost_expr_ext_cost = dt
        ocp.model.cost_expr_ext_cost_e = 0

        f_max = 10.0
        dt_max = 2.0
        ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0, 0.000001])
        ocp.constraints.ubu = np.array([f_max, f_max, f_max, f_max, dt_max])
        ocp.constraints.idxbu = np.array([0,1,2,3,4])

        x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
        xf = np.array([3.0,1.0,3.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])

        ocp.constraints.x0 = x0
        ocp.constraints.lbx_e = xf
        ocp.constraints.ubx_e = xf
        ocp.constraints.idxbx_e = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

        ocp.solver_options.tf = Tf

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        # ocp.solver_options.exact_hess_constr = 0
        # ocp.solver_options.exact_hess_dyn = 0
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 2
        # ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        # ocp_solver = AcadosOcpSolver(ocp, build=False, generate=False)

        ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

        ocp_solver.reset()

        # ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
        # AcadosOcpSolver.generate(ocp, json_file='acados_ocp_nlp.json')
        # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        # ocp_solver = AcadosOcpSolver.create_cython_solver('acados_ocp_nlp.json')

        # for i, tau in enumerate(np.linspace(0, 1, N)):
        #     ocp_solver.set(i, 'x', (1-tau)*x0 + tau*xf)
        #     ocp_solver.set(i, 'u', np.array([2.0, 2.0, 2.0, 2.0, 0.1]))

        
        x_sim = np.zeros((N+1, nx))
        u_sim = np.zeros((N, nu))

        status = ocp_solver.solve()


        # if status != 0:
        #     ocp_solver.print_statistics()
        #     raise Exception(f'acados returned status {status}.')
        # if status != 0:
        #     ocp_solver.print_statistics() 
        #     raise Exception('acados returned status {}. Exiting.'.format(status))
        
        for i in range(N):
            x_sim[i,:] = ocp_solver.get(i, "x")
            u_sim[i,:] = ocp_solver.get(i, "u")
        x_sim[N,:] = ocp_solver.get(N, "x")

        dts = u_sim[:, 1]
                
        return status, x_sim, u_sim
    
    def plot_quadrotor_trajectory(self, x_sim, u_sim):
        ax = plt.figure().add_subplot(projection='3d')        
        ax.plot(x_sim[:,0], x_sim[:,1], x_sim[:,2])
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.set_title(sum(u_sim[:,4]))
        plt.show()


def main():
    params = {
        'mass': 1.0,
        'arm_length': 0.1,
        'Jxx': 0.1,
        'Jyy': 0.1,
        'Jzz': 0.1,
        'motor_torque_coefficient': 1e-3
    }
    quad3Dmpc= QuadrotorMPC(params)
    for use_cython in [True, False]:
        status, x_sim, u_sim = quad3Dmpc.solve_mpc(25, use_cython)
    quad3Dmpc.plot_quadrotor_trajectory(x_sim, u_sim)

if __name__ == '__main__':
        main()

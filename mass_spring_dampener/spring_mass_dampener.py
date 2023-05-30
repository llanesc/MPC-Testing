from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from itertools import product

def export_linear_spring_mass_dampener_model():
    model_name = 'linear_spring_mass_dampener'
    px = SX.sym('px')
    vx = SX.sym('vx')
    u = SX.sym('u')

    m = 1.0
    k = 5.0
    b = 3.0
    x = vertcat(px,vx)

    pxdot = vx
    vxdot = -k / m * px * px - b / m * vx + u / m
    f_expl = vertcat(pxdot, vxdot)

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.p = []
    model.name = model_name

    return model

def solve_ocp(x0,xr):
    ocp = AcadosOcp()

    model = export_linear_spring_mass_dampener_model()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx

    Tf = 5
    N = 500
    t_sim = np.linspace(0, Tf, N+1)

    ocp.dims.N = N

    Q = np.diag([10, 1])
    R = np.diag([])

    ocp.cost.W = Q
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.Vx = np.eye((nx))
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.yref = xr

    ocp.cost.W_e = Q
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_e = np.eye((nx))
    ocp.cost.yref_e = xr

    Fmax = 50
    ocp.constraints.lbu = -Fmax * np.ones((nu,))
    ocp.constraints.ubu = +Fmax * np.ones((nu,))
    ocp.constraints.idxbu = np.array(range(nu))
    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    nlp_iter = 50
    ocp.solver_options.nlp_solver_max_iter = nlp_iter    
    ocp.solver_options.print_level = 1
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2
    ocp.solver_options.qp_solver_cond_N = N
    nlp_tol = 1e-5
    ocp.solver_options.qp_tol = nlp_tol
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.tf = Tf


    ocp_solver = AcadosOcpSolver(ocp, json_file=f'{model.name}_ocp.json')

    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    sqp_iter = ocp_solver.get_stats('sqp_iter')[0]
    print(f'acados returned status {status}.')

    simX = np.array([ocp_solver.get(i,"x") for i in range(N+1)])
    simU = np.array([ocp_solver.get(i,"u") for i in range(N)])
    pi_multiplier = [ocp_solver.get(i, "pi") for i in range(N)]

    if status != 0:
        raise Exception('ocp_nlp solver returned status nonzero')
    else:
        print(f'spring_mass_dampener: success')
        plt.figure(figsize=(10,6))
        plt.subplot(3, 1, 1)
        plt.plot(t_sim, simX[:,0])
        plt.ylabel('x [m]')
        plt.xlabel('time [s]')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t_sim, simX[:,1])
        plt.ylabel('v [m/s]')
        plt.xlabel('time [s]')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t_sim[:-1], simU)
        plt.ylabel('u [N]')
        plt.xlabel('time [s]')
        plt.grid(True)

        plt.tight_layout()
        print(f"cost function value = {ocp_solver.get_cost()}")
        plt.show()


def main():
    x0 = np.array([0,0])
    xr = np.array([3,0])
    solve_ocp(x0, xr)

if __name__ == '__main__':
    main()



import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.mpc import DynamicModelInterface
import numpy as np
import matplotlib.pyplot as plt


def solve_mpc(x0, v0, xr, vr, horizon=5.0, dt = 0.01):
    k = 5.0
    b = 3.0
    mass = 1.0
    udot_max = 1000.0
    u_min = -50.0
    u_max = 50.0
    position_cost = 10.0
    velocity_cost = 1.0
    t_sim = np.linspace(0,horizon,round(horizon/dt)+1)
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(0.0, horizon))

    m.x = pyo.Var(m.t)
    m.v = pyo.Var(m.t)
    m.u = pyo.Var(m.t,bounds=(u_min, u_max))

    m.dxdt = dae.DerivativeVar(m.x, wrt=m.t)
    m.dvdt = dae.DerivativeVar(m.v, wrt=m.t)
    m.dudt = dae.DerivativeVar(m.u, wrt=m.t)

    m.dxdt_ode = pyo.Constraint(m.t, rule= lambda m, t: m.dxdt[t] == m.v[t])
    m.dvdt_ode = pyo.Constraint(m.t, rule= lambda m, t: m.dvdt[t] == - k / mass * m.x[t] * m.x[t] - b / mass * m.v[t] + m.u[t] / mass)
    # m.dudt_ub = pyo.Constraint(m.t, rule = lambda m, t: m.dudt[t] <= udot_max)
    # m.dudt_lb = pyo.Constraint(m.t, rule = lambda m, t: -udot_max <= m.dudt[t])

    pyo.TransformationFactory('dae.finite_difference').apply_to(m, nfe=round(horizon/dt), wrt=m.t, scheme='FORWARD')

    mpc_interface = DynamicModelInterface(m, m.t)
    setpoint = {m.x[:]: xr, m.v[:]: vr}
    var_set, tr_cost = mpc_interface.get_penalty_from_target(setpoint)
    m.setpoint_idx = var_set
    m.tracking_cost = tr_cost
    
    m.objective = pyo.Objective(
        expr=
        0.5*(position_cost * sum(
            m.tracking_cost[0, t]
            for t in m.t
        ) + 
        velocity_cost * sum(
            m.tracking_cost[1, t]
            for t in m.t
        ))*dt, sense=pyo.minimize
    )
    t0 = m.t.first()
    m.x[t0].fix(x0)
    m.v[t0].fix(v0)
    # m.u[t0].fix(0.0)
    
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(m)
    pyo.assert_optimal_termination(res)

    cost_sim = pyo.value(m.objective)
    x_sim = np.array([m.x[t]() for t in m.t])
    v_sim = np.array([m.v[t]() for t in m.t])
    u_sim = np.array([m.u[t]() for t in m.t])

    print('cost value: ' + str(cost_sim))
    return m.t, x_sim, v_sim, u_sim


def main():
    t_sim, x_sim, v_sim, u_sim = solve_mpc(0.0, 0.0, 3.0, 0.0, horizon=5.0, dt=0.01)
    plt.figure(figsize=(10,6))
    plt.subplot(3, 1, 1)
    plt.plot(t_sim, x_sim)
    plt.ylabel('x [m]')
    plt.xlabel('time [s]')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_sim, v_sim)
    plt.ylabel('v [m/s]')
    plt.xlabel('time [s]')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_sim, u_sim)
    plt.ylabel('u [N]')
    plt.xlabel('time [s]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
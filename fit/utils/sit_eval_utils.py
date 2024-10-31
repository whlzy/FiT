def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sde-sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--ode-sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

# ode solvers:
# - Adaptive-step:
#   - dopri8 Runge-Kutta 7(8) of Dormand-Prince-Shampine
#   - dopri5 Runge-Kutta 4(5) of Dormand-Prince [default].
#   - bosh3 Runge-Kutta 2(3) of Bogacki-Shampine
#   - adaptive_heun Runge-Kutta 1(2)
# - Fixed-step:
#   - euler Euler method.
#   - midpoint Midpoint method.
#   - rk4 Fourth-order Runge-Kutta with 3/8 rule.
#   - explicit_adams Explicit Adams.
#   - implicit_adams Implicit Adams.
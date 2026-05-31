import ipyopt
import numpy as np


def _as_float_array(values):
    return np.asarray(values, dtype=np.float64)


class _PyIpoptProblem:
    def __init__(
        self,
        nvar,
        x_l,
        x_u,
        ncon,
        g_l,
        g_u,
        eval_f,
        eval_grad_f,
        eval_g,
        eval_jac_g,
    ):
        jac_rows, jac_cols = eval_jac_g(np.zeros(nvar, dtype=np.float64), True)
        self._jac_rows = np.asarray(jac_rows, dtype=np.int32)
        self._jac_cols = np.asarray(jac_cols, dtype=np.int32)

        def grad_wrapper(x, out):
            out[...] = _as_float_array(eval_grad_f(x))
            return out

        def g_wrapper(x, out):
            out[...] = _as_float_array(eval_g(x))
            return out

        def jac_wrapper(x, out):
            jac = _as_float_array(eval_jac_g(x, False))
            out[...] = jac[self._jac_rows, self._jac_cols]
            return out

        empty_h = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        self._problem = ipyopt.Problem(
            nvar,
            _as_float_array(x_l),
            _as_float_array(x_u),
            ncon,
            _as_float_array(g_l),
            _as_float_array(g_u),
            (self._jac_rows, self._jac_cols),
            empty_h,
            eval_f=eval_f,
            eval_grad_f=grad_wrapper,
            eval_g=g_wrapper,
            eval_jac_g=jac_wrapper,
        )

    def int_option(self, name, value):
        self._problem.set(**{name: int(value)})

    def num_option(self, name, value):
        self._problem.set(**{name: float(value)})

    def str_option(self, name, value):
        self._problem.set(**{name: str(value)})

    def solve(self, x0):
        x = _as_float_array(x0).copy()
        x, obj, status = self._problem.solve(x)
        return x, None, None, None, obj, status

    def close(self):
        return None


def create(
    nvar,
    x_l,
    x_u,
    ncon,
    g_l,
    g_u,
    nnzj,
    nnzh,
    eval_f,
    eval_grad_f,
    eval_g,
    eval_jac_g,
):
    return _PyIpoptProblem(
        nvar=nvar,
        x_l=x_l,
        x_u=x_u,
        ncon=ncon,
        g_l=g_l,
        g_u=g_u,
        eval_f=eval_f,
        eval_grad_f=eval_grad_f,
        eval_g=eval_g,
        eval_jac_g=eval_jac_g,
    )

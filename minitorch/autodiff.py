from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    lower = list(vals)
    upper = list(vals)
    lower[arg] -= epsilon / 2
    upper[arg] += epsilon / 2
    return (f(*upper) - f(*lower)) / epsilon
    raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    
    rs = []
    st = set()

    def dfs(cur_v: Variable) -> None:
        for next_u in cur_v.parents:
            if next_u.unique_id in st:
                continue
            st.add(next_u.unique_id)
            dfs(next_u)
        if not cur_v.is_constant():
            rs.append(cur_v)

    dfs(variable)

    return rs[::-1]
    raise NotImplementedError('Need to implement for Task 1.4')


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sort_ls = topological_sort(variable)
    df = {}
    df[variable.unique_id] = deriv
    for cur in sort_ls:
        if cur.is_leaf():
            cur.accumulate_derivative(df[cur.unique_id])
            continue
        for back_var, back_deriv in cur.chain_rule(df[cur.unique_id]):
            if back_var.unique_id not in df:
                df[back_var.unique_id] = 0    
            df[back_var.unique_id] += back_deriv
    return None
    raise NotImplementedError('Need to implement for Task 1.4')


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

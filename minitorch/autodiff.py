from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to the i-th arg in vals.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values from x_0 to x_n-1
        arg : the i-th arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_epsilon: List[Any] = list(vals)
    vals_minus_epsilon: List[Any] = list(vals)

    vals_plus_epsilon[arg] += epsilon
    vals_minus_epsilon[arg] -= epsilon

    return (f(*vals_plus_epsilon) - f(*vals_minus_epsilon)) / (2 * epsilon)


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
    order: List[Variable] = []
    visited = set()

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return

        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    dfs(m)

        visited.add(var.unique_id)
        order.append(var)

    dfs(variable)
    return order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    nodes = topological_sort(variable)

    d = {}
    d[variable.unique_id] = deriv

    for node in nodes:
        if node.is_leaf():
            node.accumulate_derivative(d[node.unique_id])
        else:
            # store the intermediate derivatives in the dict
            for var, der in node.chain_rule(d[node.unique_id]):
                if var.is_constant():
                    continue

                if var.unique_id not in d:
                    # this is the case where var was only used in 1 function. So we have a direct derivative with respect
                    d[var.unique_id] = der
                else:
                    # this is the case where var was used in different functions. We want to accumulate the its derivatives
                    d[var.unique_id] += der


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

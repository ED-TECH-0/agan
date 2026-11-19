"""Think-in-code: safe arithmetic and computation tool for AGAN.

Provides a ``calc`` tool that evaluates mathematical expressions safely,
using an AST-based evaluator for simple arithmetic and a sandboxed Python
runner for more complex expressions with an allowlisted standard-library
subset.

Library allowlist by hardware profile:
  - Always: ``math``, ``fractions``, ``decimal``, ``statistics``
  - M and above: also ``numpy``
  - L: also ``sympy``
"""

from __future__ import annotations

import ast
import math
import operator
from decimal import Decimal
from fractions import Fraction
from typing import Any
from statistics import mean, median, stdev, variance

from agan.sandbox import SandboxRunner
from agan.hardware.detect import HardwareProfile


# -----------------------------------------------------------------------
# AST-based arithmetic evaluator (safe: no eval, no builtins access)
# -----------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.inv,
    ast.Not: operator.not_,
}

_COMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_ALLOWED_NAMES = {
    # math constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # math functions (safe subset)
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "floor": math.floor,
    "ceil": math.ceil,
    "fmod": math.fmod,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "degrees": math.degrees,
    "radians": math.radians,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "expm1": math.expm1,
    "log1p": math.log1p,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
}


def _eval_ast(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool, str)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(operand)
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator)
            cmp_op = _COMPOPS.get(type(op))
            if cmp_op is None:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
            if not cmp_op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Undefined name: {node.id}")
    if isinstance(node, ast.Call):
        func = _eval_ast(node.func)
        if not callable(func):
            raise ValueError(f"Not callable: {node.func}")
        args = [_eval_ast(a) for a in node.args]
        kwargs = {kw.arg: _eval_ast(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)
    if isinstance(node, ast.List):
        return [_eval_ast(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e) for e in node.elts)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def eval_arithmetic(expression: str) -> Any:
    """Evaluate a mathematical expression safely using AST parsing.

    Only arithmetic, comparisons, boolean logic, and allowlisted math
    functions/constants are permitted. No imports, no attribute access,
    no function definitions, no code execution beyond the allowed set.

    Raises ValueError on any disallowed construct.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
    return _eval_ast(tree)


# -----------------------------------------------------------------------
# Library allowlist by hardware profile
# -----------------------------------------------------------------------

_ALWAYS_ALLOWED = {
    "math",
    "fractions",
    "decimal",
    "statistics",
}
_M_AND_ABOVE = _ALWAYS_ALLOWED | {"numpy"}
_L_AND_ABOVE = _M_AND_ABOVE | {"sympy"}

_PROFILE_LIBS = {
    HardwareProfile.S: _ALWAYS_ALLOWED,
    HardwareProfile.M: _M_AND_ABOVE,
    HardwareProfile.L: _L_AND_ABOVE,
}


def allowed_libraries(profile: HardwareProfile) -> set[str]:
    """Return the set of importable library names for the given profile."""
    return _PROFILE_LIBS.get(profile, _ALWAYS_ALLOWED)


# -----------------------------------------------------------------------
# Sandbox-backed calc tool
# -----------------------------------------------------------------------

class CalcTool:
    """``calc`` tool: evaluate math expressions, optionally via sandboxed Python.

    Parameters
    ----------
    profile:
        Hardware profile determining which libraries are allowlisted.
    sandbox:
        Optional SandboxRunner for executing code that needs imports.
    """

    def __init__(self, profile: HardwareProfile, sandbox: SandboxRunner | None = None):
        self.profile = profile
        self.sandbox = sandbox or SandboxRunner()
        self._allowed_libs = allowed_libraries(profile)

    def __call__(self, expression: str) -> dict[str, Any]:
        """Evaluate a math expression.

        Tries the fast AST evaluator first. If the expression contains
        imports or calls outside the allowed set, falls back to sandboxed
        Python execution with an import guard.
        """
        # Fast path: pure AST arithmetic
        try:
            result = eval_arithmetic(expression)
            return {
                "ok": True,
                "result": result,
                "expression": expression,
                "method": "ast",
                "error": None,
            }
        except (ValueError, SyntaxError):
            pass

        # Slow path: sandboxed Python with import guard
        libs = ", ".join(sorted(self._allowed_libs))
        code = (
            "import sys\n"
            "sys.path = [p for p in sys.path if 'site-packages' not in p]\n"
            f"allowed = {{{libs!r}}}\n"
            "__imports__ = __builtins__.__import__\n"
            "def guarded_import(name, *a, **k):\n"
            "    if name.split('.')[0] not in allowed:\n"
            "        raise ImportError(f'{name} not allowlisted')\n"
            "    return __imports__(name, *a, **k)\n"
            "__builtins__.__import__ = guarded_import\n"
            f"result = ({expression})\n"
            "print(repr(result))\n"
        )

        sandbox_result = self.sandbox.run(code, timeout_s=30)

        if not sandbox_result.ok:
            return {
                "ok": False,
                "result": None,
                "expression": expression,
                "method": "sandbox",
                "error": sandbox_result.error or sandbox_result.stderr or "unknown error",
            }

        return {
            "ok": True,
            "result": sandbox_result.stdout.strip(),
            "expression": expression,
            "method": "sandbox",
            "error": None,
        }


def register_calc_tool(registry, profile: HardwareProfile, sandbox: SandboxRunner | None = None) -> None:
    """Register the ``calc`` tool on a ToolRegistry.

    Parameters
    ----------
    registry:
        ToolRegistry instance.
    profile:
        Hardware profile for library allowlisting.
    sandbox:
        Optional SandboxRunner instance.
    """
    calc = CalcTool(profile, sandbox)
    registry.register("calc", calc)

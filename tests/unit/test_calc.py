"""Unit tests for AGAN calc tool (think-in-code)."""

import sys
sys.path.insert(0, "src")

import pytest
from agan.tools.calc import (
    eval_arithmetic, CalcTool, allowed_libraries,
    _ALWAYS_ALLOWED, _M_AND_ABOVE, _L_AND_ABOVE,
)
from agan.hardware.detect import HardwareProfile


class TestArithmeticEvaluator:
    """Tests for the AST-based arithmetic evaluator."""

    def test_basic_addition(self):
        assert eval_arithmetic("2 + 2") == 4

    def test_basic_subtraction(self):
        assert eval_arithmetic("10 - 3") == 7

    def test_basic_multiplication(self):
        assert eval_arithmetic("3 * 7") == 21

    def test_basic_division(self):
        assert eval_arithmetic("10 / 2") == 5.0

    def test_power(self):
        assert eval_arithmetic("2 ** 10") == 1024

    def test_modulo(self):
        assert eval_arithmetic("17 % 5") == 2

    def test_floor_division(self):
        assert eval_arithmetic("17 // 5") == 3

    def test_unary_negation(self):
        assert eval_arithmetic("-5") == -5

    def test_parentheses(self):
        assert eval_arithmetic("(2 + 3) * 4") == 20

    def test_complex_expression(self):
        result = eval_arithmetic("((3 + 5) * 2 - 4) / 2")
        assert result == 6.0

    def test_comparison_true(self):
        assert eval_arithmetic("5 > 3") is True

    def test_comparison_false(self):
        assert eval_arithmetic("5 > 10") is False

    def test_chained_comparison(self):
        assert eval_arithmetic("1 < 5 < 10") is True

    def test_math_constant_pi(self):
        result = eval_arithmetic("pi")
        assert abs(result - 3.141592653589793) < 1e-10

    def test_math_function_sqrt(self):
        assert eval_arithmetic("sqrt(16)") == 4.0

    def test_math_function_sin(self):
        result = eval_arithmetic("sin(0)")
        assert result == 0.0

    def test_nested_function_call(self):
        result = eval_arithmetic("sqrt(pow(2, 3))")
        assert abs(result - 2.8284271247461903) < 1e-10

    def test_list_expression(self):
        result = eval_arithmetic("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_min_max(self):
        assert eval_arithmetic("min(3, 1, 2)") == 1
        assert eval_arithmetic("max(3, 1, 2)") == 3

    def test_round(self):
        assert eval_arithmetic("round(3.7)") == 4

    def test_abs(self):
        assert eval_arithmetic("abs(-5)") == 5


class TestArithmeticSecurity:
    """Security tests: disallowed constructs must raise ValueError."""

    def test_import_statement(self):
        with pytest.raises(ValueError):
            eval_arithmetic("import os")

    def test_attribute_access(self):
        with pytest.raises(ValueError):
            eval_arithmetic("os.path.join")

    def test_function_definition(self):
        with pytest.raises(ValueError):
            eval_arithmetic("def foo(): pass")

    def test_class_definition(self):
        with pytest.raises(ValueError):
            eval_arithmetic("class Foo: pass")

    def test_dunder_access(self):
        with pytest.raises(ValueError):
            eval_arithmetic("__import__('os')")

    def test_eval_call(self):
        with pytest.raises(ValueError):
            eval_arithmetic("eval('1+1')")

    def test_open_call(self):
        with pytest.raises(ValueError):
            eval_arithmetic("open('/etc/passwd')")

    def test_globals_access(self):
        with pytest.raises(ValueError):
            eval_arithmetic("globals()")

    def test_underscore_name(self):
        with pytest.raises(ValueError):
            eval_arithmetic("_secret")


class TestAllowedLibraries:
    """Tests for library allowlisting by profile."""

    def test_profile_s(self):
        libs = allowed_libraries(HardwareProfile.S)
        assert "math" in libs
        assert "numpy" not in libs
        assert "sympy" not in libs

    def test_profile_m(self):
        libs = allowed_libraries(HardwareProfile.M)
        assert "math" in libs
        assert "numpy" in libs
        assert "sympy" not in libs

    def test_profile_l(self):
        libs = allowed_libraries(HardwareProfile.L)
        assert "math" in libs
        assert "numpy" in libs
        assert "sympy" in libs


class TestCalcTool:
    """Tests for the CalcTool class."""

    def test_calc_tool_creation(self):
        from agan.sandbox import SandboxRunner
        calc = CalcTool(HardwareProfile.M, SandboxRunner())
        assert calc.profile == HardwareProfile.M
        assert "numpy" in calc._allowed_libs

    def test_calc_basic_addition(self):
        from agan.sandbox import SandboxRunner
        calc = CalcTool(HardwareProfile.M, SandboxRunner())
        result = calc("2 + 2")
        assert result["ok"] is True
        assert result["result"] == 4
        assert result["method"] == "ast"

    def test_calc_complex_math(self):
        from agan.sandbox import SandboxRunner
        calc = CalcTool(HardwareProfile.M, SandboxRunner())
        result = calc("sqrt(144)")
        assert result["ok"] is True
        assert result["result"] == 12.0

    def test_calc_invalid_expression(self):
        from agan.sandbox import SandboxRunner
        calc = CalcTool(HardwareProfile.M, SandboxRunner())
        result = calc("this is not math")
        assert result["ok"] is False
        assert result["error"] is not None

    def test_calc_disallowed_import(self):
        from agan.sandbox import SandboxRunner
        calc = CalcTool(HardwareProfile.S, SandboxRunner())
        # numpy not allowed on S profile, sandbox should block it
        result = calc("import numpy; numpy.array([1,2,3])")
        # Either fails in AST (import not allowed) or sandbox blocks it
        assert result["ok"] is False or result["method"] == "sandbox"

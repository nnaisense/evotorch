from typing import Optional, Union

import torch


def _violation(lhs: torch.Tensor, comparison: str, rhs: torch.Tensor) -> torch.Tensor:
    from torch.nn.functional import relu

    if comparison == ">=":
        return relu(rhs - lhs)
    elif comparison == "<=":
        return relu(lhs - rhs)
    elif comparison == "==":
        return torch.abs(lhs - rhs)
    else:
        raise ValueError(
            f"Unrecognized comparison operator: {repr(comparison)}."
            " Supported comparison operators are: '>=', '<=', '=='"
        )


def violation(
    lhs: Union[float, torch.Tensor],
    comparison: str,
    rhs: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Get the amount of constraint violation.

    Args:
        lhs: The left-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            left-hand-side values.
        comparison: The operator used for comparing the left-hand-side and the
            right-hand-side. Expected as a string. Acceptable values are:
            '<=', '==', '>='.
        rhs: The right-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            right-hand-side values.
    Returns:
        The amount of violation of the constraint. A value of 0 means that
        the constraint is not violated at all. The returned violation amount(s)
        are always non-negative.
    """
    from ..decorators import expects_ndim

    return expects_ndim(_violation, (0, None, 0))(lhs, comparison, rhs)


def _log_barrier(
    lhs: torch.Tensor,
    comparison: str,
    rhs: torch.Tensor,
    sharpness: torch.Tensor,
    penalty_sign: str,
    inf: torch.Tensor,
) -> torch.Tensor:
    from torch.nn.functional import relu

    if comparison == ">=":
        log_input = relu(lhs - rhs)
    elif comparison == "<=":
        log_input = relu(rhs - lhs)
    else:
        raise ValueError(
            f"Unrecognized comparison operator: {repr(comparison)}. Supported comparison operators are: '>=', '<='"
        )

    log_input = log_input

    result = torch.log(log_input) / sharpness

    inf = -inf
    result = torch.where(result < inf, inf, result)

    if penalty_sign == "-":
        pass  # nothing to do
    elif penalty_sign == "+":
        result = -result
    else:
        raise ValueError(f"Unrecognized penalty sign: {repr(penalty_sign)}. Supported penalty signs are: '+', '-'")

    return result


def log_barrier(
    lhs: Union[float, torch.Tensor],
    comparison: str,
    rhs: Union[float, torch.Tensor],
    *,
    penalty_sign: str,
    sharpness: Union[float, torch.Tensor] = 1.0,
    inf: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Return a penalty based on how close the constraint is to being violated.

    If the left-hand-side is equal to the right-hand-side, or if the constraint
    is violated, the returned penalty will be infinite (`+inf` or `-inf`,
    depending on `penalty_sign`). Such `inf` values can result in numerical
    instabilities. To overcome such instabilities, you might want to set the
    keyword argument `inf` as a large-enough finite positive quantity `M`, so
    that very large (or infinite) penalties will be clipped down to `M`.

    Args:
        lhs: The left-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            left-hand-side values.
        comparison: The operator used for comparing the left-hand-side and the
            right-hand-side. Expected as a string. Acceptable values are:
            '<=', '>='.
        rhs: The right-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            right-hand-side values.
        penalty_sign: Expected as string, either as '+' or '-', which
            determines the sign of the penalty (i.e. determines if the penalty
            will be positive or negative). One should consider the objective
            sense of the fitness function at hand for deciding `penalty_sign`.
            For example, if a fitness function is written from the perspective
            of maximization, the penalties should be negative, and therefore,
            `penalty_sign` must be given as '-'.
        sharpness: The logarithmic penalty will be divided by this number.
            By default, this value is 1. A sharper log-penalization allows
            the constraint to get closer to its boundary, and then makes
            a more sudden jump towards infinity.
        inf: When concerned about the possible numerical instabilities caused
            by infinite penalties, one can specify a finite large-enough
            positive quantity `M` through this argument. As a result,
            infinite penalties will be clipped down to the finite `M`.
            One might also think of this as temporarily replacing `inf` with
            `M` while computing the log-penalties.
    Returns:
        Log-penalty amount(s), whose sign(s) is/are determined by
        `penalty_sign`.
    """
    from ..decorators import expects_ndim

    if inf is None:
        inf = float("inf")

    return expects_ndim(_log_barrier, (0, None, 0, 0, None, 0))(lhs, comparison, rhs, sharpness, penalty_sign, inf)


def _penalty(
    lhs: torch.Tensor,
    comparison: str,
    rhs: torch.Tensor,
    penalty_sign: str,
    linear: torch.Tensor,
    step: torch.Tensor,
    exp: torch.Tensor,
    exp_inf: torch.Tensor,
) -> torch.Tensor:
    # Get the amount of violation that exists on the constraint
    violation_amount = _violation(lhs, comparison, rhs)

    # Get the constants 0 and 1 on the correct device, with the correct dtype
    zero = torch.zeros_like(violation_amount)
    one = torch.zeros_like(violation_amount)

    # Initialize the penalty as the amount of violation times the `linear` multiplier
    penalty = linear * violation_amount

    # Increase the penalty by the constant `step` if there is violation
    penalty = penalty + torch.where(violation_amount > zero, step, zero)

    # Check if `exp` is given. `nan` means that `exp` is omitted, in which case no penalty will be applied.
    exp_given = ~(torch.isnan(exp))

    # Compute an exponential penalty if the argument `exp` is given
    exped_penalty = violation_amount ** torch.where(exp_given, exp, one)

    # Clip the exponential penalty so that it does not exceed `exp_inf`
    exped_penalty = torch.where(exped_penalty > exp_inf, exp_inf, exped_penalty)

    # If the argument `exp` is given, add the exponential penalty onto the main penalty
    penalty = penalty + torch.where(exp_given, exped_penalty, zero)

    # Apply the correct sign on the penalty
    if penalty_sign == "+":
        pass  # nothing to do
    elif penalty_sign == "-":
        penalty = -penalty
    else:
        raise ValueError(f"Unrecognized penalty sign: {repr(penalty_sign)}." "Supported penalty signs are: '+', '-'")

    # Finally, the accumulated penalty is returned
    return penalty


def penalty(
    lhs: Union[float, torch.Tensor],
    comparison: str,
    rhs: Union[float, torch.Tensor],
    *,
    penalty_sign: str,
    linear: Optional[Union[float, torch.Tensor]] = None,
    step: Optional[Union[float, torch.Tensor]] = None,
    exp: Optional[Union[float, torch.Tensor]] = None,
    exp_inf: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Return a penalty based on the amount of violation of the constraint.

    Depending on the provided arguments, the penalty can be linear,
    or exponential, or based on step function, or a combination of these.

    Args:
        lhs: The left-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            left-hand-side values.
        comparison: The operator used for comparing the left-hand-side and the
            right-hand-side. Expected as a string. Acceptable values are:
            '<=', '==', '>='.
        rhs: The right-hand-side of the constraint. In the non-batched case,
            this is expected as a scalar. If it is given as an n-dimensional
            tensor where n is at least 1, this is considered as a batch of
            right-hand-side values.
        penalty_sign: Expected as string, either as '+' or '-', which
            determines the sign of the penalty (i.e. determines if the penalty
            will be positive or negative). One should consider the objective
            sense of the fitness function at hand for deciding `penalty_sign`.
            For example, if a fitness function is written from the perspective
            of maximization, the penalties should be negative, and therefore,
            `penalty_sign` must be given as '-'.
        linear: Multiplier for the linear component of the penalization.
            If omitted (i.e. left as None), the value of this multiplier will
            be 0 (meaning that there will not be any linear penalization).
            In the non-batched case, this argument is expected as a scalar.
            If this is provided as a tensor 1 or more dimensions, those
            dimensions will be considered as batch dimensions.
        step: The constant amount that will be added onto the penalty if there
            is a violation. If omitted (i.e. left as None), this value is 0.
            In the non-batched case, this argument is expected as a scalar.
            If this is provided as a tensor 1 or more dimensions, those
            dimensions will be considered as batch dimensions.
        exp: A constant `p` that will enable exponential penalization in the
            form `amount_of_violation ** p`. If this is left as None or is
            given as `nan`, there will be no exponential penalization.
            In the non-batched case, this argument is expected as a scalar.
            If this is provided as a tensor 1 or more dimensions, those
            dimensions will be considered as batch dimensions.
        exp_inf: Upper bound for exponential penalty values. If exponential
            penalty is enabled but `exp_inf` is omitted (i.e. left as None),
            the exponential penalties can jump to very large values or to
            infinity, potentially causing numerical instabilities. To avoid
            such numerical instabilities, one might provide a large-enough
            positive constant `M` via the argument `exp_inf`. When such a value
            is given, exponential penalties will not be allowed to exceed `M`.
            One might also think of this as temporarily replacing `inf` with
            `M` while computing the exponential penalties.
    Returns:
        The penalty amount(s), whose sign(s) is/are determined by
        `sign_penalty`.
    """
    from ..decorators import expects_ndim

    if linear is None:
        linear = 0.0
    if step is None:
        step = 0.0
    if exp is None:
        exp = float("nan")
    if exp_inf is None:
        exp_inf = float("inf")

    return expects_ndim(_penalty, (0, None, 0, None, 0, 0, 0, 0))(
        lhs,
        comparison,
        rhs,
        penalty_sign,
        linear,
        step,
        exp,
        exp_inf,
    )

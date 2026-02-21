"""Bunger's 1D product toy example: shrink wrapping improves lower bound.
    Matches example and values in Bunger's preconditiong paper with test cases"""

from sage.all import RIF, PolynomialRing

from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.TMVector import TMVector
from TERA.TMFlow import Precondition


def _make_identity_tm(delta: float, max_order: int = 8) -> TaylorModel:
    ring = PolynomialRing(RIF, names=("x",))
    x = ring.gens()[0]
    poly = Polynomial(_poly=x, _ring=ring)
    domain = [Interval(-1.0, 1.0)]
    ref_point = (0.0,)
    rem = Interval(-delta, delta)
    return TaylorModel(poly=poly, rem=rem, domain=domain, ref_point=ref_point, max_order=max_order)


def _bound_tm(tm: TaylorModel) -> Interval:
    return tm.bound()


def run():
    deltas = [0.01, 0.1, 0.25]
    eps = 1e-10

    for delta in deltas:
        print(f"Delta: {delta}")
        tm = _make_identity_tm(delta)
        prod = tm * tm
        bound_R = _bound_tm(prod)

        sw = Precondition.shrink_wrap_corrected(
            TMVector([tm]),
            time_var=None,
            slack_q=1e-12,
            max_iter=10,
            q_cap=1.3,
            use_preconditioning=False,
            verbose=True,
        )
        assert sw.get("success", False), f"shrink wrap failed for delta={delta}: {sw}"
        tm_sw = sw["T_sw"].tms[0]
        prod_sw = tm_sw * tm_sw
        bound_S = _bound_tm(prod_sw)

        print(f"Bound R: {bound_R}")
        print(f"Bound S: {bound_S}")
        print()


        exp_R_lo = -delta * (2.0 + delta)
        exp_R_hi = (1.0 + delta) ** 2
        exp_S_lo = 0.0
        exp_S_hi = (1.0 + delta) ** 2

        assert float(bound_R.lower) <= exp_R_lo + eps, (
            f"delta={delta} bound_R.lower={bound_R.lower} exp_R_lo={exp_R_lo}"
        )
        assert float(bound_R.upper) >= exp_R_hi - eps, (
            f"delta={delta} bound_R.upper={bound_R.upper} exp_R_hi={exp_R_hi}"
        )
        assert float(bound_S.lower) <= exp_S_lo + eps, (
            f"delta={delta} bound_S.lower={bound_S.lower} exp_S_lo={exp_S_lo}"
        )
        assert float(bound_S.upper) >= exp_S_hi - eps, (
            f"delta={delta} bound_S.upper={bound_S.upper} exp_S_hi={exp_S_hi}"
        )

        assert float(bound_S.lower) >= float(bound_R.lower) - eps, (
            f"delta={delta} bound_S.lower={bound_S.lower} bound_R.lower={bound_R.lower}"
        )
        assert float(bound_S.lower) >= -eps, (
            f"delta={delta} bound_S.lower={bound_S.lower}"
        )
        assert float(bound_S.upper) <= float(bound_R.upper) + 1e-6, (
            f"delta={delta} bound_S.upper={bound_S.upper} bound_R.upper={bound_R.upper}"
        )


if __name__ == "__main__":
    run()

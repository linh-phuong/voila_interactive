def LV_mono(n, t, pars):
    r, K = pars
    return r * (1 - n / K) * n


def LV_mono1(n, t, pars):
    r, a = pars
    return (r + a * n) * n


def LV_compete(n, t, pars):
    r1, K1, r2, K2, alpha, beta = pars
    n1, n2 = n
    return (r1 * (1 - n1 / K1 - alpha * n2 / K1) * n1, r2 * (1 - n2 / K2 - beta * n1 / K2) * n2)


def LV_compete1(n, t, pars):
    r1, a_11, r2, a_22, a_12, a_21 = pars
    n1, n2 = n
    return ((r1 - a_11 * n1 - a_12 * n2) * n1, (r2 - a_21 * n1 - a_22 * n2) * n2)


def LV_compete_jac(n, pars):
    r1, K1, r2, K2, alpha, beta = pars
    n1, n2 = n
    return (
        (r1 - 2 * n1 * r1 / K1 - alpha * n2 * r1 / K1, -alpha * n1 * r1 / K1),
        (-beta * n2 * r2 / K2, r2 - beta * n1 * r2 / K2 - 2 * n2 * r2 / K2),
    )


def Her_jac(n, pars):
    r1, K1, r2, K2, alpha, beta = pars
    n1, n2 = n
    return [
        [
            r1 * (K1 - 2 * n1 - n2 * alpha) / K1,
            -(K2 * n1 * r1 * alpha + K1 * n2 * r2 * beta) / (2 * K1 * K2),
        ],
        [
            -(K2 * n1 * r1 * alpha + K1 * n2 * r2 * beta) / (2 * K1 * K2),
            r2 * (K2 - 2 * n2 - n1 * beta) / K2,
        ],
    ]


def LV_compete_eq_coexist(pars):
    K1, K2, alpha, beta = pars
    return [(K1 - K2 * alpha) / (1 - alpha * beta), (K2 - K1 * beta) / (1 - alpha * beta)]


def get_competition_coefficient(fit_param_object):
    """
    get the competition coefficient a_11, a_12, a_21, a_22
    from the Lotka Voltera model
    """
    r1, K1, r2, K2, alpha, beta = fit_param_object.x[1::]
    a_11 = r1 / K1
    a_22 = r2 / K2
    a_12 = alpha * a_11
    a_21 = beta * a_22
    return (r1, a_11, r2, a_22, a_12, a_21)


def get_compete_slope(fit_param_object):
    _, a_11, _, a_22, a_12, a_21 = get_competition_coefficient(fit_param_object)
    return (a_21 / a_11, a_22 / a_12)

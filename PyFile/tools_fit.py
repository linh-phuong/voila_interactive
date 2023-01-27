import numpy as np
import pandas as pd
from scipy.integrate import odeint


def ll_lognormal(dat: list, sigma: np.float):
    """
    Calculating the log likelihood of data with errors following a log normal distribution

    Args
    ====
    dat: a list of length 3, elements include observe data, expect data and the number of observations
        observe and expect data are log values
    sigma: sigma parameter of the log normal distribution

    Return
    ======
    log likelihood value
    """
    observe, expect, timelast = dat
    logdiff = observe - expect
    loglikelihood = (
        -(timelast / 2) * np.log(2 * np.pi)
        - (timelast / 2) * np.log(sigma**2)
        - (1 / (2 * sigma**2)) * np.sum(logdiff**2)
        - np.sum(observe)
    )
    return loglikelihood


def negll_lognormal(pars: list, args: tuple):
    """
    This is the likelihood function that can be used for the scipy.optimize.minimize to find the
    maximum likelihood estimation for data whose error follows a log normal distribution

    Args
    ====
    pars (list): parameters to be estimated
                parameters order has to be exactly: sigma, and then model parameters
    args (tuple): the order of the elements in the tuple is model (func as input to odeint),
                observe data (true value and not log)

    Return
    ======
    negative value of the likelihood so that using scipy.optimize.minimize will return the maximum likelihood
    """
    mdl, observe = args
    assert isinstance(observe, pd.DataFrame), "observe has to be pandas DataFrame"
    assert "day" in observe.columns, "time data is not in the observe data"
    assert "poplog" in observe.columns, "log density is not in the observe data"
    assert "dens" in observe.columns, "raw density is not in the observe data"
    sigma = pars[0]
    mdl_pars = pars[1::]
    t_series = observe.day
    start_dens = observe.dens[0]
    expect_pop = odeint(mdl, start_dens, t_series, args=(mdl_pars,))
    return -ll_lognormal((observe.poplog, np.log(expect_pop[:, 0]), t_series.iloc[-1]), sigma)


def negll_lognormal_compete(pars: list, args: tuple):
    sigma = pars[0]
    mdl_pars = pars[1::]
    mdl, observe = args
    t_series = observe.day
    start_dens = (observe.dens_0[0], observe.dens_1[1])
    expect_pop = odeint(mdl, start_dens, t_series, args=(mdl_pars,))
    ll0 = ll_lognormal((observe.poplog_0, np.log(expect_pop[:, 0]), t_series.iloc[-1]), sigma)
    ll1 = ll_lognormal((observe.poplog_1, np.log(expect_pop[:, 1]), t_series.iloc[-1]), sigma)
    return -ll0 - ll1


def negll_full(pars: list, args: tuple):
    sigma = pars[0]
    sp1_pars = pars[1:3]
    sp2_pars = pars[3:5]
    mdl_mono, mdl_compete, observe1, observe2, observe_12 = args
    start_1, start_2 = observe1.dens[0], observe2.dens[0]
    start_compete = (observe_12.dens_0[0], observe_12.dens_1[0])
    t_series1, t_series2, t_series12 = observe1.day, observe2.day, observe_12.day
    expt_1 = odeint(mdl_mono, start_1, t_series1, args=(sp1_pars,))
    ll_mono1 = ll_lognormal((observe1.poplog, np.log(expt_1[:, 0]), t_series1.iloc[-1]), sigma)
    expt_2 = odeint(mdl_mono, start_2, t_series2, args=(sp2_pars,))
    ll_mono2 = ll_lognormal((observe2.poplog, np.log(expt_2[:, 0]), t_series2.iloc[-1]), sigma)
    expt_compete = odeint(mdl_compete, start_compete, t_series12, args=(pars[1::],))
    ll_compete1 = ll_lognormal(
        (observe_12.poplog_0, np.log(expt_compete[:, 0]), t_series12.iloc[-1]), sigma
    )
    ll_compete2 = ll_lognormal(
        (observe_12.poplog_1, np.log(expt_compete[:, 1]), t_series12.iloc[-1]), sigma
    )
    return -ll_mono1 - ll_mono2 - ll_compete1 - ll_compete2


def ls_multivariate(pars: float, dats: list):
    """
    Args
    ====
    pars (float): competition parameter in lotka-voltera
    dats (list): include focal population, population that has effect, r of focal population,
                carrying capacity of focal population, expected rate

    Return
    ======
    sum of least square
    """
    pop_focal, pop_compete, restime, Kestime, expect = dats
    pred = restime - pop_focal * restime / Kestime - restime * pop_compete * pars / Kestime
    return sum((pred - expect) ** 2)


def logistic_ls(pars, dat):
    a, b = pars
    ypred = a + b * dat.dens
    return sum((ypred - dat.rate) ** 2)

"""
Copyright (C) Enzo Busseti 2016-2019 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Code written before September 2016 is copyrighted to 
Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.,
and is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import numpy as np
import logging
import cvxpy as cvx
from abc import ABCMeta, abstractmethod
from datetime import datetime

from cvxportfolio.costs import BaseCost
from cvxportfolio.returns import BaseReturnsModel, ReturnsForecast
from cvxportfolio.constraints import BaseConstraint
from cvxportfolio.utils.data_management import time_locator, null_checker

logging.basicConfig(format='%(asctime)s %(message)s')

__all__ = ['Hold', 'FixedTrade', 'PeriodicRebalance', 'AdaptiveRebalance',
           'SinglePeriodOpt', 'MultiPeriodOpt', 'BlackLittermanSPOpt', 'MultiPeriodScenarioOpt',
           'ProportionalTrade', 'RankAndLongShort', 'ModelPredictiveControlScenarioOpt']


class BasePolicy(object):
    """ Base class for a trading policy. """
    __metaclass__ = ABCMeta

    def __init__(self, trading_freq='day'):
        """
        :param trading_freq: supported options are "day", "week", "month", "quarter", "year".
                    rebalance on the first day of each new period
        """
        self.costs = []
        self.constraints = []
        self.trading_freq = trading_freq
        logging.debug('BasePolicy: trading_freq {}'.format(trading_freq))

    @abstractmethod
    def get_trades(self, portfolio, t=datetime.today(), *args):
        """Trades list given current portfolio and time t.
        """
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        return np.round(self.get_trades(portfolio,
                                        t) / time_locator(prices, t))[:-1]

    def is_start_period(self, t):
        """
        Validates if time == start of period -> should rebalance
        :param t:
        :return:
        """
        # If we haven't yet stored the last trade date, this is the first time we're running
        #   Run once at the very start and only again when we're at a rebalance date
        if hasattr(self, 'last_t'):
            try:
                result = getattr(t, self.trading_freq) != getattr(self.last_t, self.trading_freq)
            except AttributeError as e:
                logging.debug('BasePolicy: trading_freq `{t}` is not supported, skipping.'.
                              format(t=self.trading_freq))
                result = False
        else:
            result = True

            if not hasattr(t, self.trading_freq):
                logging.warning('BasePolicy: trading_freq `{t}` is not supported, the policy will only trade once.'.
                                format(t=self.trading_freq))
        logging.debug('BasePolicy: dt {0} is {1} a start period'.format(str(t), "" if result else " not "))

        self.last_t = t
        return result


class Hold(BasePolicy):
    """Hold initial portfolio.
    """

    def get_trades(self, portfolio, t=datetime.today(), *args):
        return self._nulltrade(portfolio)


class RankAndLongShort(BasePolicy):
    """Rank assets, long the best and short the worst (cash neutral).

    """

    def __init__(self, return_forecast, num_long, num_short, target_turnover=1, trading_freq='day', **kwargs):
        """

        :param return_forecast:
        :param num_long:
        :param num_short:
        :param target_turnover:
        :param trading_freq: supported options are "day", "week", "month", "quarter", "year".
                    rebalance on the first day of each new period
        """
        self.return_forecast = return_forecast
        self.num_long = num_long
        self.num_short = num_short
        self.target_turnover = target_turnover
        self.trading_freq = trading_freq
        super(RankAndLongShort, self).__init__(trading_freq)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        # Create flattening trades
        u_flatten = portfolio * -1 * self.target_turnover

        # Retrieve the current time period's return predictions
        prediction = time_locator(self.return_forecast, t, as_numpy=False)
        sorted_ret = prediction.sort_values()

        # Grab the indices for the longs and shorts
        short_trades = sorted_ret.index[:self.num_short]
        long_trades = sorted_ret.index[-self.num_long:]

        # Create trades vector and assign equal weight for long and for shorts
        u_enter = pd.Series(0., index=prediction.index)
        if self.num_short > 0:
            u_enter[short_trades] = -1. / self.num_short
        if self.num_long:
            u_enter[long_trades] = 1. / self.num_long

        # Normalize and allocate cash
        u_enter /= sum(abs(u_enter))
        u_enter = u_enter * sum(portfolio) * self.target_turnover

        # Total trades = flatten + enter
        u = u_flatten + u_enter

        return u if self.is_start_period(t) else self._nulltrade(portfolio)


class ProportionalTrade(BasePolicy):
    """Gets to target in given time steps."""

    def __init__(self, targetweight, time_steps, **kwargs):
        self.targetweight = targetweight
        self.time_steps = time_steps
        super(ProportionalTrade, self).__init__(**kwargs)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        try:
            missing_time_steps = len(
                self.time_steps) - next(i for (i, x)
                                        in enumerate(self.time_steps)
                                        if x == t)
        except StopIteration:
            raise Exception(
                "ProportionalTrade can only trade on the given time steps")
        deviation = self.targetweight - portfolio / sum(portfolio)
        return sum(portfolio) * deviation / missing_time_steps


class SellAll(BasePolicy):
    """Sell all non-cash assets."""

    def get_trades(self, portfolio, t=datetime.today(), *args):
        trade = -pd.Series(portfolio, copy=True)
        trade.ix[-1] = 0.
        return trade


class FixedTrade(BasePolicy):
    """Trade a fixed trade vector.
    """

    def __init__(self, tradevec=None, tradeweight=None, **kwargs):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        if tradevec is not None and tradeweight is not None:
            raise(Exception(
                'only one of tradevec and tradeweight can be passed'))
        if tradevec is None and tradeweight is None:
            raise(Exception('one of tradevec and tradeweight must be passed'))
        self.tradevec = tradevec
        self.tradeweight = tradeweight
        assert(self.tradevec is None or sum(self.tradevec) == 0.)
        assert(self.tradeweight is None or sum(self.tradeweight) == 0.)
        super(FixedTrade, self).__init__(**kwargs)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        if self.tradevec is not None:
            return self.tradevec
        return sum(portfolio) * self.tradeweight


class BaseRebalance(BasePolicy):

    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class PeriodicRebalance(BaseRebalance):
    """Track a target portfolio, rebalancing at given times.
    """

    def __init__(self, target, trading_freq, **kwargs):
        """
        Args:
            target: target weights, n+1 vector
            trading_freq: supported options are "day", "week", "month", "quarter",
                "year".
                rebalance on the first day of each new period
        """
        self.target = target
        self.trading_freq = trading_freq
        super(PeriodicRebalance, self).__init__(trading_freq)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        return self._rebalance(portfolio) if self.is_start_period(t) else \
            self._nulltrade(portfolio)


class AdaptiveRebalance(BaseRebalance):
    """ Rebalance portfolio when deviates too far from target.
    """

    def __init__(self, target, tracking_error, **kwargs):
        self.target = target
        self.tracking_error = tracking_error
        super(AdaptiveRebalance, self).__init__(**kwargs)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        weights = portfolio / sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class SinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(self, return_forecast, costs, constraints, solver=None, solver_opts=None, **kwargs):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast

        super(SinglePeriodOpt, self).__init__(**kwargs)

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)
            logging.info('SPOpt: Adding cost: {}'.format(str(cost)))

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)
            logging.info('SPOpt: Adding constraint: {}'.format(str(constraint)))

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts
        logging.debug('SPOpt: solver {0}, solver_opts {1}'.format(solver, str(solver_opts)))

    def get_trades(self, portfolio, t=None, *args):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = datetime.today()

        # Exit early if we're not trading in this period
        if not self.is_start_period(t):
            logging.info('Skipping ' + str(t) + ', no trading allowed by policy')
            return self._nulltrade(portfolio)

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(cvx.multiply(
                time_locator(self.return_forecast, t, as_numpy=True), wplus))

        assert(alpha_term.is_concave())

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [item for item in (con.weight_expr(t, wplus, z, value)
                                          for con in self.constraints)]

        for el in costs:
            assert (el.is_convex())

        for el in constraints:
            assert (el.is_dcp())

        self.prob = cvx.Problem(
            cvx.Maximize(alpha_term - sum(costs)),
            [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == 'unbounded':
                logging.error(
                    'The problem is unbounded. Defaulting to no trades')
                return self._nulltrade(portfolio)

            if self.prob.status == 'infeasible':
                logging.error(
                    'The problem is infeasible. Defaulting to no trades')
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except cvx.SolverError:
            logging.error(
                'The solver %s failed. Defaulting to no trades' % self.solver)
            return self._nulltrade(portfolio)


class MultiPeriodOpt(SinglePeriodOpt):

    def __init__(self, trading_times, terminal_weights, lookahead_periods=None,
                 trading_freq='day', **kwargs):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.trading_freq = trading_freq
        # Should there be a constraint that the final portfolio is the bmark?
        self.terminal_weights = terminal_weights
        super(MultiPeriodOpt, self).__init__(trading_freq, **kwargs)

    def get_trades(self, portfolio, t=datetime.today(), *args):

        # Exit early if we're not trading in this period
        if not self.is_start_period(t):
            logging.info('Skipping ' + str(t) + ', no trading allowed by policy')
            return self._nulltrade(portfolio)

        value = sum(portfolio)
        assert (value > 0.)
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in \
                self.trading_times[self.trading_times.index(t):
                                   self.trading_times.index(t) +
                                   self.lookahead_periods]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:

            #            tau = t + delta_t
            z = cvx.Variable(*w.size)
            wplus = w + z
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(
                    t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(z) == 0]
            constr += [con.weight_expr(t, wplus, z, value)
                       for con in self.constraints]

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        sum(prob_arr).solve(solver=self.solver)
        return pd.Series(index=portfolio.index,
                         data=(z_vars[0].value * value))


class BlackLittermanSPOpt(BasePolicy):
    """
    Implements the Black Litterman model for asset allocation
    """
    def __init__(self, r_posterior, sigma_posterior, delta,
                 trading_freq='day', target_turnover=1, **kwargs):
        """
        Initialize required variables for the optimization
        :param r_posterior: returns with views incorporated
        :param sigma_posterior: sigmas with views incorporated
        :param delta: risk aversion coefficient
        :param target_turnover: target turnover
                    1 is 100% turnover each time
        :param trading_freq: supported options are "day", "week", "month", "quarter", "year".
                    rebalance on the first day of each new period
        """
        # Initialization
        self.r_posterior = r_posterior
        self.sigma_posterior = sigma_posterior
        self.delta = delta
        self.trading_freq = trading_freq
        self.target_turnover = target_turnover

        super().__init__(trading_freq)

    def get_trades(self, portfolio, t=datetime.today(), *args):
        # Retrieve the current time period's return & sigma predictions
        r_post = time_locator(self.r_posterior, t, as_numpy=False)
        sigma_post = time_locator(self.sigma_posterior, t, as_numpy=False)

        # BL optimization result
        u = sum(portfolio) * np.dot(np.linalg.inv(self.delta * sigma_post), r_post)
        u = pd.Series(index=r_post.index, data=u)

        if sum(abs(u.subtract(portfolio))) >= 0.5 * sum(abs(portfolio)):
            logging.debug('Today: {d}, we would have turned over >50% of the portfolio'.format(d=str(t)))
            logging.debug('BLOpt: portfolio: {s}'.format(s=str(portfolio)))
            logging.debug('BLOpt: u: {s}'.format(s=str(u)))

        return u.subtract(portfolio) * self.target_turnover if self.is_start_period(t) else self._nulltrade(portfolio)


class MultiPeriodScenarioOpt(BasePolicy):

    def __init__(self, alphamodel, horizon, scenarios, costs, constraints,
                 scenario_mode='c', scenario_ret_src='bl', solver=None, solver_opts=None, **kwargs):
        """

        :param alphamodel: instance of alpha model class
            can be any class that has generate_forward_scenario method()
        :param horizon: periods ahead for prediction
        :param scenarios: scenarios to optimize against
        :param trading_freq: supported options are "day", "week", "month", "quarter", "year".
                    rebalance on the first day of each new period
        :param scenario_mode: supported options are "g", "c" and "hmm".
                    See alphamodel.ss_hmm or alphamodel.ss_bl_hmm for more details.
        :param scenario_ret_src: support options are "bl" and "pred"
                    See alphamodel.ss_hmm or alphamodel.ss_bl_hmm for more details.
        :param costs:
        :param constraints:
        :param solver:
        :param solver_opts:
        :param kwargs:
        """
        # Initialization
        self.model = alphamodel
        self.horizon = horizon
        self.scenarios = scenarios
        self.scenario_mode = scenario_mode
        self.scenario_ret_src = scenario_ret_src
        logging.info('MultiPeriodScenarioOpt: model {0}, horizon {1}, scenarios {2}, scn mode {3}, scn ret_src {4}'.
                     format(str(alphamodel), horizon, scenarios, scenario_mode, scenario_ret_src))
        if 'generate_forward_scenario' not in dir(self.model):
            raise ValueError('MultiPeriodScenarioOpt: Model class requires a generate_forward_scenario() method. '
                             'See: alphamodel.ss_hmm')

        # Should there be a constraint that the final portfolio is the bmark?
        # self.terminal_weights = terminal_weights

        # Initialize base policy and cost/constraint arrays
        super().__init__(**kwargs)

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)
            logging.info('MultiPeriodScenarioOpt: Adding cost: {}'.format(str(cost)))

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)
            logging.info('MultiPeriodScenarioOpt: Adding constraint: {}'.format(str(constraint)))

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts
        logging.debug('MultiPeriodScenarioOpt: solver {}, solver_opts {}'.format(solver, str(solver_opts)))

    def get_trades(self, portfolio, t=datetime.today(), *args, **kwargs):

        # Exit early if we're not trading in this period
        if not self.is_start_period(t):
            logging.debug('MultiPeriodScenarioOpt: Skipping {}, no trading allowed by policy'.format(str(t)))
            return self._nulltrade(portfolio)
        logging.info('MultiPeriodScenarioOpt: Running for {}'.format(str(t)))

        # Retrieve weights from portfolio allocation
        value = sum(portfolio)
        assert (value > 0.)
        w = cvx.Constant(portfolio.values / value)

        # Initialization of optimization problem
        prob_arr = []
        z_vars = []

        # Generate scenarios to optimize for
        scenarios = []
        for s in range(self.scenarios):
            scenarios.append(self.model.generate_forward_scenario(t, self.horizon,
                                                                  mode=self.scenario_mode,
                                                                  return_src=self.scenario_ret_src))

        # For each timeslot tau
        for tau in scenarios[0].returns.index:

            # Initialize a new set of trades z & a new objective under new constraints
            z = cvx.Variable(w.size)
            wplus = w + z
            obj = cvx.Constant(0)
            constr = []

            # Across all scenarios, optimize the same set of trades z
            for s in scenarios:
                # Initialize returns
                return_forecast = ReturnsForecast(s.returns)
                obj += return_forecast.weight_expr(tau, wplus)

                # Construct costs for scenario s
                costs = []
                for cost in self.costs:
                    cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
                    costs.append(cost_expr)
                    constr += const_expr

                # Add costs & constraints
                obj -= sum(costs)
                constr += [cvx.sum(z) == 0]
                constr += [con.weight_expr(t, wplus, z, value) for con in self.constraints]

            # Add objective of optimizing timeslot tau to problem set
            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)

            # Step to next timeslot tau
            w = wplus

        problem = sum(prob_arr)
        problem.solve(solver=self.solver)
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            logging.info('MultiPeriodScenarioOpt: Optimization solved {}'.format(str(problem.solver_stats)))
            logging.info("MultiPeriodScenarioOpt: Optimal value: %s" % problem.value)
            logging.info('MultiPeriodScenarioOpt: z_vars[0] {}'.format(z_vars[0].value))
            return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))
        else:
            logging.warning('MultiPeriodScenarioOpt: Optimization failed, {}'.format(str(problem.status)))
            logging.warning('MultiPeriodScenarioOpt: Problem: {}'.format(str(problem)))
            return self._nulltrade(portfolio)


class ModelPredictiveControlScenarioOpt(MultiPeriodScenarioOpt):

    def __init__(self, return_target, mpc_method='c', **kwargs):
        """

        :param return_target: target return per period
        :param mpc_method: supported options are "c", "e", "er".
                    c: constant target return
        :param kwargs:
        """
        # Initialization
        self.return_target = return_target
        self.mpc_method = mpc_method
        logging.info('ModelPredictiveControlScenarioOpt: return target {}'.format(return_target))
        if self.return_target < 0:
            raise ValueError('ModelPredictiveControlScenarioOpt: return target is required to be >= 0.')

        # Initialize base policy and cost/constraint arrays
        super().__init__(**kwargs)

    def get_trades(self, portfolio, t=datetime.today(), results=None, *args, **kwargs):

        # Exit early if we're not trading in this period
        if not self.is_start_period(t):
            logging.debug('ModelPredictiveControlScenarioOpt: Skipping {}, no trading allowed by policy'.format(str(t)))
            return self._nulltrade(portfolio)
        logging.info('ModelPredictiveControlScenarioOpt: Running for {}'.format(str(t)))

        # Generate scenarios to optimize for
        scenarios = []
        for s in range(self.scenarios):
            scenarios.append(self.model.generate_forward_scenario(t, self.horizon,
                                                                  mode=self.scenario_mode,
                                                                  return_src=self.scenario_ret_src))

        # Compute the desired reference trajectory depending on the MPC method used
        reference_trajectory = pd.Series(index=scenarios[0].returns.index)
        # Constant target trajectory
        if self.mpc_method == 'c':
            reference_trajectory[:] = self.return_target
        else:
            return NotImplemented('ModelPredictiveControlScenarioOpt: Unsupported mpc_method, please see fn signature.')

        # Retrieve weights from portfolio allocation
        value = sum(portfolio)
        assert (value > 0.)
        w = cvx.Constant(portfolio.values / value)

        # Initialization of optimization problem
        prob_arr = []
        z_vars = []
        # return_obj = cvx.Constant(0)

        # For each timeslot tau
        for tau in scenarios[0].returns.index:

            # Initialize a new set of trades z & a new objective under new constraints
            z = cvx.Variable(w.size)
            wplus = w + z
            obj = cvx.Constant(0)
            constr = []

            # Across all scenarios, optimize the same set of trades z
            for s in scenarios:
                # Initialize returns & add to return objective
                obj -= cvx.square(reference_trajectory[tau] - time_locator(s.returns, tau, as_numpy=True) * wplus)

                # Construct costs for scenarios
                costs = []
                for cost in self.costs:
                    cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
                    costs.append(cost_expr)
                    constr += const_expr

                # Add costs & scenario constraints
                obj -= sum(costs)
                constr += [cvx.sum(z) == 0]
                constr += [con.weight_expr(t, wplus, z, value) for con in self.constraints]

            # Add objective of optimizing timeslot tau to problem set
            horizon_prob = cvx.Problem(cvx.Maximize(obj), constr)
            logging.debug('ModelPredictiveControlScenarioOpt: Adding problem {}'.format(str(horizon_prob)))
            prob_arr.append(horizon_prob)
            z_vars.append(z)

            # Step to next timeslot tau
            w = wplus

        problem = sum(prob_arr)
        problem.solve(solver=self.solver)
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            logging.info('ModelPredictiveControlScenarioOpt: Optimization solved {}'.format(str(problem.solver_stats)))
            logging.info("ModelPredictiveControlScenarioOpt: Optimal value: %s" % problem.value)
            logging.info('ModelPredictiveControlScenarioOpt: z_vars[0] {}'.format(z_vars[0].value))
            return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))
        else:
            logging.warning('ModelPredictiveControlScenarioOpt: Optimization failed, {}'.format(str(problem.status)))
            logging.warning('ModelPredictiveControlScenarioOpt: Problem: {}'.format(str(problem)))
            return self._nulltrade(portfolio)


from theano.compile.ops import FromFunctionOp
import pymc3
import numpy as np
from model_constructor import ModelConstructor
import theano.tensor as T
import simpy.core
import scipy

import matplotlib.pyplot as plt
import functools

from measurer import Measurer, EventMeasurer, EventIntervalMeasurer

# import pandas as pd


def run_and_log(sim, fn="output"):
    with open(fn, "w") as f:
        while True:
            try:
                sim.step()
                f.write(str(sim.current_event))
                f.write("\n")
            except simpy.core.EmptySchedule:
                break


def init_model_constructor(f):
    """
    Function decorator that initialises the ModelConstructor of the trainer.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        if not self._init:
            self.model_constructor.parse()
            self._init = True
        return f(*args, **kwargs)
    return wrapper


class Trainer():

    param_distributions = {
        "eye_mvt_scaling_parameter": (pymc3.HalfNormal, {"sd": 1.0}),
        "eye_mvt_angle_parameter":   (pymc3.Gamma, {"alpha": 1.3, "beta": 2}),
        "rule_firing":               (pymc3.Uniform, {"lower": 0.022,
                                                      "upper": 0.1}),

        "sigma":                     (pymc3.HalfNormal, {"sd": 0.5}),

        "intercept":                 (pymc3.Uniform, {"lower": 0.1,
                                                      "upper": 0.4}),
        "latency_factor":           (pymc3.Gamma, {"alpha": 2, "beta": 6}),
        "latency_exponent":         (pymc3.HalfNormal, {"sd": 0.5})
    }

    def __init__(self, sentence_filepath, word_rt_filepath, measurer,
                 params_to_train, model_args={}, verbose=False,
                 advanced=False, default_param_values=None, param_max=100):

        self.default_param_values = default_param_values
        self.check_params_to_train(params_to_train)
        self.params_to_train = set(params_to_train)

        self.mc_model = pymc3.Model()
        self.model_constructor = ModelConstructor(sentence_filepath,
                                                  word_rt_filepath,
                                                  advanced=advanced,
                                                  **model_args)

        self.results = {'eye_mvt_scaling_parameter': 0.002377572678935413,
                        'eye_mvt_angle_parameter': 0.8539963243897579,
                        'rule_firing': 0.05995633050087895}

        # The maximum value any parameter can have in order for the model to
        # still work. This is a very rough lower bound
        self.param_max = param_max

        self.verbose = verbose
        self.measurer = measurer

        self._init = False

    def check_params_to_train(self, params):
        s = set(params) - set(self.param_distributions.keys())
        if s != set():
            raise Exception("Some parameters don't have clearly defined"
                            "distributions: {}".format(str(s)))

    def inspect_run(self, iteration):
        """
        Run a certain simulation iteration in the training process with gui.

        Arguments:
            iteration: a tuple (x, y) where x is the iteration of the training
                       process and y is the index of the sentence on which the
                       model should run.
        """
        self.measurer.add_callback(iteration, lambda x: x.run())

    def inspect_run_log(self, iteration):
        """
        Log a certain simulation iteration in the training process to a file.
        Logs to log_x_y.txt where iteration = (x, y).

        Arguments:
            iteration: a tuple (x, y) where x is the iteration of the training
                       process and y is the index of the sentence on which the
                       model should run.
       """

        ln = "log_{}_{}.txt".format(*iteration)
        self.measurer.add_callback(iteration, functools.partial(run_and_log,
                                                                fn=ln))

    def create_distr(self, name):
        """
        Given a parameter name, create the pymc3 distribution for this
        parameter.

        Arguments:
            name: the name of the parameter.

        Returns:
            A pymc3 distribution.
        """
        p = self.param_distributions[name]
        return p[0](name, **p[1])

    @init_model_constructor
    def observed_measures(self):
        return np.array(list(self.model_constructor.rts()))

    @init_model_constructor
    def train(self):
        """
        Estimate the specified parameters.

        Arguments:
            verbose: When true, verbosely print info to stdout.
        """
        def generic_reading_measure(param_list, *params):
            # Construct kwargs from param_list and provided parameters
            if len(param_list) != len(params):
                raise Exception("Dimension mismatch between parameter name "
                                "list and parameters for "
                                "generic_reading_measure")
            kwargs = dict(zip(param_list, params))

            try:
                intercept = kwargs['intercept']
                # Since intercept is not a model parameter, delete it from the
                # model arguments
                del(kwargs['intercept'])
                print("intercept: {}".format(intercept))
            except KeyError:
                intercept = 0

            # Add default parameters when specified
            # TODO: warn when default_param_values and params_to_train overlap
            if self.default_param_values:
                kwargs.update(self.default_param_values)

            # Return really bad results when param_max is exceded. We assume we
            # cannot run a model with a parameter exceeding param_max
            if True in [x > self.param_max for x in params]:
                return np.full(len(Y), float("inf"))

            gen = self.model_constructor.model_generator(**kwargs)
            m = self.measurer.measure(gen) + intercept
            if self.verbose:
                print(Y - m)
            return m

        Y = self.observed_measures()

        param_name_list = self.params_to_train
        fn = functools.partial(generic_reading_measure, param_name_list)

        # Inject __name__ attribute since partial doesn't seem to do this...
        # Could be related to this:
        # https://stackoverflow.com/questions/20594193/dynamic-create-method-and-decorator-got-error-functools-partial-object-has-no
        # or could be intended behaviour. Either way, theano needs a __name__
        fn.__name__ = generic_reading_measure.__name__

        theano_op = FromFunctionOp(fn, [T.dscalar] * len(param_name_list),
                                   [T.dvector], None)

        with self.mc_model:

            mu = theano_op(*[self.create_distr(x) for x in param_name_list])

            pymc3.Normal("Y_obs", mu=mu, sd=self.create_distr("sigma"),
                         observed=Y)

            # step = pymc3.Slice(self.mc_model.vars)
            # trace = pymc3.sample(800, step, njobs=1, init='MAP')
            # traceframe = pd.DataFrame.from_dict({name: trace[name] for name in
            #                                      [param_name_list] +
            #                                      ["sigma"]})
            # print(traceframe)
            # traceframe.to_csv("output_parametersearch.csv", sep=",",
            #                   encoding="utf-8")
            # pymc3.summary(trace)

        map_est = pymc3.find_MAP(model=self.mc_model,
                                 fmin=scipy.optimize.fmin_powell)

        print(map_est)

    @init_model_constructor
    def collect_results(self, **params):
        v = self.measurer.verbose
        self.measurer.verbose = False
        if 'intercept' in params:
            intercept = params['intercept']
            del(params['intercept'])
        else:
            intercept = 0
        print(params)
        m = self.measurer.measure(
            self.model_constructor.model_generator(**params))
        self.measurer.verbose = v
        return m + intercept

    def plot_results(self, **params):
        o = self.observed_measures()
        r = self.collect_results(params)
        x = np.arange(len(o))
        plt.bar(x, (o - r))
        plt.show()


if __name__ == "__main__":
    model_args = {"gui": True, "subsymbolic": True}
    # results = {'eye_mvt_scaling_parameter': 0.002377572678935413,
    #            'eye_mvt_angle_parameter': 0.8539963243897579,
    #            'rule_firing': 0.05995633050087895}



    # basic_measurer = EventMeasurer("KEY PRESSED: SPACE", True)

    # basic_trainer = Trainer("data/fillers.txt", "data/results_fillers_RTs.csv",
    #                         basic_measurer, model_args=model_args,
    #                         verbose=True, mode="full")

    # basic_trainer.train()

    # The results of the basic training
    results = {'eye_mvt_angle_parameter': 0.8778596227717579,
               'rule_firing': 0.05996869897845805,
               'eye_mvt_scaling_parameter': 0.002454310962708778}

    advanced_measurer = EventIntervalMeasurer(("RULE SELECTED: lexeme "
                                               "retrieved (noun): "
                                               "start reference retrieval"),
                                              ("RULE FIRED: reference"
                                               " retrieved"), True)

    advanced_trainer = Trainer("data/target_sentences.txt",
                               "data/pronouns_RTs.csv", advanced_measurer,
                               model_args=model_args, verbose=True,
                               default_param_values=results)

    # advanced_trainer.train()

    # # results of the advanced training match-mis + mis-match
    # results = {
    #     'latency_exponent': 0.23459438771567767,
    #     'latency_factor': 0.07025957880534615,
    #     'eye_mvt_angle_parameter': 0.8778596227717579,
    #     'rule_firing': 0.05996869897845805,
    #     'eye_mvt_scaling_parameter': 0.002454310962708778,
    #     'intercept': 0.14036324690675073
    # }

    # results of the advanced training match-mis
    results = {
        'latency_factor': 0.07169216002195634,
        'latency_exponent': 0.39318476021560095,
        'eye_mvt_angle_parameter': 0.8778596227717579,
        'rule_firing': 0.05996869897845805,
        'eye_mvt_scaling_parameter': 0.002454310962708778,
        'intercept': 0.15135767246512238
    }

    t = advanced_trainer

    r = t.collect_results(**results)
    # print(t.observed_measures())
    print(r - t.observed_measures())
    print("Max error of {}".format(np.max(np.abs(r - t.observed_measures()))))
    print("Mean error of {}.".format(np.mean(np.abs(r -
                                                    t.observed_measures()))))
#     t.plot_results(**results)

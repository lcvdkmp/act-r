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

# XXX: estimate only case: mismatch - match
# without spreading
# add intercept


def run_and_log(sim, fn="output"):
    with open(fn, "w") as f:
        while True:
            try:
                sim.step()
                f.write(str(sim.current_event))
                f.write("\n")
            except simpy.core.EmptySchedule:
                break


class Trainer():

    param_distributions = {
        "eye_mvt_scaling_parameter": (pymc3.HalfNormal, {"sd": 1.0}),
        "eye_mvt_angle_parameter":   (pymc3.Gamma, {"alpha": 1.3, "beta": 2}),
        "rule_firing":               (pymc3.Uniform, {"lower": 0.022,
                                                      "upper": 0.1}),

        "sigma":                     (pymc3.HalfNormal, {"sd": 0.5}),

        # "intercept":                 (pymc3.Uniform, {"lower": 0.1,
        #                                               "upper": 0.4}),
        # XXX: estimate latency_factor, latency_exponent
        # "latency_factor":           (pymc3.Gamma, {"alpha": 2, "beta": 6}),
        # "latency_exponent":         (pymc3.HalfNormal, {"sd": 0.5})
    }

    def __init__(self, sentence_filepath, word_freq_filepath, measurer,
                 model_args={}, verbose=False, mode="noun",
                 default_param_values=None):

        self.mode = mode
        self.default_param_values = default_param_values

        self.mc_model = pymc3.Model()
        self.model_constructor = ModelConstructor(sentence_filepath,
                                                  word_freq_filepath,
                                                  advanced=(self.mode ==
                                                            "noun"),
                                                  noun_mode=(self.mode ==
                                                             "noun"),
                                                  **model_args)


        # self.eye_only_results = {'eye_mvt': np.exp(-8.814191179269775),
        #                          'eye_map': np.exp(-2.302589265832653)}

        # self.results = {'eye_mvt_scaling_parameter': 0.002200098239712775,
        #                 'eye_mvt_angle_parameter': 0.41134056484219267,
        #                 'rule_firing': 0.060418825445918777}
        self.results = {'eye_mvt_scaling_parameter': 0.002377572678935413,
                        'eye_mvt_angle_parameter': 0.8539963243897579,
                        'rule_firing': 0.05995633050087895}

        # The maximum value any parameter can have in order for the model to
        # still work. This is a very rough lower bound
        self.max_param = 100

        self.verbose = verbose
        self.measurer = measurer

        # NOTE: you can inspect a certain simulation iteration of the training
        # procedure in the following way:
        # self.measurer.add_callback((0, 0), lambda x: x.run())
        # ln = "log_0_2.txt"
        # self.measurer.add_callback((0, 0), functools.partial(run_and_log, fn=ln))

    def params_to_train(self):
        """
        Returns:
            A list of parameter names that should be trained in the
            current trainer mode
        """
        params = {"eye_only": ["eye_mvt_scaling_parameter",
                               "eye_mvt_angle_parameter"],
                  "full": ["eye_mvt_scaling_parameter",
                           "eye_mvt_angle_parameter",
                           "rule_firing"],
                  "noun": ["latency_factor",
                           "latency_exponent",
                           "intercept"]}

        return params[self.mode]

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

    def observed_measures(self):
        return np.array(list(self.model_constructor.freqs()))

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
                del(kwargs['intercept'])
                print("intercept: {}".format(intercept))
            except KeyError:
                intercept = 0

            # Add default parameters when specified
            # TODO: warn when default_param_values and params_to_train overlap
            if self.default_param_values:
                kwargs.update(self.default_param_values)

            # Return really bad results when max_param is exceded. We assume we
            # cannot run a model with a parameter exceeding max_param
            if True in [x > self.max_param for x in params]:
                return np.full(len(Y), float("inf"))

            gen = self.model_constructor.model_generator(**kwargs)
            m = self.measurer.measure(gen) + intercept
            if self.verbose:
                print(Y - m)
            return m

        Y = self.observed_measures()

        param_name_list = set(self.params_to_train())
        fn = functools.partial(generic_reading_measure, param_name_list)

        # Inject __name__ attribute since partial doesn't seem to do this...
        # Could be related to this:
        # https://stackoverflow.com/questions/20594193/dynamic-create-method-and-decorator-got-error-functools-partial-object-has-no
        # or could be intended behaviour. Either way, theano needs a __name__
        fn.__name__ = generic_reading_measure.__name__

        # Theano also requires a __module__ for the op if multithreading is
        # used.
        # print(generic_reading_measure.__module__)
        # fn.__module__ = "model"

        theano_op = FromFunctionOp(fn, [T.dscalar] * len(param_name_list),
                                   [T.dvector], None)

        with self.mc_model:

            mu = theano_op(*[self.create_distr(x) for x in param_name_list])
            # if "intercept" in self.param_distributions:
            #     intercept = self.create_distr("intercept")
            #     mu += intercept

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

    def collect_results(self, **params):
        v = self.measurer.verbose
        self.measurer.verbose = False
        m = self.measurer.measure(
            self.model_constructor.model_generator(**params))
        self.measurer.verbose = v
        return m

    def plot_results(self, **params):
        o = self.observed_measures()
        r = self.collect_results(params)

        # bw = 0.35
        x = np.arange(len(o))
        # fig, ax = plt.subplots()
        # ax.bar(x, o, bw, color='b')
        # ax.bar(x + bw, r, color='r')
        # ax.set_xticks(x + bw / 2)
        plt.bar(x, (o - r))
        plt.show()


if __name__ == "__main__":
    model_args = {"gui": True, "subsymbolic": True}
    results = {'eye_mvt_scaling_parameter': 0.002377572678935413,
               'eye_mvt_angle_parameter': 0.8539963243897579,
               'rule_firing': 0.05995633050087895}
    basic_measurer = EventMeasurer("KEY PRESSED: SPACE", True)

    basic_trainer = Trainer("data/fillers.txt", "data/results_fillers_RTs.csv",
                            basic_measurer, model_args=model_args,
                            verbose=True, mode="full")

    basic_trainer.train()

    # advanced_measurer = EventIntervalMeasurer(("RULE SELECTED: lexeme "
    #                                            "retrieved (noun): "
    #                                            "start reference retrieval"),
    #                                           ("RULE FIRED: reference"
    #                                            " retrieved"), True)

    # advanced_trainer = Trainer("data/target_sentences.txt",
    #                            "data/pronouns_RTs.csv", advanced_measurer,
    #                            model_args=model_args, verbose=True,
    #                            default_param_values=results)

    # advanced_trainer.train()

#     r = t.collect_results(**results)
#     # print(t.observed_measures())
#     print(r - t.observed_measures())
#     print("Max error fo {}".format(np.max(np.abs(r - t.observed_measures()))))
#     print("Mean error of {}.".format(np.mean(np.abs(r -
#                                                     t.observed_measures()))))
#     t.plot_results(**results)

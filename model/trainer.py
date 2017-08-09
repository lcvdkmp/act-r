from theano.compile.ops import FromFunctionOp
import pymc3
import numpy as np
from model_constructor import ModelConstructor
import theano.tensor as T
import simpy.core
import scipy

import matplotlib.pyplot as plt
import functools

#       - eye_mvt
#           half normal sd=0.5
#       - eye_map
#           gamma as=0.7
#       - rule_firing (with and without)
#           half normal sd=0.5


def run_and_log(sim, fn):
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
        "eye_mvt_scaling_parameter": (pymc3.HalfNormal, {"sd": 0.5}),
        "eye_mvt_angle_parameter": (pymc3.Gamma, {"alpha": 1, "beta": 3}),
        "sigma": (pymc3.HalfNormal, {"sd": 1.0})
    }

    def __init__(self, sentence_filepath, word_freq_filepath, model_args={}):
        self.mc_model = pymc3.Model()
        self.model_constructor = ModelConstructor(sentence_filepath,
                                                  word_freq_filepath,
                                                  advanced=False,
                                                  **model_args)

        self.mode = "eye_only"
        self.counter = [0, 0]
        self.inspect_number = [-1, 0]

        self.eye_only_results = {'eye_mvt': np.exp(-8.814191179269775),
                                 'eye_map': np.exp(-2.302589265832653)}

    def params_to_train(self):
        params = {"eye_only": ["eye_mvt_scaling_parameter",
                               "eye_mvt_angle_parameter"]}

        return params[self.mode]

    def create_distr(self, name):
        p = self.param_distributions[name]
        return p[0](name, **p[1])

    def observed_measures(self):
        return np.array(list(self.model_constructor.freqs()))

    def measure(self, gen, verbose=False):
        measures = []
        printed_params = not verbose
        for m in gen:
            if not printed_params:
                print(m.model_params)
                printed_params = True

            if self.counter == self.inspect_number:
                sim = m.sim()
                sim.run()
                run_and_log(sim, "log_{}.{}.txt".format(self.counter[0],
                                                        self.counter[1]))

            if verbose:
                print("*" * 30, "sim {}.{}".format(self.counter[0],
                                                   self.counter[1]), "*" * 30)
            self.counter[1] += 1
            sim = m.sim()

            s = m.sentence_pairs[0][0] + m.sentence_pairs[0][1]
            dts = list(self.sim_event_dt(sim, "KEY PRESSED: SPACE"))
            assert(len(dts) == len(s))
            measures += dts

        if verbose:
            print("=" * 80)
        self.counter[0] += 1
        self.counter[1] = 0
        return np.array(measures)

    def train(self, verbose=False):

        def generic_reading_measure(param_list, *params):
            # Construct kwargs from param_list and provided parameters
            if len(param_list) != len(params):
                raise Exception("Dimension mismatch between parameter name "
                                "list and parameters for "
                                "generic_reading_measure")
            kwargs = dict(zip(param_list, params))
            gen = self.model_constructor.model_generator(**kwargs)
            m = self.measure(gen, verbose=verbose)
            if verbose:
                print(Y - m)
            return m

        Y = self.observed_measures()

        param_name_list = self.params_to_train()
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
            # trace = pymc3.sample(500, step, njobs=1, init='MAP')
            # pymc3.summary(trace)

        map_est = pymc3.find_MAP(model=self.mc_model,
                                 fmin=scipy.optimize.fmin_powell)

        print(map_est)

    def sim_event_dt(self, sim, event):
        t = sim.show_time()
        while True:
            try:
                sim.step()
            except simpy.core.EmptySchedule:
                break

            if sim.current_event.action == event:
                yield (sim.show_time() - t)
                t = sim.show_time()

    def collect_results(self, **params):
        return self.measure(
            self.model_constructor.model_generator(**params))

    def plot_results(self):
        o = t.observed_measures()
        r = t.collect_results(
            eye_mvt_scaling_parameter=t.eye_only_results['eye_mvt'],
            eye_mvt_angle_parameter=t.eye_only_results['eye_map'])

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
    t = Trainer("data/fillers.txt", "data/results_fillers_RTs.csv",
                model_args=model_args)
    # print(t.reading_measure(1, 1, 1))
    t.train(verbose=True)
    # r = t.collect_results(
    #     eye_mvt_scaling_parameter=t.eye_only_results['eye_mvt'],
    #     eye_mvt_angle_parameter=t.eye_only_results['eye_map'])
    # # print(t.observed_measures())
    # print(np.max(np.abs(r - t.observed_measures())))
    # print("Mean error of {}.".format(np.mean(np.abs(r -
    #                                                 t.observed_measures()))))
    # print(r)
    # t.plot_results()

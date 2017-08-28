from collections import defaultdict
import numpy as np
import simpy


class Measurer():

    counter = [0, 0]
    callbacks = defaultdict(list)

    def __init__(self, verbose=False):
        self.verbose = verbose
        pass

    def add_callback(self, count, callback):
        self.callbacks[tuple(count)].append(callback)

    def measure(self, gen):
        models = list(gen)
        if self.verbose and len(models) > 0:
            print(models[0].model_params)

        ar = np.array(sum(map(self.measure_model_wrapper, models), []))
        self.counter[0] += 1
        self.counter[1] = 0
        if self.verbose:
            print("=" * 80)
        return ar

    def measure_model_wrapper(self, model):
        if tuple(self.counter) in self.callbacks:
            s = model.sim()
            for f in self.callbacks[tuple(self.counter)]:
                f(s)

        if self.verbose:
            print("*" * 30, "sim {}.{}".format(self.counter[0],
                                               self.counter[1]), "*" * 30)
        self.counter[1] += 1

        return self.measure_model(model)

    def measure_model(self, model):
        pass


class EventMeasurer(Measurer):
    def __init__(self, event, verbose=False):
        self.event = event
        super(Measurer, self)
        self.verbose = verbose

    def measure_model(self, m):
        sim = m.sim()

        s = m.sentence_pairs[0][0] + m.sentence_pairs[0][1]
        dts = list(self.sim_event_dt(sim, self.event))
        assert(len(dts) == len(s))

        return dts

    def sim_event_dt(self, sim, event):
        """
        Run a pyactr simulation and measure the time between certain events.

        Arguments:
            sim: The simulation to run.
            event: The event of which delta time is measured.

        Returns:
            Yields delta time between events matching event
        """

        t = sim.show_time()
        while True:
            try:
                sim.step()
            except simpy.core.EmptySchedule:
                break

            if sim.current_event.action == event:
                yield (sim.show_time() - t)
                t = sim.show_time()


class EventIntervalMeasurer(Measurer):
    def __init__(self, ev1, ev2, verbose=False):
        self.events = (ev1, ev2)
        super(Measurer, self)
        self.verbose = verbose

    def measure_model(self, m):
        sim = m.sim()
        return self.sim_event_interval_dt(sim, self.events)

    def sim_event_interval_dt(self, sim, events):
        while True:
            times = (None, None)
            try:
                sim.step()
            except simpy.core.EmptySchedule:
                break

            # Skip ev0s after first ev0 occurred since we are waiting for
            # an ev1
            if sim.current_event.action == self.events[0] and \
                    times[0] is None:
                times[0] = sim.show_time()

            # We want to record the first occurance of a ev1 after a ev0
            if sim.current_event.action == self.events[1] and \
                    times[0] is not None and \
                    times[1] is None:
                times[1] = sim.show_time()

            if None in times:
                raise Exception("proper ev0, ev1 interval did not occur!")
            return times[1] - times[0]

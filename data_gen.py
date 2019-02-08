import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s %(asctime)s %(message)s')


base_vnf_profiles = {
     'low':  {'MME':  [17.7, 15.9, 5.8],
              'SGW':  [0.7, 0.3, 0.14],
              'HSS':  [0.9, 1.1, 0.7],
              'PCRF': [1.2, 0.6, 0.5],
              'PGW':  [1.7, 2.1, 0.8]},

     'high': {'MME':  [2.9, 3.8, 1.9],
              'SGW':  [79.1, 3.3, 91.2],
              'HSS':  [2.9, 4.5, 1.3],
              'PCRF': [1.9, 3.9, 0.9],
              'PGW':  [53.1, 37.2, 92]}
}


class VNF_Profile:
    def __init__(self, measures, aff_group=-1):
        self.measures = measures
        self.aff_group = aff_group


class TSEntry:
    def __init__(self, entry, time):
        # entry should be np.array
        self.entry = entry
        self.time = time
        self.next = None
        self.prev = None


class TimeSeries:
    def __init__(self, init_entry):
        self.first = TSEntry(init_entry, 0)
        self.last = self.first
        self.shape = init_entry.shape
        self.time_elapsed = 0

    def add_entry_delta(self, delta):
        new_entry = TSEntry(cap_profiles(self.last.entry+delta), self.time_elapsed+1)
        new_entry.prev = self.last
        self.last.next = new_entry

        self.last = new_entry
        self.time_elapsed += 1


class Simulation:
    # generates time series for each vnf
    def __init__(self, init_profiles=None, steps=1000, std=0.1):
        self.logger = logging.getLogger('Simulation{}'.format(std))
        self.logger.info('Initialising simulation...')
        if not init_profiles:
            init_profiles = gen_init_profiles(base_vnf_profiles['high'], 200)

        self.logger.info('Generating time series...')
        # timeseries of each vnf compute needs
        self.time_series = TimeSeries(init_profiles)
        for step in range(steps):
            delta = np.random.normal(0, std, self.time_series.shape)
            self.time_series.add_entry_delta(delta)

        self.logger.info('Time series generated!')

        # create default gravity centres for ekm based on base vnf profiles
        self.default_centres = list()
        for key in base_vnf_profiles['high']:
            self.default_centres.append(base_vnf_profiles['high'][key])
        self.default_centres = np.array(self.default_centres)

        # centres evolution will contain evolution of centres of gravity after running k-means
        self.centres_evolution = list()

        # Q-Learning table, to be filled during first run of simulation and then further updated
        self.q_table = np.zeros((len(init_profiles)+1, 21))
        # Number of times each Q-Table state was visited
        self.num_visited = np.zeros(len(init_profiles)+1)

        self.logger.info('Simulation initialised!')

    def run_ekm(self, init_centres=None, points=None):
        if init_centres is None:
            init_centres = self.default_centres
        if points is None:
            points = self.time_series.last.entry
        self.logger.info('Running ekm with {} clusters and {} points...'.format(len(init_centres), len(points)))
        centres = np.array(init_centres)
        steps = 0
        aff_groups = [-1]*len(points)
        converged = False
        while not converged:
            granularity = max(1e-10, 100*steps**(np.sqrt(len(points)))/2.0**(len(points)))
            for i in range(len(points)):
                min_dist = 1e10
                for j in range(len(centres)):
                    dist = np.linalg.norm(points[i]-centres[j])
                    if dist < min_dist:
                        min_dist = dist
                        aff_groups[i] = j
            new_centres = self.calc_centres(points, aff_groups, len(centres))

            centres = self.snap_to_grid(centres, granularity)
            new_centres = self.snap_to_grid(new_centres, granularity)

            if np.all(np.equal(new_centres, centres)):
                converged = True
            centres = new_centres
            steps += 1
        self.logger.info('ekm converged in {} steps'.format(steps))
        return steps, centres, aff_groups, points

    def run_sim(self, centres=None, surv_epoch=1):
        self.logger.info('Running simulation with surveillance epoch of {}t...'.format(surv_epoch))
        if centres is None:
            centres = np.array([[75, 75, 75], [25, 25, 25]])

        ts_entry = self.time_series.first

        steps, centres, old_aff_groups, points = self.run_ekm(init_centres=centres, points=ts_entry.entry)

        num_deviations = list()
        surv_epoch_lengths = list()

        num_aff_groups = list()

        while ts_entry:
            num_aff_groups.append(len(centres))
            if ts_entry.time % surv_epoch == 0:

                steps, centres, aff_groups, points = self.run_ekm(init_centres=centres, points=ts_entry.entry)
                num_deviations.append(np.sum(np.not_equal(old_aff_groups, aff_groups)))
                surv_epoch_lengths.append(surv_epoch)

                if ts_entry.time >= 1:
                    # action taken is the difference between epoch lengths
                    action_taken = surv_epoch_lengths[-1] - surv_epoch_lengths[-2]
                    # we were in a state 'num_deviations[-2]', took action and ended up in 'num_deviations[-1]'
                    self.update_q_table(num_deviations[-2], action_taken, num_deviations[-1], surv_epoch_lengths[-1])

                action = self.get_action(num_deviations[-1])
                surv_epoch += action - 10
                surv_epoch = max(surv_epoch, 1)
                self.logger.info('Time: {} Action: {} Surv epoch: {}'.format(ts_entry.time, action-10, surv_epoch))

                if num_deviations[-1] > 0:

                    # 2 affinity groups is the lower bound
                    if len(centres) > 2:
                        centres = centres[:-1]
                        self.logger.info('Decreasing affinity groups to {}'.format(len(centres)))

                        # rerun ekm with new centres
                        steps, centres, aff_groups, points = self.run_ekm(init_centres=centres, points=ts_entry.entry)

                    # reset deviation flag
                    no_deviation = False

                # no deviation occured for 2 consecutive epochs
                elif ts_entry.time >= 2 and num_deviations[-1] == 0 and num_deviations[-2] == 0:
                    # pick a random vnf profile to act as a new centre
                    cid = np.random.randint(len(ts_entry.entry))
                    centres = np.append(centres, [ts_entry.entry[cid]], axis=0)
                    self.logger.info('Increasing affinity groups to {} with a new center at {}'.
                                     format(len(centres), centres[-1]))

                    # rerun ekm with new centres
                    steps, centres, aff_groups, points = self.run_ekm(init_centres=centres, points=ts_entry.entry)

                    # reset deviation flag
                    no_deviation = False

                old_aff_groups = aff_groups
            ts_entry = ts_entry.next
        self.logger.info('Simulation finished!')
        self.logger.info('FINAL STATS Number of affinity groups: {}'.format(len(centres)))
        return np.array(num_aff_groups)

    def get_action(self, state, random_prob=0.5):
        is_random = (np.random.uniform(0.0, 1.0) < random_prob)
        # return random action (for exploration purposes)
        if is_random:
            return np.random.randint(0, 21)
        # otherwise return the best action
        best_action = 0
        for action in range(len(self.q_table[state])):
            if self.q_table[state, action] > self.q_table[state, best_action]:
                best_action = action
        return best_action

    def get_reward(self, num_deviations, surv_epoch_length, beta=0.5):
        # ensure reward function is able to deal with 0 deviations
        if num_deviations == 0:
            num_deviations = 0.5
        return surv_epoch_length/(num_deviations**beta)

    def update_q_table(self, state, action, next_state, surv_epoch_length, learning_rate=0.5, discount_rate=0.9):
        q_max = 0
        # loop through the actions in next state to find max reward
        for reward in self.q_table[next_state]:
            q_max = max(q_max, reward)

        self.q_table[state, action] = (1-learning_rate)*self.q_table[state, action] + \
            learning_rate*(self.get_reward(state, surv_epoch_length) + discount_rate*q_max)

    def calc_centres(self, points, point_group, num_centres):
        counts = [0]*num_centres
        centres = [[0]*len(points[0]) for _ in range(num_centres)]
        for pid in range(len(points)):
            pg = point_group[pid]
            counts[pg] += 1
            centres[pg] += points[pid]
        for cid in range(len(centres)):
            if counts[cid] > 0:
                centres[cid] /= counts[cid]
        centres = np.array(centres)
        return np.array(centres)

    def snap_to_grid(self, points, grid_step):
        grid_points = list()
        for point in points:
            gp = list()
            for coord in point:
                gp.append(min(100.0, np.round(coord/grid_step)*grid_step))
            grid_points.append(gp)
        return np.array(grid_points)


def cap_profiles(profiles):
    for prof in profiles:
        for i in range(len(prof)):
            prof[i] = min(prof[i], 100)
    return profiles


def gen_init_profiles(base_profiles, count_per_base):
    profiles = list()
    for key in base_profiles:
        base = base_profiles[key]
        for i in range(count_per_base):
            profile = (np.random.pareto(1.16, size=len(base)) + 1) + np.array(base)
            profiles.append(profile)
    profiles = np.array(cap_profiles(profiles))
    return profiles


def group_points(points, group_ids):
    groups = dict()
    for i in range(len(points)):
        gid = group_ids[i]
        if gid not in groups:
            groups[gid] = list()
        groups[gid].append(points[i])
    for gid in groups:
        groups[gid] = np.array(groups[gid])
    return groups


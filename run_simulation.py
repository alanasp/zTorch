import sys
import multiprocessing as multiproc
import custom_logger
import datetime

import ztorch_simulation as zsim


if __name__ == '__main__':

    start_time = datetime.datetime.utcnow()

    logger = custom_logger.get_logger('Run_Simulations')
    logger.info('Starting simulations...')

    num_time_steps = 100000
    if len(sys.argv) > 1:
        num_time_steps = int(sys.argv[1])

    on_the_fly = True
    if num_time_steps is not None and len(sys.argv) > 2:
        on_the_fly = bool(sys.argv[2])

    # (std, num_vnf_profiles, num_time_steps, output_file_prefix, input_file_prefix)
    params = [
        #{
        #    'std': 0.50,
        #    'num_init_profiles': 100,
        #    'steps': num_time_steps,
        #    'input_file': not on_the_fly,
        #    'on_the_fly': on_the_fly
        #},
        {
            'std': 0.10,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        #{
        #    'std': 0.50,
        #    'num_init_profiles': 1000,
        #    'steps': num_time_steps,
        #    'input_file': not on_the_fly,
        #    'on_the_fly': on_the_fly
        #},
        {
            'std': 0.06,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        {
            'std': 0.08,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
        {
            'std': 0.12,
            'num_init_profiles': 1000,
            'steps': num_time_steps,
            'input_file': not on_the_fly,
            'on_the_fly': on_the_fly
        },
    ]

    procs = []

    for param in params:
        sim = zsim.Simulation(**param)
        proc = multiproc.Process(target=sim.run_sim)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    end_time = datetime.datetime.utcnow()
    delta = int((end_time - start_time).seconds)
    logger.info('Simulations finished in {}h {}min {}s'.format(delta//3600, (delta % 3600)//60, delta % 60))

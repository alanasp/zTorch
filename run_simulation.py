import sys
import multiprocessing  as multiproc
import custom_logger
import datetime

import ztorch_simulation as zsim


if __name__ == '__main__':

    start_time = datetime.datetime.utcnow()

    logger = custom_logger.get_logger('Run_Simulations')
    logger.info('Starting simulations...')

    num_time_steps = 1000
    if len(sys.argv) > 1:
        num_time_steps = int(sys.argv[1])

    # (std, num_vnf_profiles, num_time_steps, output_file_prefix, input_file_prefix)
    params = [(0.1, 750, num_time_steps, None, True), (0.1, 1000, num_time_steps, None, True), (0.1, 1250, num_time_steps, None, True),
              (0.06, 1000, num_time_steps, None, True), (0.08, 1000, num_time_steps, None, True), (0.12, 1000, num_time_steps, None, True)]

    procs = []

    for param in params:
        sim = zsim.Simulation(*param)
        proc = multiproc.Process(target=sim.run_sim)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    end_time = datetime.datetime.utcnow()
    delta = int((end_time - start_time).seconds)
    logger.info('Simulations finished in {}h {}min {}s'.format(delta//3600, (delta % 3600)//60, delta % 60))

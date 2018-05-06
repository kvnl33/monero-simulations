from multiprocessing import Process, Pool
import sim_2017_03_18, sim_2018_03_29, counterm
import argparse, time

parser = argparse.ArgumentParser(description='Monero Simulations')
parser.add_argument('-p', action="store", dest="threads", type=int, required=True)
parser.add_argument('-f', action="store", dest="filename", type=str, required=True)
args = parser.parse_args()

def main():
    maxthreads = args.threads
    filename = args.filename

    pool = Pool(processes=maxthreads, maxtasksperchild=1)
    jobs = []

    for mixin in range(1,16):
        for period in [0,12]:
            proc = pool.apply_async(sim_2017_03_18.sim, args=(100000, mixin, period,'0.10', True))
            jobs.append(proc)

    for mixin in range(1,16):
        for ver in ['0.10', '0.11']:
            proc = pool.apply_async(sim_2018_03_29.sim, args=(100000, mixin, ver, True))
            jobs.append(proc)

    for mixin in range(1,16):
        for year in ['2017', '2018']:
            for period in [0,12]:
                if not (year =='2018' and period == 12):
                    proc = pool.apply_async(counterm.sim, args=(100000, mixin, period, year, True))
                    jobs.append(proc)

    while(not all([p.ready() for p in jobs])):
        time.sleep(5)

    counterm.graph_anonset(filename)

if __name__ == '__main__':
    main()

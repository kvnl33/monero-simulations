# Simulation of countermeasures for the Monero selection protocol
import sys, sqlite3, matplotlib, collections, bisect, itertools, os
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Kevin Lee and Andrew Miller"
__maintainer__ = "Kevin Lee"
__email__ = "klee160@illinois.edu"

# define some global variables
CRYPTONOTE_MINED_MONEY_UNLOCK_WINDOW = 60
CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE = 10
top06mo = 1500203268
top12mo = 1515755268
rct_start = 1484051946

if 'top_time' not in globals():
    top_time = None
    top_idx = {}
    time_diff = []
    time_dict = collections.OrderedDict()
    time_dict_keys = []

def extrap(x, xp, yp):
    '''The function adds linear extrapolation to the np.interp function. It is called
    in extrapolation simulations in which we want to measure top block heights in the future.
    '''
    y = np.interp(x, xp, yp)
    y = np.where(x<xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x>xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]), y)
    return y

def preprocess(time):
    '''Sets up the variables in the simulation. The most recent timestamp in the database is
    set to be the time of our simulation, the maximum global index is found. We keep track of that in 
    our simulation. Afterwards, we read in the recent and top zones for 6 and 12 month simulations to use.
    Lastly, we read the time differences between transaction time and output creation (receive) time for the
    transactions that were spent in-the-clear, because those will define our simulation behavior for choosing
    the real spend.
    '''
    global top_time, top_idx, time_diff, time_dict, time_dict_keys
    print 'Running preprocess first'
    if time=='2017':
        conn_1 = sqlite3.connect('outs_2017_03_18.db')
    else:
        conn_1 = sqlite3.connect('outs_2018_03_29.db')
    c_1 = conn_1.cursor()
    c_1.execute('''SELECT MAX(block_height) FROM out_table''')
    top_block, = c_1.fetchone()

    cmd = '''SELECT timestamp FROM out_table WHERE block_height = {ht}'''
    cmd = cmd.format(ht = top_block)
    c_1.execute(cmd)
    timenow, = c_1.fetchone()
    top_time = timenow

    if time == "2017":
        cmd = '''SELECT MAX(g_idx) as idx from out_table WHERE amount = {denom}'''
        cmd = cmd.format(denom = 0)
    else:
        cmd = '''SELECT MAX(g_idx) as idx from out_table'''
    c_1.execute(cmd)
    top = c_1.fetchone()
    top_idx[0] = {}
    top_idx[0][0] = top[0]

    if time == "2017":        
        cmd = '''SELECT timestamp, MAX(g_idx) FROM out_table WHERE amount={denom} GROUP BY timestamp'''
        cmd = cmd.format(denom = 0)
    else:
        cmd = '''SELECT timestamp, MAX(g_idx) FROM out_table GROUP BY timestamp'''
    c_1.execute(cmd)
    result = c_1.fetchall()
    x = np.asarray([int(row[0]) for row in result])
    y = np.asarray([int(row[1]) for row in result])
    for row in result:
    	time_dict[row[0]] = row[1]

    if time == '2017':
        if os.path.isfile("extrap_time_dict.npz"):
            npzfile = np.load("extrap_time_dict.npz")
            ts = npzfile["ts"]
            g = npzfile["g"]
        else:
            ts, g = ([] for i in range(2))
            for f in range(top_time+1, top12mo+1):
                ts.append(f)
                done = int(extrap(f,x,y))
                g.append(done)
            ts = np.asarray(ts)
            g = np.asarray(g)
            outfile = "extrap_time_dict"
            np.savez(outfile, ts = ts, g = g)
        for a,b in itertools.izip(ts,g):
            time_dict[a] = b
        top_idx[0][6] = time_dict[top06mo]
        top_idx[0][12] = time_dict[top12mo]
    time_dict_keys = time_dict.keys()

    conn = sqlite3.connect('zinput.db')
    c_2 = conn.cursor()
    cmd = '''SELECT tx_timestamp - mixin_timestamp as time_diff from first'''
    c_2.execute(cmd)
    diff = c_2.fetchall()
    for row in diff:
        if row[0] >= 0:
            time_diff.append(row[0])
    time_diff = np.asarray(time_diff)

def fetch_real_output(top_tm, top_idx):
    '''Randomly selects an output to be used as the real spend. A time difference is
    selected from our list of zero-input transactions we collected from the blockchain. Next,
    using the time difference, we find the closest global index if the block is in the 
    dictionary. We normalize the index returned by dividing the 
    real index by the top index (mixin offset). For instance, if the top index is Y
    and the chosen index is X, we return X/Y.
    '''
    while True:
        rhd = np.random.choice(time_diff)
        rb = top_tm - rhd
        if rb >= rct_start:
            break
    closest = bisect.bisect_left(time_dict_keys, rb)
    real_g_idx = time_dict[time_dict_keys[closest]]
    return float(real_g_idx)/top_idx

def fetch_mixin(top_tm, time_back, top_idx):
    '''Selects mixins based on time back. We find the closest global index if the block is in the 
    dictionary, otherwise find closest. We normalize the index returned by dividing the 
    real index by the top index (mixin offset). For instance, if the top index is Y
    and the chosen index is X, we return X/Y.
    '''
    rb = top_tm - time_back
    if rb < rct_start: 
        return None
    closest = bisect.bisect_left(time_dict_keys, rb)
    avail_idx = time_dict[time_dict_keys[closest]]
    return float(avail_idx)/top_idx      

def sample_mixins(amount, num_mix, period, is_rct=True):
    '''Our proposed mixin-sampling protocol samples mixins from a gamma distribution
    over the mixin set. We do not sample more mixins that necessary. We normalize the 
    mixin returned by dividing the real index by the top index. For instance, if 
    the top index is Y and the chosen index is X, we return X/Y.
    '''
    mixin_vector = []
    top_sim_time = -1
    if period == 0:
    	top_sim_time = top_time
    elif period ==  6:
    	top_sim_time = top06mo
    elif period == 12:
    	top_sim_time = top12mo
    assert top_sim_time >= 0
    top_global_idx = top_idx[amount][period]
    real = fetch_real_output(top_sim_time, top_global_idx)
    num_found = 0 
    while (num_found < num_mix):
        time_back = np.exp(np.random.gamma(shape=19.28, scale=1/1.61))+120
        candidate = fetch_mixin(top_sim_time, time_back, top_global_idx)
        if candidate is None:
        	continue
        if candidate not in mixin_vector and candidate != real:
            mixin_vector.append(candidate)
            num_found += 1
    return real, mixin_vector

def sim(N, M, time, year, is_rct=True):
    '''The main simulation function starts the simulations. The simulation is run N times
    with M mixins chosen. We normalize the mixin returned by dividing the 
    real index by the top index. For instance, if the top index is Y and the chosen index is X, 
    we return X/Y. We will put all the percentages returned inside a vector and save it 
    externally so we can run various graphing schemes on one simulation batch in the future. 
    The simulation is run with sim(# of trials, # of mixins), 0 to sample all, 6 for six months, 12 for 12 months.
    '''
    if year == "2017":
        assert time in [0,6,12]
    else:
        assert time == 0
    preprocess(year)
    real, rest = ([] for i in range(2))
    for x in range(0,N):
        print(x+1)
        m = 0
        n, q = sample_mixins(m, M, time, is_rct)
        real.append(n)
        rest.append(q)
    real = np.asarray(real)
    rest = np.asarray(rest)
    outfile = "counterm_outfile_%d_mo_%d_mixins_%s" % (time, M, year)
    np.savez(outfile, real=real, rest=rest)

def graph_figure(M, time, version='counterm', is_rct=False):
    '''Takes in results of the simulation runs and generates a graph of all simulation runs.
    The recents, rest, and real mixins graphed separately on the same graph. These graphs can be 
    compared to the 0-mixin behavior graphs, and should be similar to Monero graphs.
    '''   
    outfile = "counterm_outfile_%d_mo_%d_mixins_%s.npz" % (time, M, version)
    npzfile = np.load(outfile)
    real = npzfile['real']
    rest = npzfile['rest']
    rest = list(itertools.chain.from_iterable(rest))    
    plt.ioff()
    plt.clf()
    plt.hist((1.0-np.array(rest), 1.0-np.array(real)), bins=np.logspace(-3,0, 200), normed=True, histtype='stepfilled', stacked=True, label=['Rest', 'Real'])
    plt.ylim(ymin=0, ymax=100)
    plt.xscale('log')
    plt.xlabel('Mixin Offset (% of available)')
    plt.ylabel('PDF')
    plt.legend()
    timestr = {0: 'Block XXX (TODO)',
               6: '6 months',
               12: '12 months'}[time]
    rct_str = '(RingCT) ' if is_rct else ''
    plt.title('%s Simulation at %s (%d Mixins, %d Trials)' % (rct_str, timestr, M, len(real)))
    plt.savefig('resultsim_%d_%d_%s_%s.png' % (M, time, version, rct_str))

def graph_guesser(filename):
    '''Does the guess-most-recent algorithm on our simulation data, and adds to a running score for each mixin.
    It also does a guess-least-recent on our simulation, and adds that to a running score of each mixin.
    At the conclusion, the worst score vs mixin total is graphed to give our countermeasure credibility.
    '''    
    plt.ioff()
    plt.clf()
    for period in [0,6,12]:
        xs = []
        ys = []
        for M in range(1,16):
            outfile = "counterm_outfile_%d_mo_%d_mixins_%s.npz" % (period, M, 'counterm')
            npzfile = np.load(outfile)
            real = npzfile['real']
            rest = npzfile['rest']
            print real.shape, rest.shape
            correct_fg = 0
            total_fg = 0
            for r1, r2 in zip(real, rest):
                total_fg += 1
                if r1 <= np.min(r2):
                    correct_fg += 1
            correct_lg = 0
            total_lg = 0
            for r1, r2 in zip(real, rest):
                total_lg += 1
                if r1 >= np.max(r2):
                    correct_lg += 1
            assert total_lg == total_fg
            if correct_lg >= correct_fg:
            	correct = correct_lg
            	total = total_lg
            else:
            	correct = correct_fg
            	total = total_fg
            xs.append(M)
            ys.append(float(correct)/total)
            print(float(correct)/total)
        if period == 0:
            label = "Proposed Protocol Now"
        else:
            label = "Proposed Protocol, %d Months" % (period)
        if period == 0:
        	line = 'b+-'
        elif period == 6:
        	line = 'b^--'
        else:
        	line = 'bs-'
        plt.plot(xs, ys, line, label=label)
    for period in [0,6,12]:
        xs = []
        ys = []     
        for M in range(1,16):
            outfile = "current_outfile_%d_mo_%d_mixins_%s.npz" % (period, M, '0.10')
            npzfile = np.load(outfile)
            real = npzfile['real']
            recents = npzfile['recents']
            rest = npzfile['rest']
            print real.shape, recents.shape, rest.shape
            correct = 0
            total = 0
            for r1, r2, r3 in zip(real, recents, rest):
                total += 1
                if r1 >= np.max(np.concatenate((r2,r3))):
                    correct += 1                
            xs.append(M)
            ys.append(float(correct)/total)
            print(float(correct)/total)
        if period == 0:
            label = "Current Protocol Now"
        else:
            label = "Current Protocol, %d Months" % (period)
        if period == 0:
        	line = 'r+-'
        elif period == 6:
        	line = 'r^--'
        else:
        	line = 'rs-'
        plt.plot(xs, ys, line, label=label)
    xs = range(1,16)
    ys = [1/float(_+1) for _ in xs]
    plt.plot(xs, ys, 'k-', label="Ideal")
    plt.xlabel('Number of Mixins')
    plt.ylabel('Fraction Correct')
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.legend(loc='best')
    plt.savefig('first_guess_counterm_v_before_rct_final.png')

def graph_anonset(filename):
    '''Does the guess-most-recent algorithm on our simulation data, and adds to a running score for each mixin.
    It also does a guess-least-recent on our simulation, and adds that to a running score of each mixin.
    At the conclusion, the 1/(worst score) vs mixin total is graphed to give our countermeasure credibility in
    terms of anonymity-set.
    '''
    plt.ioff()
    plt.clf()
    for period in [0,12]:
        xs, ys = ([] for i in range(2))
        for M in range(1,16):
            outfile = "counterm_outfile_%d_mo_%d_mixins_%s.npz" % (period, M, '2017')
            npzfile = np.load(outfile)
            real = npzfile['real']
            rest = npzfile['rest']
            print real.shape, rest.shape
            correct_fg = total_fg = 0
            for r1, r2 in zip(real, rest):
                total_fg += 1
                if r1 <= np.min(r2):
                    correct_fg += 1
            correct_lg = total_lg = 0
            for r1, r2 in zip(real, rest):
                total_lg += 1
                if r1 >= np.max(r2):
                    correct_lg += 1
            assert total_lg == total_fg
            if correct_lg >= correct_fg:
                correct, total = correct_lg, total_lg
            else:
                correct, total = correct_fg, total_fg
            xs.append(M)
            ys.append(1/(float(correct)/total))
            print(float(correct)/total)
        if period == 0:
            label = "Ppd. Prot, Mar '17"
            linestyle='+-'
            color='blue'
        else:
            linestyle='+--'
            if period ==6:
                color='teal'
                label = "Ppd. Prot, Jul '17*"
            else:
                color='navy'
                label = "Ppd. Prot, Jan '18*"
        plt.plot(xs, ys, linestyle, color=color,label=label)

    xs, ys = ([] for i in range(2))
    for M in range(1,16):
        outfile = "counterm_outfile_0_mo_%d_mixins_%s.npz" % (M, '2018')
        npzfile = np.load(outfile)
        real = npzfile['real']
        rest = npzfile['rest']
        print real.shape, rest.shape
        correct_fg = total_fg = 0
        for r1, r2 in zip(real, rest):
            total_fg += 1
            if r1 <= np.min(r2):
                correct_fg += 1
        correct_lg = total_lg = 0
        for r1, r2 in zip(real, rest):
            total_lg += 1
            if r1 >= np.max(r2):
                correct_lg += 1
        assert total_lg == total_fg
        if correct_lg >= correct_fg:
            correct, total = correct_lg, total_lg
        else:
            correct, total = correct_fg, total_fg
        xs.append(M)
        ys.append(1/(float(correct)/total))
        print(float(correct)/total)
    label = "Ppd. Prot, Mar '18"
    linestyle='+-'
    color='magenta'
    plt.plot(xs, ys, linestyle, color=color, label=label)

    for period in [0,12]:
        xs, ys = ([] for i in range(2))
        for M in range(1,16):
            outfile = "outfile_%d_mo_%d_mixins_%s.npz" % (period, M, '0.10')
            npzfile = np.load(outfile)
            real = npzfile['real']
            recents = npzfile['recents']
            rest = npzfile['rest']
            print real.shape, recents.shape, rest.shape
            correct = total = 0
            for r1, r2, r3 in zip(real, recents, rest):
                total += 1
                if r1 >= np.max(np.concatenate((r2,r3))):
                    correct += 1 
            xs.append(M)
            ys.append(1/(float(correct)/total))
            print(float(correct)/total)
        if period == 0:
            label = "v0.10.1, Mar '17"
            linestyle='+-'
            color='red'
        else:
            linestyle='+--'
            if period == 6:
                color='tomato'
                label = "v0.10.1, Jul '17*"
            else:
                color='salmon'
                label = "v0.10.1, Jan '18*"
        plt.plot(xs, ys, linestyle, color=color, label=label)
    
    xs, ys = ([] for i in range(2))
    for M in range(1,16):
        outfile = "outfile_%d_mixins_%s.npz" % (M, '0.10')
        npzfile = np.load(outfile)
        real = npzfile['real']
        recents = npzfile['recents']
        rest = npzfile['rest']
        print real.shape, recents.shape, rest.shape
        correct = total = 0
        for r1, r2, r3 in zip(real, recents, rest):
            total += 1
            if r1 >= np.max(np.concatenate((r2,r3))):
                correct += 1
        xs.append(M)
        ys.append(1/(float(correct)/total))
        print(float(correct)/total)
    label = "v0.10.1, Mar '18"
    linestyle='+-'
    color='brown'
    plt.plot(xs, ys, linestyle, color=color, label=label)

    xs, ys = ([] for i in range(2))
    for M in range(1,16):
        outfile = "outfile_%d_mixins_%s.npz" % (M, '0.11')
        npzfile = np.load(outfile)
        real = npzfile['real']
        recents = npzfile['recents']
        rest = npzfile['rest']
        print real.shape, recents.shape, rest.shape
        correct = total = 0
        for r1, r2, r3 in zip(real, recents, rest):
            total += 1
            if r1 >= np.max(np.concatenate((r2,r3))):
                correct += 1
        xs.append(M)
        ys.append(1/(float(correct)/total))
    plt.plot(xs, ys, '+-', color="green",label="v.0.11.0, Mar '18")

    xs = range(1,16)
    ys = [float(_+1) for _ in xs]
    plt.plot(xs, ys, '+-', label="Ideal", color="black")
    # figtext(.02, .02, "* projected from data on March 18, 2017")
    
    plt.xlabel('Number of Mixins')
    plt.ylabel('Effective-untraceability Set')
    plt.legend(loc='best', prop={'size':'medium'})
    plt.title('Effective-untraceability Set vs. Mixins, 100000 Trials')
    plt.savefig(filename)

def main():
    M = int(sys.argv[1])
    period = int(sys.argv[2])
    year = sys.argv[3]
    print 'Processing:', M, period, year
    sim(100000, M, period, year, True)

try: __IPYTHON__
except NameError:
    if __name__ == '__main__':
        main()

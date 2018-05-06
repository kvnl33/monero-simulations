# simulation of the Monero selection process for different protocols
import sys, sqlite3, matplotlib, collections, itertools, bisect, os
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Kevin Lee and Andrew Miller"
__maintainer__ = "Kevin Lee"
__email__ = "klee160@illinois.edu"

# define some global variables
CRYPTONOTE_MINED_MONEY_UNLOCK_WINDOW = 60
CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE = 10
recent_zone = 5 * 86400
recent_ratio = 0.25
top06mo = 1500203268
rec06mo = top06mo - recent_zone
top12mo = 1515755268
rec12mo = top12mo - recent_zone
rct_start = 1484051946
amounts = []
weights = []

if 'amount_dict' not in globals():
    amount_dict = collections.OrderedDict()
    amount_dict_keys = collections.OrderedDict()
    top_block = None
    top_time = None
    top_idx = {}
    recent_zones = {}
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

def amount_frequency():
    '''From the database we scraped blockchain data into, we get the denomination and frequencies
    of coins from the past month (~21900 blocks). We ignore RCT transactions b/c they are not going
    to be used for this set of simulations, and we ignore outputs with frequencies of less than 1000
    because they are insignificant. Afterwards, the counts are used to weigh the denominations such
    that the sum of weights adds up to 1.
    '''
    global amounts, weights
    conn = sqlite3.connect('outs_2017_03_18.db')
    c_1 = conn.cursor()
    c_1.execute('''SELECT amount,count(*) FROM out_table WHERE block_height >= 1220516 - 21900 GROUP BY amount ORDER BY count(*) DESC''')
    recent = c_1.fetchall()
    for row in recent:
        if row[1] > 1000 and int(row[0]) != 0:
            amounts.append(int(row[0]))
            weights.append(int(row[1]))
    amounts = np.asarray(amounts)
    weights = np.asarray(weights)
    weights = np.asarray([float(i)/np.sum(weights) for i in weights])

def preprocess(is_rct):
    '''Sets up the variables in the simulation. The most recent timestamp in the database is
    set to be the time of our simulation, the maximum global index for the each amount in the denominations
    we gleaned from the blockchain is found, as well as the global index corresponding to the recent zone 
    (~5 days) for each denomination. We keep track of those in our simulation. We do the same for 
    RCT values as well. Afterwards, we read in the recent and top zones for 6 and 12 month simulations to use.
    Lastly, we read the time differences between transaction time and output creation (receive) time for the
    transactions that were spent in-the-clear, because those will define our simulation behavior for choosing
    the real spend.
    '''
    global amount_dict, top_block, top_idx, time_diff, time_dict, time_dict_keys, top_time
    amount_frequency()
    
    if len(amount_dict) == len(amounts): return
    print 'Running preprocess first'
    conn_1 = sqlite3.connect('outs_2017_03_18.db')
    c_1 = conn_1.cursor()

    if is_rct:
        c_1.execute('''SELECT MAX(block_height) FROM out_table''')
        top_block, = c_1.fetchone()
    else:
        top_block = 1220516

    cmd = '''SELECT timestamp FROM out_table WHERE block_height = {ht}'''
    cmd = cmd.format(ht = top_block)
    c_1.execute(cmd)
    timenow, = c_1.fetchone()
    print 'top_block', top_block
    print 'timenow', timenow
    top_time = timenow

    for amt in amounts:
        cmd = '''SELECT MAX(g_idx) as idx from out_table WHERE amount = {denom}'''
        cmd = cmd.format(denom = amt)
        c_1.execute(cmd)
        top = c_1.fetchone()
        top_idx[amt] = top[0]
        
        cmd = '''SELECT MIN(g_idx) as idx FROM out_table WHERE amount = {denom} AND timestamp >= {start}'''
        cmd = cmd.format(denom = amt, start = timenow - recent_zone)
        c_1.execute(cmd)
        recent = c_1.fetchone()
        recent_zones[amt] = recent[0]

        cmd = '''SELECT timestamp, MAX(g_idx) FROM out_table WHERE amount={denom} GROUP BY timestamp'''
        cmd = cmd.format(denom = amt)
        c_1.execute(cmd)
        recent = c_1.fetchall()

        amount_dict_keys[amt] = []
        amount_dict[amt] = {}

        for row in recent:
            amount_dict[amt][int(row[0])] = int(row[1])
            amount_dict_keys[amt].append(int(row[0]))
    
    cmd = '''SELECT MAX(g_idx) as idx from out_table WHERE amount = {denom}'''
    cmd = cmd.format(denom = 0)
    c_1.execute(cmd)
    top = c_1.fetchone()
    top_idx[0] = {}
    top_idx[0][0] = top[0]

    cmd = '''SELECT MIN(g_idx) as idx FROM out_table WHERE amount = {denom} AND timestamp >= {start}'''
    cmd = cmd.format(denom = 0, start = timenow - recent_zone)
    c_1.execute(cmd)
    recent = c_1.fetchone()
    recent_zones[0] = {}
    recent_zones[0][0] = recent[0]

    cmd = '''SELECT timestamp, MAX(g_idx) FROM out_table WHERE amount={denom} GROUP BY timestamp'''
    cmd = cmd.format(denom = 0)
    c_1.execute(cmd)
    recent = c_1.fetchall()
    x = np.asarray([int(row[0]) for row in recent])
    y = np.asarray([int(row[1]) for row in recent])
    for row in recent:
        time_dict[row[0]] = row[1]

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
    recent_zones[0][6] = time_dict[rec06mo]
    recent_zones[0][12] = time_dict[rec12mo]
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

def fetch_real_output(N, period, top_global_idx):
    '''Randomly selects an output to be used as the real spend. A time difference is
    selected from our list of zero-input transactions we collected from the blockchain. Next,
    using the time difference, we find the closest global index if the block is in the 
    dictionary. If this is a RCT transaction, we also take into
    account whether or not we are doing 6 month or 12 month simulations. We normalize the index
    returned by dividing the real index by the top index. For instance, if the top index is Y
    and the chosen index is X, we return X/Y.
    '''
    if N==0:
        while True:
            rhd = np.random.choice(time_diff)
            if period == 6:
                rb = top06mo - rhd
            elif period == 12:
                rb = top12mo - rhd
            else:
                rb = top_time - rhd
            if rb >= rct_start:
                break
        closest = bisect.bisect_left(time_dict_keys, rb)
        real_g_idx = time_dict[time_dict_keys[closest]]
        return float(real_g_idx)/top_global_idx
    else:
        while True:
            rhd = np.random.choice(time_diff)
            rb = top_time - rhd
            if rb >= 0:
                break
        closest = bisect.bisect_left(amount_dict_keys[N], rb)
        real_g_idx = amount_dict[N][amount_dict_keys[N][closest]]
        return float(real_g_idx)/top_global_idx

def sample_mixins(amount, num_mix, period, version='pre0.9', is_rct=False):
    '''The mixin-sampling protocol in our simulation changes based on the version argument
    that is passed into it and reflects the different protocols Monero has employed since its 
    inception (see https://github.com/monero-project/monero/blob/master/src/wallet/wallet2.cpp). 
    From a high-level, pre0.9 will select mixins from genesis to top-height uniformly,
    ver0.9 will select mixins from genesis to top-height with a triangular distribution, and 0.10
    will employ ver0.9 selection, but also select uniformly across a recent zone (the past 5 days),
    and combine that with the other mixin candidates. The mixins are then uniformly selected from the
    candidate mixins. RingCT simulations and recent zone selections are only available starting in version 0.10. 
    The lists for mixins, recents, and final are initialized, and a top height is selected based on 
    amount. The protocol expands to select more mixins than that are actually used in a 
    transaction, and we gauge that based on version. We normalize the mixin returned by dividing the 
    real index by the top index. For instance, if the top index is Y and the chosen index is X, we return X/Y.
    '''
    assert version in ['pre0.9','0.9','0.10']
    if is_rct: assert version == '0.10', "RingCT only available after 0.10"
    mixin_vector, recent_vector, final_vector = ([] for i in range(3))
    top_global_idx = top_idx[amount] if not is_rct else top_idx[amount][period]
    recent_idx = recent_zones[amount] if not is_rct else recent_zones[amount][period]
    if (recent_idx < 0): recent_idx = 0
    req = int((num_mix + 1) * 1.5) + 1
    if is_rct: req += (CRYPTONOTE_MINED_MONEY_UNLOCK_WINDOW - CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE)
    real = fetch_real_output(amount, period, top_global_idx)
    recent_req = int(req * recent_ratio)
    if(recent_req<=1):
        recent_req = 1
    if(recent_req> top_global_idx-recent_idx+1):
        recent_req = top_global_idx-recent_idx+1

    if(real>=recent_idx/float(top_global_idx)):
        recent_req -= 1
    if version != '0.10': recent_req = 0
    num_found = 0
    recent_found = 0
    while (num_found < req):
        while (recent_found < recent_req):
            r = np.random.randint(recent_idx, top_global_idx+1)
            rmixin = r/float(top_global_idx)
            if (rmixin not in recent_vector and rmixin != real):
                recent_vector.append(rmixin)
                mixin_vector.append(rmixin)
                num_found += 1
                recent_found += 1        
        if version == 'pre0.9':
            a = np.random.randint(0,top_global_idx+1) 
        else:
            a = int(np.random.triangular(0,1,1)*top_global_idx) 
        amixin = a/float(top_global_idx)
        if (amixin not in mixin_vector and amixin != real):
            mixin_vector.append(amixin)
            num_found += 1
    while(num_mix!=0):
        re = np.random.choice(mixin_vector)
        if (re not in final_vector):
            final_vector.append(re)
            num_mix-=1
    recent_mixins = [val for val in recent_vector if val in final_vector]
    rest = filter(lambda x: x not in recent_mixins, final_vector)
    return real, recent_mixins, rest

def sim(N,M,time,version='pre0.9', is_rct=False):
    '''The main simulation function starts the simulations. The simulation is run N times
    with M mixins chosen. We normalize the mixin returned by dividing the 
    real index by the top index. For instance, if the top index is Y and the chosen index is X, 
    we return X/Y. This is necessary because different denominations will have varying
    top global indices, and our simulation needs to be generalized. We will put all the 
    percentages returned inside a vector and save it externally so we can run various graphing
    schemes on one simulation batch in the future. The simulation is run with 
    sim(# of trials, # of mixins), 0 to sample all, 6 for six months, 12 for 12 months.
    '''
    preprocess(is_rct)
    real, recents, rest = ([] for i in range(3))
    for x in range(0,N):
        print(x+1)
        m = np.random.choice(amounts, p=weights) if is_rct == False else 0
        n, p, q = sample_mixins(m,M,time, version, is_rct)
        real.append(n)
        recents.append(p)
        rest.append(q)
    real = np.asarray(real)
    recents = np.asarray(recents)
    rest = np.asarray(rest)

    outfile = "outfile_%d_mo_%d_mixins_%s" % (time, M, version)
    np.savez(outfile, real=real, recents=recents, rest=rest)

def main():
    M = int(sys.argv[1])
    time = int(sys.argv[2])
    print 'Processing:', M, time
    sim(100000, M, time, '0.10', is_rct=True)

try: __IPYTHON__
except NameError:
    if __name__ == '__main__':
        main()


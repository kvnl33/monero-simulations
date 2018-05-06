# simulation of the Monero selection process for different protocols
import sys, sqlite3, matplotlib, collections, itertools, bisect, os
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Kevin Lee and Andrew Miller"
__maintainer__ = "Kevin Lee"
__email__ = "klee160@illinois.edu"

#define some global variables
CRYPTONOTE_MINED_MONEY_UNLOCK_WINDOW = 60
CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE = 10
rct_start = 1484051946
top_block = None
top_time = None
top_idx = None
recent_zones = {"0.10": 5 * 86400, "0.11": 1.8 * 86400}
recent_ratios = {"0.10": 0.25, "0.11": 0.50}
recent_start = {}
time_diff = []
time_dict = collections.OrderedDict()
time_dict_keys = []

def preprocess(version="0.11"):
    '''Sets up the variables in the simulation. The most recent timestamp in the database is
    set to be the time of our simulation, the maximum global index for the each amount in the denominations
    we gleaned from the blockchain is found, as well as the global index corresponding to the recent zone 
    (~5 days) for each denomination. We keep track of those in our simulation. We do the same for 
    RCT values as well. Lastly, we read the time differences between transaction time and output creation 
    (receive) time for the transactions that were spent in-the-clear, because those will define 
    our simulation behavior for choosing the real spend.
    '''
    global top_block, top_idx, time_diff, time_dict, time_dict_keys, top_time
    print 'Running preprocess first'
    conn_1 = sqlite3.connect('outs_2018_03_29.db')
    c_1 = conn_1.cursor()
    c_1.execute('''SELECT MAX(block_height) FROM out_table''')
    top_block, = c_1.fetchone()

    cmd = '''SELECT timestamp FROM out_table WHERE block_height = {ht}'''
    cmd = cmd.format(ht = top_block)
    c_1.execute(cmd)
    timenow, = c_1.fetchone()
    print 'top_block', top_block
    print 'timenow', timenow
    top_time = timenow

    cmd = '''SELECT MAX(g_idx) as idx from out_table'''
    c_1.execute(cmd)
    top, = c_1.fetchone()
    top_idx = top

    cmd = '''SELECT MIN(g_idx) as idx FROM out_table WHERE timestamp >= {start}'''
    cmd = cmd.format(denom = 0, start = timenow - recent_zones[version])
    c_1.execute(cmd)
    recent, = c_1.fetchone()
    recent_start[version] = recent
    print 'recent_zone', recent

    cmd = '''SELECT timestamp, MAX(g_idx) FROM out_table GROUP BY timestamp'''
    c_1.execute(cmd)
    recent = c_1.fetchall()
    x = np.asarray([int(row[0]) for row in recent])
    y = np.asarray([int(row[1]) for row in recent])

    for row in recent:
    	time_dict[row[0]] = row[1]
    time_dict_keys = time_dict.keys()

    conn= sqlite3.connect('zinput.db')
    c_2 = conn.cursor()
    cmd = '''SELECT tx_timestamp - mixin_timestamp as time_diff from first'''
    c_2.execute(cmd)
    diff = c_2.fetchall()
    for row in diff:
        if row[0] >= 0:
            time_diff.append(row[0])
    time_diff = np.asarray(time_diff)

def fetch_real_output(top_global_idx):
    '''Randomly selects an output to be used as the real spend. A time difference is
    selected from our list of zero-input transactions we collected from the blockchain. Next,
    using the time difference, we find the closest global index if the block is in the 
    dictionary. We normalize the index
    returned by dividing the real index by the top index. For instance, if the top index is Y
    and the chosen index is X, we return X/Y.
    '''
    while True:
        rhd = np.random.choice(time_diff)
        rb = top_time - rhd
        if rb >= rct_start:
            break
    closest = bisect.bisect_left(time_dict_keys, rb)
    real_g_idx = time_dict[time_dict_keys[closest]]
    return float(real_g_idx)/top_global_idx

def sample_mixins(num_mix, is_rct=True, version='0.11'):
    '''The mixin-sampling protocol in our simulation changes based on the version argument
    that is passed into it and reflects the different protocols Monero has employed since its 
    inception (see https://github.com/monero-project/monero/blob/master/src/wallet/wallet2.cpp). 
    From a high-level, the mixins are selected over a triangular distribution, 
    but also select over a triangular distribution across a recent zone (the past 1.8 days),
    and combine that with the other mixin candidates. The mixins are then uniformly selected from the
    candidate mixins. 
    The lists for mixins, recents, and final are initialized, and a top height is selected based on 
    amount. The protocol expands to select more mixins than that are actually used in a 
    transaction, and we gauge that based on version. We normalize the mixin returned by dividing the 
    real index by the top index. For instance, if the top index is Y and the chosen index is X, we return X/Y.
    '''        
    mixin_vector, recent_vector, final_vector = ([] for i in range(3))
    top_global_idx = top_idx
    recent_idx = recent_start[version]

    if (recent_idx < 0):
        recent_idx = 0
    req = ((num_mix + 1) * 1.5) + 1
    req = int(req)
    if is_rct:
        req += (CRYPTONOTE_MINED_MONEY_UNLOCK_WINDOW - CRYPTONOTE_DEFAULT_TX_SPENDABLE_AGE)
    real = fetch_real_output(top_global_idx)
    recent_req = req * recent_ratios[version]
    recent_req = int(recent_req)
    if(recent_req<=1):
        recent_req = 1
    if(recent_req> top_global_idx-recent_idx+1):
        recent_req = top_global_idx-recent_idx+1
    if(real>=recent_idx/float(top_global_idx)):
        recent_req -= 1
    recent_req_num = recent_req

    num_found = 0
    recent_found = 0 
    while (num_found < req):
        while (recent_found < recent_req):
            if version == "0.11":
                r = int(np.random.triangular(recent_idx, top_global_idx, top_global_idx))
            else:
                r = np.random.randint(recent_idx, top_global_idx+1)
            rmixin = r/float(top_global_idx)
            if (rmixin not in recent_vector and rmixin != real):
                recent_vector.append(rmixin)
                mixin_vector.append(rmixin)
                num_found += 1
                recent_found += 1

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

def sim(N, M, version='0.11', is_rct=True):
    '''The main simulation function starts the simulations. The simulation is run N times
    with M mixins chosen. We normalize the mixin returned by dividing the 
    real index by the top index. For instance, if the top index is Y and the chosen index is X, 
    we return X/Y. This is necessary because different denominations will have varying
    top global indices, and our simulation needs to be generalized. We will put all the 
    percentages returned inside a vector and save it externally so we can run various graphing
    schemes on one simulation batch in the future. The simulation is run with 
    sim(# of trials, # of mixins).
    '''
    real, recents, rest = ([] for i in range(3))
    preprocess(version)
    for x in range(0,N):
        print(x+1)
        n, p, q = sample_mixins(M, is_rct, version)
        real.append(n)
        recents.append(p)
        rest.append(q)
    real = np.asarray(real)
    recents = np.asarray(recents)
    rest = np.asarray(rest)
    outfile = "outfile_%d_mixins_%s" % (M, version)
    np.savez(outfile, real=real, recents=recents, rest=rest)

def main():
    M = int(sys.argv[1])
    ver = sys.argv[2]
    print 'Processing:', M
    sim(100000, M, ver, is_rct=True)

try: __IPYTHON__
except NameError:
    if __name__ == '__main__':
        main()


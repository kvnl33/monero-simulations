# Monero Spend Simulations

These are the simulation files used in our paper to model the distribution of the different mixin-sampling protocols deployed in Monero. We use the Monte Carlo method to run 100000 independent transaction simulations for each mixin number specified, and for each version of the mixin-sampling protocol (pre v0.9, v0.9, v0.10, v0.10 with RingCT, and v0.11). In order to simulate growth, we also linearly extrapolate what the distribution would look like at 6 months and 12 months. 

This repository does not contain the deducibility analysis using the Sudoku algorithm. It can be found at [maltemoeser/moneropaper](https://github.com/maltemoeser/moneropaper).

## Background

The simulations for current and past mixin-sampling protocols are found in ```sim_2017_03_18.py``` and ```sim_2018_03_29.py```. To highlight, here are the key differences between sampling protocols:
* pre v0.9: Mixins are selected uniformly across the output set.
* v0.9: Mixins are selected with a triangular distribution over the output set. This means that more recent outputs are more likely to be chosen.
* v0.10: The selection from v0.9 still holds, but now a fraction of the mixins are also selected uniformly from a recent zone (~5 days) and combined to the set.
* v0.10 (RingCT): There is no distinction between denominations, and more candidate mixins are requested to compensate for the initially smaller mixin set.
* v0.11 (RingCT): The recent zone is now shortened to the past 1.8 days. A larger fraction of the mixins are now chosen from the recent zone (50%), and are selected with a triangular distribution over said zone.

Our simulation involves selecting real data from the blockchain that was scraped into our databases. We use the most recent time to simulate our transaction time, and uniformly select a time difference between fund creation and spend for zero-input transactions (spending in the clear) to select a **real spend**. Note that zero-input transactions have no mixins, and thus accurately reflect user spending behavior. Because funds are timestamped, we select the output that is closest to the difference between the transaction time and selected time difference. Afterwards, candidate mixins are selected based on the version we are simulating, and of those, a final obfuscation set is selected with the the real spend included. 

In designing this method, we decided to simulate on denominations from the past 30 days as to give less weight to stale denominations, though the top 10 most frequent denominations do not vary much. Also, we ignore amounts that have fewer than 1000 occurances, as those are not significant. Because each denomination occurs a different number of times, we return the mixin-offset; i.e., the global index of the the real spend or mixin divided by the top global index for that fund. This allows for meaningful graphs and trends.

### Blockchain Data

The data we scraped from the blockchain are in  ```zinput.db```, ```outs_2017_03_18.db```, and ```outs_2018_03_29.db```. These were all collected by synchronizing a local copy of the Monero blockchain from a running full node up to the date of our experiment, and then connecting our scraper to the blockchain and parsing relevant data.
* ```zinput.db``` contains all 0-mixin transaction data.
* ```outs_2017_03_18.db``` contains all created outputs in the network from genesis to block 1268880. 
* ```outs_2018_03_29.db``` contains all created RingCT outputs in the network from block 1220517 (start of RingCT) to 1540516. 

The databases are not included due to size restrictions, but are hosted [here](https://uofi.box.com/s/0tu8i9hezx11geujl5e3q3dufm6smxgd). The MD5 hashes of the files are:
| File      | MD5SUM |
| ----------- | ----------- |
| ```zinput.db```      | ecc667e72b234a9311fb7c5b2fdc85aab |
| ```outs_2017_03_18.db```   | 62588a6a6caf8e3e795ef4f811e46c33 |
| ```outs_2018_03_29.db```   | 172bfb34f1ab180cc692e1f20f17c044 |

Alternatively, the **CREATE** statement(s) for the tables in the databases are:

#### zinput.db
```sqlite3
CREATE TABLE first (tx_hash STRING, block_height INTEGER, amount INTEGER, mixin_height INTEGER, tx_timestamp INTEGER, mixin_timestamp INTEGER, mixin_tx STRING);
CREATE INDEX idx on first (tx_timestamp, mixin_timestamp);
```
#### outs_2017_03_18.db
```sqlite3
CREATE TABLE out_table (tx_hash STRING, block_height INTEGER, amount INTEGER, g_idx INTEGER, timestamp INTEGER);
CREATE INDEX idx ON out_table (amount, timestamp, block_height);
```
#### outs_2018_03_29.db
```sqlite3
CREATE TABLE out_table (block_hash STRING, block_height INTEGER, tx_hash STRING, timestamp INTEGER, outkey STRING, g_idx INTEGER);
```    

## Countermeasure

The simulations for our countermeasure protocol are found in ```counterm.py```.
Our countermeasure selects the **real spend** in the same manner as the current protocols do, but mixins are selected over a fitted gamma distribution, modeled after Bitcoin user spend patterns.

To measure the effectiveness of the countermeasure, the `graph_guesser` and `graph_anonset` functions run a guess-most-recent and guess-least-recent analysis on our simulation results, and returns the worst-case result. We have found that the worst-case performance of the countermeasure is significantly better than that of current and past protocols.

### Installation

All required packages can be found in ```requirements.txt```.

## Running the tests

The simulation files can either be run in IPython or directly in the terminal. For example, to run ```sim.py``` within IPython, entering:
```python
sim(1000, 5, 6, '0.9', False)
```
runs the 1000 runs of a simulation with 5 mixins, extrapolated to 6 months for protocol v0.9. RingCT is only available for v0.10, so that argument would not have any effect.

Within the terminal, entering:
```
python sim_2017_03_18.py 5 0 
```
runs 100000 runs of a simulation with 5 mixins with no time extrapolation for Monero v0.10 with RingCT.

## Generating the graph

The graph (Fig. 12) in our paper compares the performance of our proposed countermeasure against that of past and current mixin selection protocols at different periods of time for 1 to 15 mixins. Specifically, we compare the effective-untraceability set across all protocols. The simulations used as well as the grapher can be used to replicate our trials and generate the figure from the paper, and are included in the script ```grapher.py```. 

To run the script, enter:
```
python grapher.py -p PROCESSES -f FILENAME
```

For instance, running ```python grapher.py -p 8 -f figure.png``` will spawn 8 workers to handle all of the simulations and output the graph ```figure.png```.

### Related work

Our paper, **An Empirical Analysis of Traceability in the Monero Blockchain**, was presented at PETS 2018. It can be found [here](https://arxiv.org/pdf/1704.04299.pdf).

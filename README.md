# Monero Spend Simulations

These are the simulation files used in our paper to model the distribution of the different mixin-sampling protocols deployed in Monero. We use the Monte Carlo method to run 100000 indepent transaction simulations for each mixin number specified, and for each version of the mixin-sampling protocol (pre v0.9, v0.9, v0.10, and v0.10 with RCT). In order to growth, we also extrapolate to what the distribution would look like at 6 months and 12 months. 

## Background

The simulations for current and past mixin-sampling protocols are found in ```sim.py```. To highlight, here are the key differences between sampling protocols:
* pre v0.9: Mixins are selected uniformly across the output set.
* v0.9: Mixins are selected with a triangular distribution over the output set. This means that more recent outputs are more likely to be chosen.
* v0.10: The selection from v0.9 still holds, but now a fraction of the mixins are also selected uniformly from a recent zone (~5 days) and combined to the set.
* v0.10 (RCT): There is no distinction between denomination, and more candidate mixins are requested to compensate for the initially smaller mixin set.

Our simulation involves selecting real data from the blockchain that was scraped into our databases. We use the most recent time to simulate our transaction time, and uniformly select a time difference between fund creation and spend for zero-input transactions (spending in the clear) to select a **real spend**. Note that zero-input transactions have no mixins, and thus accurately reflect user spending behavior. Because funds are timestamped, we select the output that is closest to the difference between the transaction time and selected time difference. Afterwards, candidate mixins are selected based on the version we are simulating, and of those, a final obfuscation set is selected with the the real spend included. 

In designing this method, we decided to simulate on denominations from the past 30 days as to give less weight to stale denominations, though the top 10 most frequent denominations do not vary much. Also, we ignore amounts that have fewer than 1000 occurances, as those are not significant. Because each denomination occurs a different number of times, we return the mixin-offset; i.e., the global index of the the real spend or mixin divided by the top global index for that fund. This allows for meaningful graphs and trends.   

## Countermeasure

The simulations for our countermeasure protocol are found in ```counterm.py```.
Our countermeasure selects the **real spend** in the same manner as the current protocols do, but mixins are selected over a fitted gamma distribution, modeled after Bitcoin user spend patterns.

To measure the effectiveness of the countermeasure, the `graph_guesser` and `graph_anonset` functions run a guess-most-recent and guess-least-recent analysis on our simulation results, and returns the worst-case result. We have found that the worst-case performance of the countermeasure is significantly better than that of current and past protocols.

### Installation

All required packages can be found in ```requirements.txt```.

## Running the tests

The simulation files can either be run in IPython or directly in the terminal. For example, to run ```sim.py``` within IPython, entering:
```python
sim(1000, 5, 6, version='0.9', is_rct=False)
```
runs the 1000 runs of a simulation with 5 mixins, extrapolated to 6 months for Monero v0.9. RingCT is only available for v0.10, so that argument would not have any effect.

Within the terminal, entering:
```
python sim.py 5 0.10 
```
runs 100000 runs of a simulation with 5 mixins with no time extrapolation for Monero v0.10 without RingCT.

### Related work

The results of our simulation as well as other components of our paper are included [here](http://monerolink.com/).

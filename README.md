# CLANGNEURALNET — Jack Lenox

CLANGNEURALNET is a C machine learning library and passion project built to explore both modern and classical ML concepts, and to test experimental ideas in network architecture.

The library expanded over time to include experimental implementations of ideas I had regarding perceptrons and MLPs. Please feel free to leave comments or message me regarding design decisions, suggestions, or any errors I may have made. Feel free to experiment with the library and combine things I may not have thought of.

For usage of each function please see [reference.md](reference.md).

If you wish to contact me: jack.lenox@trojans.dsu.edu — I can't promise I can answer everything but I'll do my best.

---

## Experimental Implementations

### Spatially Conscious Perceptron Chaining

Training begins on a single perceptron which trains until it classifies all points or stagnates. Any remaining misclassified points spawn a child perceptron whose spatial view is constrained to the region where failures occurred — typically around difficult decision boundaries.

> **Note:** This functions as intended across a variety of 2D datasets of varying complexity. A test implementation using a small XOR dataset is included in `main.c` along with a graph of one result. LabPlot visualization integration is planned.

### Consensus Mask Pruning

Partially trains N networks and prunes them before convergence. The intent was to save computation and highlight important connections faster than full training.

> **Note:** Didn't perform as intended. Further investigation planned through statistical analysis rather than intuition alone.

---

## How to Build

Run the following from the `/CLANGNEURALNET/` directory:

```bash
cmake --build .\Build\
```

Currently Windows only — outputs `a.exe` which runs `main.c`.

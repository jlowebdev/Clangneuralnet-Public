# CLANGNEURALNET — Function Reference

A complete reference for all public functions in CLANGNEURALNET. For project overview and build instructions see README.md.

Labels throughout this library use **+1 and -1** — not 0 and 1. Passing 0/1 labels will produce silent incorrect behavior.

---

## Perceptron

Functions for constructing, initializing, training, and running the spatially conscious perceptron chain.
## NOTE
Seconddim is usually just used as a place holder so that the function knows what size things are the functions assume two dimensional data

---

### `perc_random_init`
```c
void perc_random_init(struct perc *perc)
```
Xavier-initializes a two-weight perceptron with zero bias. Weights are drawn from the uniform range √(6 / (1+1)).

**When to use:** At the start of training when you have no prior knowledge of the data distribution. If you have labeled training data available, prefer `smart_weight_init()` — it converges faster and more consistently.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perc` | `struct perc *` | Pointer to an allocated perceptron. Must not be NULL. |

---

### `smart_weight_init`
```c
void smart_weight_init(struct perc *ada, struct set *temp)
```
Seeds weights from the vector between class centroids and places the bias at the midpoint between them. Falls back to `perc_random_init()` if the dataset contains only one point.

**When to use:** Whenever labeled training data is available. Significantly reduces chain length and eliminates the explosion behavior that random initialization can produce on difficult datasets. Do not use with heavily noisy or imbalanced labels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Perceptron to initialize. |
| `temp` | `struct set *` | Labeled training set with at least one sample of each class. |

---

### `calc_perceptron_output`
```c
void calc_perceptron_output(struct perc *perc)
```
Computes the raw weighted sum of inputs and bias and stores it in `perc->output`. Does not apply activation.

**When to use:** Call before `perc_activation()` in any manual training or inference loop.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perc` | `struct perc *` | Perceptron with inputs already set. |

---

### `perc_activation`
```c
void perc_activation(struct perc *perc)
```
Applies a step activation to `perc->output` and stores the result in `perc->pred`. Returns +1 if output > 0, -1 otherwise.

**When to use:** Call immediately after `calc_perceptron_output()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perc` | `struct perc *` | Perceptron with output already computed. |

---

### `train_perc`
```c
void train_perc(struct perc *perc, float target)
```
Applies one perceptron learning rule update: `w += 0.001 * (target - pred) * input`. Bias is updated with a rate of 0.1.

**When to use:** Inside a manual training loop after `calc_perceptron_output()` and `perc_activation()`. For full dataset training with stagnation detection, prefer `train_perc_stagsp()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `perc` | `struct perc *` | Perceptron to update. |
| `target` | `float` | Ground truth label. Must be +1 or -1. |

---

### `train_perc_stagsp`
```c
void train_perc_stagsp(struct perc *ada, struct set *hold_subset)
```
Trains a perceptron over a full dataset until error reaches zero or stagnates for 200 consecutive epochs without improvement.

**When to use:** When training a newly created child node before inserting it into the chain. Used internally by `veri_perc_chain2()`. Not recommended for manual training loops where you want finer control — use `train_perc()` directly instead.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Perceptron to train. |
| `hold_subset` | `struct set *` | Labeled dataset this perceptron is responsible for. |

---

### `find_perc_bounds`
```c
void find_perc_bounds(struct perc *ada, struct set *set)
```
Scans the dataset and sets the spatial bounding box (`min[0]`, `min[1]`, `max[0]`, `max[1]`) on the perceptron with a small padding margin.

**When to use:** Call after creating a new perceptron and before training it, so it knows the spatial region it is responsible for. Called automatically within `veri_perc_chain2()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Perceptron to assign bounds to. |
| `set` | `struct set *` | Dataset whose coordinate extents define the bounds. |

---

### `perceptron_trav`
```c
int perceptron_trav(struct perc *ada, float x, float y, float threshold, int depth)
```
Traverses the perceptron chain to find the node whose bounds contain (x, y) and returns its prediction. Recurses to `ada->prev` if the input falls outside the current node's bounds. Prints prediction, confidence, and depth to stdout.

**When to use:** At inference time on a fully trained chain. Pass the tail node (the one with `last == 1`, returned by `veri_perc_chain2()`) as `ada`. Pass 0 for both `threshold` and `depth` on the initial call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Entry point — use the tail node returned by `veri_perc_chain2()`. |
| `x` / `y` | `float` | Input coordinates to classify. |
| `threshold` | `float` | Reserved. Pass 0. |
| `depth` | `int` | Initial recursion depth. Pass 0. |

**Returns:** Predicted class label: +1 or -1.

---

### `veri_perc_chain2`
```c
struct perc *veri_perc_chain2(struct perc *ada, int firstdim, int seconddim,
                               float data[firstdim][seconddim], int labels[firstdim], int depth)
```
The main training function for the spatially conscious perceptron chain. Walks the existing chain, identifies misclassified points within each node's spatial bounds, creates and trains a child perceptron for those failures, and inserts it into the linked list. Repeats until all points are handled.

**When to use:** Call once after constructing an initial perceptron with `create_perc()`, setting its bounds with `find_perc_bounds()`, and doing an initial training pass with `train_perc_stagsp()`. Pass `depth = 0`. Use the returned pointer as the entry point for `perceptron_trav()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Head of the chain, initialized and with bounds set. |
| `firstdim` | `int` | Number of training samples. |
| `seconddim` | `int` | Number of dimensions (currently 2). |
| `data` | `float[][]` | Coordinate array of shape [firstdim][2]. |
| `labels` | `int[]` | Class labels, one per sample. Must be +1 or -1. |
| `depth` | `int` | Pass 0 on initial call. |

**Returns:** Pointer to the tail node (`last == 1`). Use this as the entry point for `perceptron_trav()`.

---

### `create_perc`
```c
struct perc *create_perc()
```
Allocates and randomly initializes a new two-weight perceptron.

**When to use:** To create the first node in a new chain before calling `find_perc_bounds()` and `veri_perc_chain2()`.

**Returns:** Heap-allocated `struct perc *`. The chain owns this pointer after it is passed to `veri_perc_chain2()`.

---

## Neural Network

Functions for forward pass, backpropagation, and pruning in a fully connected neural network.

---

### `nodes_gen`
```c
struct node *nodes_gen(int len, int next_layer_len)
```
Allocates and Xavier-initializes a layer of nodes. All weights are active and all values and deltas are zeroed.

**When to use:** Once per layer when constructing a new network. Pass the size of the next layer as `next_layer_len` so weights are sized correctly. For the output layer, the next layer size is 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `len` | `int` | Number of nodes in this layer. |
| `next_layer_len` | `int` | Number of nodes in the following layer. |

**Returns:** Heap-allocated array of `struct node`. Caller is responsible for freeing.

---

### `hidden_math`
```c
void hidden_math(struct node *alayer, struct node *blayer, int alen, int blen, int isout)
```
Forward pass from alayer into blayer. Computes a masked weighted sum and applies tanh for hidden layers or sigmoid for the output layer.

**When to use:** Once per layer pair in the forward pass, from input toward output. Set `isout = 0` for all hidden layers and `isout = 1` only for the final output layer. After pruning, use `lth_hidden_math()` instead, which uses weight magnitude rather than the active mask.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Source layer. |
| `blayer` | `struct node *` | Destination layer. |
| `alen` / `blen` | `int` | Node counts for source and destination. |
| `isout` | `int` | 1 = sigmoid output, 0 = tanh hidden. |

---

### `lth_hidden_math`
```c
void lth_hidden_math(struct node *alayer, struct node *blayer, int alen, int blen, int isout)
```
Forward pass variant for lottery-ticket style sparse networks. Only includes weights with absolute value above 0.6.

**When to use:** As a drop-in replacement for `hidden_math()` after applying `pruning()`. Uses a hard weight threshold rather than the active mask.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Source layer. |
| `blayer` | `struct node *` | Destination layer. |
| `alen` / `blen` | `int` | Node counts for source and destination. |
| `isout` | `int` | 1 = sigmoid output, 0 = tanh hidden. |

---

### `final_layer_delta_calc`
```c
void final_layer_delta_calc(struct node *alayer, struct node *blayer, int alen, int blen)
```
Computes output layer deltas as the difference between target values and predictions: `delta = target - output`.

**When to use:** First step of the backward pass, applied to the output layer only. Follow with `delta_calc()` for each hidden layer working backward toward the input.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Output layer (predictions). |
| `blayer` | `struct node *` | Target layer (ground truth values). |
| `alen` / `blen` | `int` | Layer sizes. |

---

### `delta_calc`
```c
void delta_calc(struct node *alayer, struct node *blayer, int alen, int blen)
```
Backpropagates deltas from blayer into alayer using the tanh derivative: `delta = sum(next_deltas * weights) * (1 - value²)`.

**When to use:** After `final_layer_delta_calc()`, call this for each hidden layer in reverse order (output side toward input side) before calling `back_prop()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Hidden layer receiving the delta signal. |
| `blayer` | `struct node *` | Layer closer to output with already-computed deltas. |
| `alen` / `blen` | `int` | Layer sizes. |

---

### `back_prop`
```c
void back_prop(struct node *alayer, struct node *blayer, int alen, int blen, int cmp_f)
```
Updates alayer weights using blayer deltas. Skips masked connections. Learning rate is 0.1 normally, or 0.4 in consensus pruning mode.

**When to use:** After deltas are computed for all layers, call this for each layer pair from output toward input to complete the backward pass. Set `cmp_f = 1` only during consensus mask pruning experiments — not during standard training.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Layer whose weights are being updated. |
| `blayer` | `struct node *` | Layer whose deltas drive the update. |
| `alen` / `blen` | `int` | Layer sizes. |
| `cmp_f` | `int` | 1 = consensus pruning mode (lr 0.4), 0 = normal (lr 0.1). |

---

### `pruning`
```c
void pruning(struct node *alayer, struct node *blayer, int alen, int blen, float minabsval)
```
Masks out any weight whose absolute value falls below `minabsval` by setting `active[j] = 0`. Masked weights are skipped in both forward and backward passes.

**When to use:** After partial convergence, not before. Typical threshold values are 0.05–0.15. After pruning, continue training with `back_prop()` to allow remaining weights to compensate. Use `density()` before and after to measure sparsity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Layer containing weights to prune. |
| `blayer` | `struct node *` | Next layer (used for dimension bounds). |
| `alen` / `blen` | `int` | Layer sizes. |
| `minabsval` | `float` | Weights with \|w\| below this value are masked. |

---

### `density`
```c
int density(struct node *alayer, int alen, int blen)
```
Returns the number of masked-out (pruned) connections in a layer.

**When to use:** After `pruning()` to measure how sparse the layer has become. Divide the result by `(alen * blen)` to get the pruning fraction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Layer to inspect. |
| `alen` / `blen` | `int` | Layer sizes. |

**Returns:** Integer count of connections where `active[j] == 0`.

---

### `weighted_sum`
```c
float weighted_sum(float *a, float *b, int len)
```
Returns the dot product of two float arrays.

**When to use:** General purpose utility for computing a weighted sum outside of the node layer abstraction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` / `b` | `float *` | Input arrays of equal length. |
| `len` | `int` | Length of both arrays. |

**Returns:** Dot product as a float.

---

### `clear_nodes`
```c
void clear_nodes(struct node *alayer, int alen)
```
Zeros the value field of every node in a layer.

**When to use:** At the start of each forward pass to prevent stale values from accumulating across samples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alayer` | `struct node *` | Layer to clear. |
| `alen` | `int` | Number of nodes. |

---

## Data

Functions for constructing and managing labeled datasets.

---

### `create_subset`
```c
struct set *create_subset(int firstdim, int seconddim,
                           float data[firstdim][seconddim], int labels[firstdim])
```
Deep-copies a coordinate array and label array into a newly allocated `struct set`.

**When to use:** When you need a self-contained copy of a dataset or subset that a perceptron can train on independently. The caller owns the returned pointer — free with `free_set()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `firstdim` | `int` | Number of data points. |
| `seconddim` | `int` | Number of dimensions per point. |
| `data` | `float[][]` | Coordinate array. |
| `labels` | `int[]` | Class labels. Must be +1 or -1. |

**Returns:** Heap-allocated `struct set *`. Free with `free_set()`.

---

### `ret_tempf`
```c
struct set *ret_tempf(struct perc *ada, struct set *temp)
```
Runs a full dataset through a perceptron and returns a new set containing only the misclassified points. Frees the original set. If all points are correctly classified, returns the original set unchanged.

**When to use:** To isolate failure cases for inspection or to pass to a new chain node. Note that `temp` may be freed internally — do not use the original pointer after calling this function.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ada` | `struct perc *` | Trained perceptron to evaluate against. |
| `temp` | `struct set *` | Dataset to test. May be freed internally. |

**Returns:** New `struct set *` containing only misclassified points, or the original set if none failed. Free with `free_set()`.

---

### `copy_set`
```c
struct set *copy_set(struct set *temp)
```
Returns a deep copy of a set without modifying or freeing the original.

**When to use:** When you need a duplicate of a set that will be modified or freed independently of the original.

| Parameter | Type | Description |
|-----------|------|-------------|
| `temp` | `struct set *` | Set to copy. |

**Returns:** Heap-allocated `struct set *`. Free with `free_set()`.

---

### `free_set`
```c
void free_set(struct set *temp)
```
Frees all memory owned by a set struct including coordinate rows, the coords array, the labels array, and the struct itself. NULL-safe.

**When to use:** Whenever you are done with any `struct set *` returned by `create_subset()`, `ret_tempf()`, `copy_set()`, or any internal chain training function. Do not call on stack-allocated sets.

| Parameter | Type | Description |
|-----------|------|-------------|
| `temp` | `struct set *` | Set to free. Safe to pass NULL. |

---

## Utility / Preprocessing

---

### `covar_activ`
```c
float covar_activ(int firstdim, int seconddim, float data[firstdim][seconddim])
```
Computes the covariance of x and y coordinates across a dataset: `Cov(x,y) = mean((x - mean_x)(y - mean_y))`.

**When to use:** As a preprocessing diagnostic before training. High covariance suggests classes may be linearly separable along a diagonal. Not required for training.

| Parameter | Type | Description |
|-----------|------|-------------|
| `firstdim` | `int` | Number of rows. |
| `seconddim` | `int` | Number of columns (2 for 2D data). |
| `data` | `float[][]` | Coordinate array. |

**Returns:** Covariance as a float.

---

### `average_slope_group`
```c
float average_slope_group(int firstdim, int seconddim, float data[firstdim][seconddim])
```
Computes the average slope across consecutive point pairs in a dataset.

**When to use:** Exploratory preprocessing alongside `covar_activ()` to understand the general orientation of a dataset before training.

| Parameter | Type | Description |
|-----------|------|-------------|
| `firstdim` | `int` | Number of rows. |
| `data` | `float[][]` | Coordinate array, expected to be ordered. |

**Returns:** Average slope as a float.

---

### `float_equal`
```c
int float_equal(float a, float b)
```
Returns 1 if two floats are within 0.001 of each other, 0 otherwise.

**When to use:** Anywhere a tolerance-aware float comparison is needed. Used internally for clone detection in the perceptron chain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` / `b` | `float` | Values to compare. |

**Returns:** 1 if equal within tolerance, 0 otherwise.


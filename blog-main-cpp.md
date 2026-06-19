# Building a Neural Network From Scratch in C++: A Deep Dive Into `main.cpp`

*How I trained an MNIST digit classifier without a single ML library — just matrices, calculus, and raw C++.*

---

## Why build a neural network without a framework?

PyTorch and TensorFlow exist for a reason — they're fast, they're battle-tested, and `loss.backward()` is one line. So why write your own forward pass, your own backward pass, and your own matrix multiplication in C++ with nothing but the standard library?

Because that one line, `loss.backward()`, is hiding all the interesting parts. The goal of this project wasn't to build the *fastest* digit classifier — it was to build one where every gradient is a gradient I derived and wrote myself. This post walks through `main.cpp` top to bottom: the `Matrix` struct, the layers, forward propagation, backpropagation, and the training loop that ties it together.

By the end, this network reads a 28×28 image of a handwritten digit and predicts what it is — entirely from operations implemented by hand.

---

## The architecture at a glance

Before diving into code, here's the shape of the network we're building:

```
Input (784) → Linear(784, 128) → ReLU → Linear(128, 10) → Softmax → 10 class probabilities
```

- **Input layer**: a flattened 28×28 MNIST image (784 pixel values, normalized to [0, 1])
- **Hidden layer**: 128 neurons, ReLU activation
- **Output layer**: 10 neurons (one per digit, 0–9), softmax activation
- **Loss**: cross-entropy

It's a small, classic multilayer perceptron — but every piece of it, including the calculus, is hand-rolled.

---

## Step 1: The `Matrix` struct — the only data structure we need

```cpp
struct Matrix {
    int r, c;
    vector<vector<float>> data;

    Matrix(int rows=1, int columns=1) {
        r = rows;
        c = columns;
        data = vector<vector<float>>(r, vector<float>(c, 0.0f));
    }
};
```

Everything in this network — inputs, weights, biases, gradients, activations — is a `Matrix`. It's a thin wrapper around `vector<vector<float>>` with row/column counts attached. There's no GPU, no SIMD, no fancy memory layout — just nested vectors. That's a deliberate simplicity tradeoff: the code optimizes for *readability of the math*, not throughput.

Every neural net operation in this project is really just a function that takes one or two `Matrix` objects and returns a new one.

---

## Step 2: The building-block operations

### Matrix transpose

```cpp
Matrix transpose(Matrix x) {
    Matrix ans(x.c, x.r);
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++)
            ans.data[j][i] = x.data[i][j];
    return ans;
}
```

Standard transpose — flips rows and columns. This shows up constantly in backprop, because the gradient with respect to a layer's input requires multiplying by the *transpose* of its weight matrix (more on that below).

### Matrix product

```cpp
Matrix product(Matrix a, Matrix b) {
    Matrix ans(a.r, b.c);
    for (int i = 0; i < a.r; i++) {
        for (int k = 0; k < a.c; k++) {
            float aval = a.data[i][k];
            for (int j = 0; j < b.c; j++) {
                ans.data[i][j] += aval * b.data[k][j];
            }
        }
    }
    return ans;
}
```
This returns the usual product of 2 matrices

## Activations

### Why do we need activation functions?

Without activation functions, there's no point in having multiple layers — stacking
linear transformations just collapses into a single linear function, so the network
would behave like a single-layer model no matter how many layers you add. Activation
functions introduce non-linearity, which is what makes depth (multiple layers) actually
useful.

### ReLU and its derivative

```cpp
Matrix RelU(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++)
            ans.data[i][j] = max(0.0f, x.data[i][j]);
    return ans;
}

Matrix RelU_Derivative(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++)
            ans.data[i][j] = (x.data[i][j] > 0) ? 1.0f : 0.0f;
    return ans;
}
```

ReLU is `f(x) = max(0, x)` — dead simple, and a big part of why it's popular: cheap to compute and cheap to differentiate. Its derivative is 1 wherever the input was positive, and 0 otherwise. This derivative matrix gets multiplied element-wise against the upstream gradient during backprop (the chain rule in its most literal form) to "gate" the gradient — blocking it everywhere the neuron was inactive.

### Softmax (with the numerical stability trick)

```cpp
Matrix SoftMax(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < ans.r; i++) {
        float m = -INFINITY;
        for (int j = 0; j < ans.c; j++)
            m = max(m, x.data[i][j]);
        float sum = 0.0f;
        for (int j = 0; j < ans.c; j++) {
            float f = expf(x.data[i][j] - m);
            sum += f;
            ans.data[i][j] = f;
        }
        for (int j = 0; j < ans.c; j++)
            ans.data[i][j] /= sum;
    }
    return ans;
}
```

Softmax converts raw output scores ("logits") into a probability distribution over the 10 digit classes. Mathematically it's:

```
softmax(x_j) = exp(x_j) / Σ exp(x_k)
```

But there's a subtlety: `expf()` of a large number explodes a `float` very fast (`expf(100)` is already astronomically large). The fix, used here, is the **max-subtraction trick**: subtract the row's maximum value `m` from every element before exponentiating. This doesn't change the mathematical result (subtracting a constant from every input to softmax cancels out in the ratio), but it keeps every exponent ≤ 0, so `expf()` never overflows. This is a textbook numerical-stability pattern, and its presence here is one of the more "production-grade" details in the code.

### Element-wise multiply

```cpp
Matrix element_wise_multiply(Matrix a, Matrix b) {
    Matrix ans(a.r, a.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            ans.data[i][j] = a.data[i][j] * b.data[i][j];
    return ans;
}
```

Used specifically to apply the ReLU derivative as a gate during backprop (see below) — this is the Hadamard product, not matrix multiplication.

---

## Step 3: The `Layer` struct — bundling weights, biases, and gradients

```cpp
struct Layer {
    Matrix W, b, X, dW, db;

    Layer(int inputs, int outputs) :
        W(inputs, outputs), b(1, outputs), X(1, 1), dW(inputs, outputs), db(1, outputs) {
        initialize_random(W);
    }
    ...
};
```

Each `Layer` owns:
- `W` — its weight matrix (`inputs × outputs`)
- `b` — its bias row vector
- `X` — a cached copy of whatever input it last received (needed later for backprop)
- `dW`, `db` — gradient accumulators, same shape as `W` and `b`

This is the object-oriented heart of the network. Every dense ("fully connected") layer in the model — there are two of them — is an instance of this struct.

### Weight initialization

```cpp
void initialize_random(Matrix &m) {
    random_device rd;
    mt19937 gen(rd());
    float scale = sqrtf(2.0f / (m.r + m.c));
    uniform_real_distribution<float> dis(-scale, scale);
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            m.data[i][j] = dis(gen);
}
```

Weights aren't initialized to zero (that would make every neuron in a layer learn identically — a well-known failure mode) or to arbitrary large random values (that would cause exploding/vanishing activations). Instead, this uses a **Xavier/Glorot-style initialization**: weights are drawn uniformly from `[-scale, scale]` where `scale = sqrt(2 / (fan_in + fan_out))`. The intuition is to keep the variance of activations roughly consistent as data flows through the network, regardless of layer width — without this, deep-ish networks tend to either blow up or flatline within a few layers.

### Forward pass of a layer

```cpp
Matrix forward(Matrix Input) {
    X = Input;
    Matrix acc = product(Input, W);
    for (int i = 0; i < acc.r; i++)
        for (int j = 0; j < acc.c; j++)
            acc.data[i][j] += b.data[0][j];
    return acc;
}
```

This computes `Z = X·W + b` — the standard affine transformation of a dense layer. Two things worth noting:

1. **`X = Input` is cached.** The layer remembers its own input. This matters because the backward pass needs it — the gradient of the loss with respect to `W` depends on what was fed *into* this layer during the forward pass.
2. **The bias add is broadcast manually.** Since `b` is a `1 × outputs` row vector, it's added to every row of `acc` (every example in the batch) via an explicit loop — there's no NumPy-style broadcasting in raw C++, so this has to be done by hand.

### Backward pass of a layer — the heart of backprop

```cpp
Matrix backward(Matrix dZ) {
    dW = product(transpose(X), dZ);
    for (int j = 0; j < dZ.c; j++) {
        float sum = 0.0f;
        for (int i = 0; i < dZ.r; i++)
            sum += dZ.data[i][j];
        db.data[0][j] = sum;
    }
    return product(dZ, transpose(W));
}
```

This is the most important function in the file, so let's go slowly. `dZ` is the gradient of the loss with respect to this layer's *output* (i.e., "how much would the loss change if this output changed slightly"). The job of `backward()` is to turn that into three things:

1. **`dW` — the gradient w.r.t. this layer's weights.** Computed as `Xᵀ · dZ`. Intuitively: each weight `W[i][j]` connects input feature `i` to output `j`, so its gradient should depend on both how active input `i` was (`X`) and how much output `j`'s error matters (`dZ`). Transposing `X` and multiplying by `dZ` computes exactly that correlation, accumulated across the whole batch.
2. **`db` — the gradient w.r.t. the bias.** Since each bias is added identically to every example in the batch, its gradient is simply the *sum* of `dZ` down each column (summed over the batch dimension).
3. **The return value — the gradient to pass to the *previous* layer.** Computed as `dZ · Wᵀ`. This is the chain rule propagating backward: "how much does the loss change if the *input* to this layer changed," which becomes the upstream layer's own `dZ`.

This one function is a complete, general implementation of backpropagation through a single dense layer. Everything else in the network is just plumbing these calls together correctly.

### Applying the gradients

```cpp
void update_weights_biases(float lr, int batch_size) {
    for (int i = 0; i < W.r; i++)
        for (int j = 0; j < W.c; j++)
            W.data[i][j] -= lr * (dW.data[i][j] / batch_size);
    for (int i = 0; i < b.c; i++)
        b.data[0][i] -= lr * (db.data[0][i] / batch_size);
}
```

Plain mini-batch stochastic gradient descent: `W -= learning_rate × (average gradient over the batch)`. Dividing by `batch_size` here converts the *summed* gradient (accumulated across all examples in `backward()`) into an *average* gradient, so the learning rate's effective scale doesn't depend on batch size.

---

## Step 4: Wiring two layers into a network

```cpp
Layer L1(784, 128);
Layer L2(128, 10);
Matrix Z1;
Matrix A1;

Matrix forwardrun(Matrix Input) {
    Z1 = L1.forward(Input);
    A1 = RelU(Z1);
    Matrix Z2 = L2.forward(A1);
    return SoftMax(Z2);
}
```

`L1` and `L2` are declared as global `Layer` objects — `784 → 128` and `128 → 10`. `Z1` and `A1` (the hidden layer's pre- and post-activation values) are also kept as globals, because the backward pass needs them.

`forwardrun()` chains everything together: linear → ReLU → linear → softmax. This single function defines the entire model's prediction logic.

### The backward pass — and a clever shortcut

```cpp
Matrix backward_softmax_crossentropy(Matrix probs, Matrix target) {
    Matrix dZ(probs.r, probs.c);
    for (int i = 0; i < probs.r; i++)
        for (int j = 0; j < probs.c; j++)
            dZ.data[i][j] = probs.data[i][j] - target.data[i][j];
    return dZ;
}

void backwardrun(Matrix probs, Matrix target) {
    Matrix dZ2 = backward_softmax_crossentropy(probs, target);
    Matrix dA1 = L2.backward(dZ2);
    Matrix dZ1 = element_wise_multiply(dA1, RelU_Derivative(Z1));
    L1.backward(dZ1);
}
```

Here's a detail that's easy to miss but is genuinely elegant: there's no function anywhere in this file that computes the softmax Jacobian, and no function that computes the derivative of cross-entropy loss in isolation. Instead, `backward_softmax_crossentropy` just computes `probs - target`.

This isn't a simplification for convenience — it's a real mathematical identity. When softmax is the output activation and cross-entropy is the loss, the gradient of the *combined* loss with respect to the *pre-softmax* logits collapses, after the dust settles, to exactly `predicted_probabilities - true_labels` (where `target` is a one-hot vector). Deriving the softmax Jacobian and the cross-entropy gradient separately and multiplying them through the chain rule would give you the same answer, but with far more arithmetic — most of which cancels. This is a standard trick in neural net implementations, and using it here means the code never needs the messy general softmax derivative at all.

From there, `backwardrun` is just `Layer::backward()` calls chained in reverse order, with one extra step in the middle: `element_wise_multiply(dA1, RelU_Derivative(Z1))` applies the ReLU gate — zeroing out gradient flow through any hidden neuron that was inactive (≤ 0) during the forward pass.

---

## Step 5: Loading MNIST

```cpp
bool load_mnist_text(string path_img, string path_lbl, Matrix &images, Matrix &labels, int count) {
    ifstream f_img(path_img);
    ifstream f_lbl(path_lbl);
    if (!f_img.is_open() || !f_lbl.is_open()) return false;

    images = Matrix(count, 784);
    labels = Matrix(count, 10);

    stringstream img_buffer, lbl_buffer;
    img_buffer << f_img.rdbuf();
    lbl_buffer << f_lbl.rdbuf();
    f_img.close();
    f_lbl.close();

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 784; j++) {
            int val; img_buffer >> val;
            images.data[i][j] = (float)val / 255.0f;
        }
        int lbl; lbl_buffer >> lbl;
        labels.data[i][lbl] = 1.0f;
    }
    return true;
}
```

Rather than parsing MNIST's original IDX binary format, this expects a simplified plain-text representation: one file with whitespace-separated pixel values, one row per image, and a parallel file with one integer label per line.

Two details worth flagging:
- **Pixel normalization**: raw pixel values (0–255) are divided by 255, scaling them into `[0, 1]`. Neural nets train far more reliably on small, normalized inputs than on raw byte ranges — this keeps activations and gradients in a numerically sane range from the very first layer.
- **One-hot encoding**: labels aren't stored as the integer digit (e.g. `7`) but as a 10-wide vector with a single `1.0` at index 7 and zeros elsewhere. This is required because the network's output is a 10-way probability distribution — cross-entropy loss needs the target in the same shape to compare against.

The whole file is slurped into a `stringstream` first (`img_buffer << f_img.rdbuf()`) rather than reading line-by-line — a simple way to avoid the overhead of repeated small reads from disk.

---

## Step 6: The training loop

```cpp
int main() {
    Matrix train_images, train_labels;
    Matrix test_images, test_labels;

    int train_size = 10000;
    int test_size = 2000;
    int batch_size = 64;
    float learning_rate = 0.1f;
    int epochs = 20;
    ...
```

The hyperparameters: 10,000 training images, 2,000 held-out test images, mini-batches of 64, a learning rate of 0.1, run for 20 full passes (epochs) over the training data.

### Building each mini-batch

```cpp
for (int i = 0; i < train_size; i += batch_size) {
    int current_batch_size = min(batch_size, train_size - i);

    Matrix x_batch(current_batch_size, 784);
    Matrix y_batch(current_batch_size, 10);

    for (int b = 0; b < current_batch_size; b++) {
        x_batch.data[b] = train_images.data[i + b];
        y_batch.data[b] = train_labels.data[i + b];
    }
    ...
```

The outer loop strides through the training set in chunks of `batch_size`. `current_batch_size` handles the edge case where `train_size` isn't an exact multiple of `batch_size` (the last batch is simply smaller). Each mini-batch is assembled by copying the relevant rows out of the full training matrix — a row-vector-copy by way of `vector`'s assignment operator (`x_batch.data[b] = train_images.data[i + b]`).

### One training step

```cpp
Matrix probs = forwardrun(x_batch);

for (int b = 0; b < current_batch_size; b++) {
    int true_digit = 0, pred_digit = 0;
    float max_p = -1.0f;
    for (int j = 0; j < 10; j++) {
        if (y_batch.data[b][j] == 1.0f) true_digit = j;
        if (probs.data[b][j] > max_p) {
            max_p = probs.data[b][j];
            pred_digit = j;
        }
    }
    if (true_digit == pred_digit) correct_train++;
}

backwardrun(probs, y_batch);
L1.update_weights_biases(learning_rate, current_batch_size);
L2.update_weights_biases(learning_rate, current_batch_size);
```

Each step is the classic four-beat rhythm of supervised learning:

1. **Forward pass** — run the batch through the network to get predicted probabilities.
2. **Accuracy tracking** — for bookkeeping only (not used in training), find the `argmax` of both the true one-hot label and the predicted probabilities, and count a hit if they match.
3. **Backward pass** — compute gradients for every weight and bias via `backwardrun`.
4. **Update** — nudge every weight and bias a small step (`learning_rate`) opposite to its gradient, finishing the loop.

After all batches in an epoch, training accuracy is printed:

```cpp
float train_acc = ((float)correct_train / train_size) * 100.0f;
cout << "Epoch " << epoch + 1 << " - Training Accuracy: " << train_acc << "%" << endl;
```

### Final evaluation and saving weights

```cpp
int correct_test = 0;
Matrix test_probs = forwardrun(test_images);
...
cout << "Final Test Accuracy: " << ((float)correct_test / test_size) * 100.0f << "%" << endl;

save_matrix_text("./data/W1.txt", L1.W);
save_matrix_text("./data/b1.txt", L1.b);
save_matrix_text("./data/W2.txt", L2.W);
save_matrix_text("./data/b2.txt", L2.b);
```

After training, the *entire* test set is run through `forwardrun` in one shot, and the final accuracy is reported. This is the network's real performance number: accuracy on data it never trained on.

Finally, the four learned parameter matrices (`W1`, `b1`, `W2`, `b2`) are stored in text files — space-separated rows, one matrix per file using `save_matrix_text`.

---

## What this implementation gets right

- **Correct backprop math**, including the softmax + cross-entropy gradient shortcut — not approximated, not hand-waved.
- **Numerically stable softmax** via max-subtraction — a detail many from-scratch implementations skip and later regret.
- **Sound weight initialization** (Xavier-style scaling) instead of naive random or zero init.
- **Cache-aware matrix multiplication** loop ordering.
- **Clean separation of concerns**: pure math functions (`product`, `transpose`, `RelU`, `SoftMax`) are stateless; the `Layer` struct is the only place that owns mutable state (weights, cached input, gradients).

## What it deliberately leaves out

This is a teaching/learning implementation, not a production training framework, and it shows in a few honest ways:

- No regularization (no dropout, no L2 weight decay) — so given enough epochs, this could overfit the small training set.
- No learning-rate schedule — a flat `0.1` for all 20 epochs.
- No validation set distinct from the test set used for the final number — test accuracy here is being used the way a validation set normally would.
- Single-threaded, no SIMD/BLAS — the `product()` function is `O(n³)` nested loops, which is fine at this scale (a few hundred neurons) but wouldn't scale to a real-sized model.

None of these are bugs — they're the right scope for a project whose purpose is to *understand* the algorithm, not to compete with cuDNN.

---

## Closing thoughts

What makes this file worth reading isn't that it does anything exotic — it's a vanilla two-layer MLP trained with vanilla SGD. What makes it worth reading is that *nothing is hidden*. The softmax-crossentropy gradient shortcut, the Xavier initialization scale, the manual bias broadcasting, the cache-friendly loop order — every one of these is a small, deliberate decision that a framework would normally make invisibly on your behalf.

Writing it by hand is slower. But it's the difference between knowing that backpropagation works and knowing *why* it works — and that gap is exactly what this project was built to close.

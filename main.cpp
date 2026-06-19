#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sstream>

using namespace std;

struct Matrix {
    int r, c;
    vector<vector<float>> data;

    Matrix(int rows=1, int columns=1) {
        r = rows;
        c = columns;
        data = vector<vector<float>>(r, vector<float>(c, 0.0f));
    }
};

Matrix transpose(Matrix x) {
    Matrix ans(x.c, x.r);
    for (int i = 0; i < x.r; i++) {
        for (int j = 0; j < x.c; j++) {
            ans.data[j][i] = x.data[i][j];
        }
    }
    return ans;
}

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

Matrix RelU(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < x.r; i++) {
        for (int j = 0; j < x.c; j++) {
            ans.data[i][j] = max(0.0f, x.data[i][j]);
        }
    }
    return ans;
}

Matrix RelU_Derivative(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < x.r; i++) {
        for (int j = 0; j < x.c; j++) {
            ans.data[i][j] = (x.data[i][j] > 0) ? 1.0f : 0.0f;
        }
    }
    return ans;
}

Matrix SoftMax(Matrix x) {
    Matrix ans(x.r, x.c);
    for (int i = 0; i < ans.r; i++) {
        float m = -INFINITY;
        for (int j = 0; j < ans.c; j++) {
            m = max(m, x.data[i][j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < ans.c; j++) {
            float f = expf(x.data[i][j] - m); 
            sum += f;
            ans.data[i][j] = f;
        }
        for (int j = 0; j < ans.c; j++) {
            ans.data[i][j] /= sum;
        }
    }
    return ans;
}

Matrix element_wise_multiply(Matrix a, Matrix b) {
    Matrix ans(a.r, a.c);
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < a.c; j++) {
            ans.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return ans;
}

Matrix backward_softmax_crossentropy(Matrix probs, Matrix target) {
    Matrix dZ(probs.r, probs.c);
    for (int i = 0; i < probs.r; i++) {
        for (int j = 0; j < probs.c; j++) {
            dZ.data[i][j] = probs.data[i][j] - target.data[i][j];
        }
    }
    return dZ;
}

void initialize_random(Matrix &m) {
    random_device rd;
    mt19937 gen(rd());
    float scale = sqrtf(2.0f / (m.r + m.c));
    uniform_real_distribution<float> dis(-scale, scale);
    for (int i = 0; i < m.r; i++) {
        for (int j = 0; j < m.c; j++) {
            m.data[i][j] = dis(gen);
        }
    }
}

struct Layer {
    Matrix W, b, X, dW, db;

    Layer(int inputs, int outputs) : 
        W(inputs, outputs), b(1, outputs), X(1, 1), dW(inputs, outputs), db(1, outputs) {
        initialize_random(W);
    }

    Matrix forward(Matrix Input) {
        X = Input;
        Matrix acc = product(Input, W);
        for (int i = 0; i < acc.r; i++) {
            for (int j = 0; j < acc.c; j++) {
                acc.data[i][j] += b.data[0][j];
            }
        }
        return acc;
    }

    Matrix backward(Matrix dZ) {
        dW = product(transpose(X), dZ);
        for (int j = 0; j < dZ.c; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dZ.r; i++) {
                sum += dZ.data[i][j];
            }
            db.data[0][j] = sum;
        }
        return product(dZ, transpose(W));
    }

    void update_weights_biases(float lr, int batch_size) {
        for (int i = 0; i < W.r; i++) {
            for (int j = 0; j < W.c; j++) {
                W.data[i][j] -= lr * (dW.data[i][j] / batch_size);
            }
        }
        for (int i = 0; i < b.c; i++) {
            b.data[0][i] -= lr * (db.data[0][i] / batch_size);
        }
    }
};

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

void backwardrun(Matrix probs, Matrix target) {
    Matrix dZ2 = backward_softmax_crossentropy(probs, target);
    Matrix dA1 = L2.backward(dZ2);
    Matrix dZ1 = element_wise_multiply(dA1, RelU_Derivative(Z1));
    L1.backward(dZ1);
}

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

void save_matrix_text(string path, Matrix m) {
    ofstream file(path);
    for (int i = 0; i < m.r; i++) {
        for (int j = 0; j < m.c; j++) {
            file << m.data[i][j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    Matrix train_images, train_labels;
    Matrix test_images, test_labels;

    int train_size = 10000;
    int test_size = 2000;
    int batch_size = 64;
    float learning_rate = 0.1f;
    int epochs = 20;

    cout << "Loading Train & Test datasets..." << endl;
    if (!load_mnist_text("./data/mnist_images.txt", "./data/mnist_labels.txt", train_images, train_labels, train_size) ||
        !load_mnist_text("./data/mnist_test_images.txt", "./data/mnist_test_labels.txt", test_images, test_labels, test_size)) {
        cout << "Error loading file matrices!" << endl;
        return 1;
    }

    cout << "\nStarting Mini-Batch Training Loop..." << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct_train = 0;

        for (int i = 0; i < train_size; i += batch_size) {
            int current_batch_size = min(batch_size, train_size - i);

            Matrix x_batch(current_batch_size, 784);
            Matrix y_batch(current_batch_size, 10);

            for (int b = 0; b < current_batch_size; b++) {
                x_batch.data[b] = train_images.data[i + b];
                y_batch.data[b] = train_labels.data[i + b];
            }

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
        }
        float train_acc = ((float)correct_train / train_size) * 100.0f;
        cout << "Epoch " << epoch + 1 << " - Training Accuracy: " << train_acc << "%" << endl;
    }

    int correct_test = 0;
    Matrix test_probs = forwardrun(test_images);
    for (int i = 0; i < test_size; i++) {
        int true_digit = 0, pred_digit = 0;
        float max_p = -1.0f;
        for (int j = 0; j < 10; j++) {
            if (test_labels.data[i][j] == 1.0f) true_digit = j;
            if (test_probs.data[i][j] > max_p) {
                max_p = test_probs.data[i][j];
                pred_digit = j;
            }
        }
        if (true_digit == pred_digit) correct_test++;
    }
    cout << "Final Test Accuracy: " << ((float)correct_test / test_size) * 100.0f << "%" << endl;

    cout << "\nSaving trained weights to disk..." << endl;
    save_matrix_text("./data/W1.txt", L1.W);
    save_matrix_text("./data/b1.txt", L1.b);
    save_matrix_text("./data/W2.txt", L2.W);
    save_matrix_text("./data/b2.txt", L2.b);
    cout << "Weights saved successfully! Ready for webcam app." << endl;

    return 0;
}
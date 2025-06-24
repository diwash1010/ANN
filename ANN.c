#include <stdio.h>
#define INPUT_NEURONS 2
#define EPOCHS 10
#define LEARNING_RATE 0.1
int activation(double z) {
    return z >= 0 ? 1 : 0;
}
int main() {

    double X[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    int y[4] = {0, 0, 0, 1}
    double weights[2] = {0.0, 0.0};
    double bias = 0.0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Epoch %d\n", epoch + 1);
        for (int i = 0; i < 4; i++) {
            double weighted_sum = X[i][0] * weights[0] + X[i][1] * weights[1] + bias;
            int y_pred = activation(weighted_sum);
            int error = y[i] - y_pred;


            weights[0] += LEARNING_RATE * error * X[i][0];
            weights[1] += LEARNING_RATE * error * X[i][1];
            bias += LEARNING_RATE * error;

            printf("Sample [%g, %g] | Target: %d | Predicted: %d | Updated Weights: [%.2f, %.2f] Bias: %.2f\n",
                X[i][0], X[i][1], y[i], y_pred, weights[0], weights[1], bias);
        }
        printf("\n");
    }

    printf("Final Trained Model for AND Gate:\n");
    for (int i = 0; i < 4; i++)
    {
        double weighted_sum = X[i][0] * weights[0] + X[i][1] * weights[1] + bias;
        int y_pred = activation(weighted_sum);
        printf("Input: [%d, %d] => Output: %d\n", (int)X[i][0], (int)X[i][1], y_pred);
    }
    return 0;
}

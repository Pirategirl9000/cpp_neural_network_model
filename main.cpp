// Online C++ compiler to run C++ program online
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <tuple>

using namespace std;

float getRandFloat() {
    srand(time(NULL));
    this_thread::sleep_for(chrono::milliseconds(750));
    return (rand() % 100 + 1) * 0.01;
}

float sigmoid(float x) {
    return 1/1+exp(-x);
}

float sigmoid_der(float x) {
    return sigmoid(x) * sigmoid(1-x);
}

float tanhyper(float x) {
    return tanh(x);
}

float tanhyper_der(float x) {
    return 1 - (tanh(x) * tanh(x));
}

tuple<float/*x2*/, float/*x3*/, float/*prediction*/> forward_propagate(float x1, float w1, float w2, float w3, float b1, float b2, float b3) {
    
    
    float x2 = sigmoid((x1 * w1) + b1);
    float x3 = tanhyper((x2 * w2) + b2);
    float pred =(x3 * w3) + b3;
    
    return make_tuple(x2, x3, pred);
}

tuple<float/*w1*/, float/*w2*/, float/*w3*/, float/*b1*/, float/*b2*/, float/*b3*/, float/*loss*/> backward_propagate(float x1, float x2, float x3, float y, float pred, float w1, float w2, float w3, float b1, float b2, float b3) {
    
    float loss = y - pred;
    
    float der1 = sigmoid_der((x1 * w1) + b1);
    float der2 = tanhyper_der((x2 * w2) + b2);
    float der3 = w3; //linear function has slope as derivative
    
    float dw3 = loss * x3 * der3 * 0.01;
    float dw2 = dw3 * x2 * der2 * 0.01;
    float dw1 = dw2 * x1 * der1 * 0.01;
    
    float db3 = loss * der3 * 0.01;
    float db2 = db3 * der2 * 0.01;
    float db1 = db2 * der1 * 0.01;
  
    return make_tuple(w1+dw1, w2+dw2, w3+dw3, b1+db1, b2+db2, b3+db3, loss);
    
}

int main() {
    srand(time(NULL));
    
    float x1 = 6, x2 = 0, x3 = 0;
    float y = 15;
    float pred = 0;
    float loss = 0;
    float w1 = getRandFloat(), w2 = getRandFloat(), w3 = getRandFloat();
    float b1 = 1, b2 = 1, b3 = 1;
    for (int i = 0; i < 100; i++) {
        tie(x2, x3, pred) = forward_propagate(x1, w1, w2, w3, b1, b2, b3);
        tie(w1, w2, w3, b1, b2, b3, loss) = backward_propagate(x1, x2, x3, y, pred, w1, w2, w3, b1, b2, b3);
        
        if (i % 10 == 0) {
            cout << "Prediction: " << pred << endl << "Loss: " << loss << endl;
        }
    }
    
    return 0;
}

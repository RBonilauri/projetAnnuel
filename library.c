#include <stdio.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

DLLEXPORT float* create_linear_model(int len){
    float* tab;
    tab = malloc(sizeof(float) * len);
    if (tab == NULL) {
        printf("Run out of memory ! \n");
        exit(1);
    }
    srand(time(NULL));
    float temp = (float) rand();
    for(int i = 0; i < len; i++){

        tab[i] = ((float)rand()/(RAND_MAX/2)) - 1.0;

    }
    return tab;
}

DLLEXPORT float predict_linear_model_regression(float* model, float* sample_inputs, int len) {
    float result = model[0] * 1.0;
    int i = 1;
    while(i < len) {
        result += model[i] * sample_inputs[i-1];
        i++;
    }
    return result;
}

DLLEXPORT float predict_linear_model_classification(float* model, float * sample_inputs, int len){
    if (predict_linear_model_regression(model,sample_inputs,len) >= 0) return 1.0;
    else return -1.0;
}

float* cut_tab(float* tab, int first, int last) {
    int len = last-first;
    float new_tab[len] ;
    int i = 0;
    while (i < len) {
        new_tab[i] = tab[first+i];
        i++;
    }
    return new_tab;
}

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float* model, float* flattened_dataset_inputs, float* flattened_dataset_expected_outputs,float alpha, int iterations_count, int model_len, int flattened_inputs_len) {
    int input_dim = model_len - 1;
    int samples_count = flattened_inputs_len / input_dim;

    int i = 0;
    while (i < iterations_count) {
        srand(time(NULL));
        int k = ((int) rand() % (samples_count - 1));

        int first = k * input_dim;
        int last = (k+1) * input_dim;
        float* Xk = cut_tab(flattened_dataset_inputs, first, last);

        float Yk = flattened_dataset_expected_outputs[k];
        float gXk = predict_linear_model_classification(model, Xk, model_len);
        model[0] += alpha * (Yk-gXk) * 1.0;
        int i = 1;
        while (i < model_len) {
            model[i] += alpha * (Yk - gXk) * Xk[i-1];
            i++;
        }
    }
}

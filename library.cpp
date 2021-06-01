#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;
using Eigen::MatrixXd;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT
#endif

//Changer la manière de recuperer la dimension d'un tableau et ne plus le passer en paramètre


/*
 * Définitions du Modèle linéaire
 */
DLLEXPORT float* create_linear_model(int len){
    float* tab;
    tab = new float[len];
    if (tab == nullptr) {
        printf("Run out of memory ! \n");
        exit(1);
    }
    srand(time(NULL));
    rand();
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

float* cut_tab(const float* tab, int first, int last) {
    int len;
    len = last - first;
    auto* new_tab = new float[len];
    int i = 0;
    while (i < len) {
        new_tab[i] = tab[first+i];
        i++;
    }
    return new_tab;
}



DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float* model, float* flattened_dataset_inputs, float* flattened_dataset_expected_outputs,float alpha, int iterations_count, int model_len, int flattened_inputs_len) {
    // Attention alpha et iterations_count sont initialisé respectivement à 0.001 et 20
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
        int j = 1;
        while (j < model_len) {
            model[i] += alpha * (Yk - gXk) * Xk[j-1];
            j++;
        }
        i++;
    }
}

//DLLEXPORT int getLength(float* model) {
//    int result = sizeof(model)/sizeof(model[0]);
//    return result;
//}

MatrixXd transformTab(float* tab,int tab_len) {
    int y = tab_len;
    MatrixXd a(1,y);
    for (int i = 0; i < y; i++) {
        a(0,i) = tab[i];
    }
    return a;
}

MatrixXd ones(int rows) {
    MatrixXd result(rows,1);
    for(int i = 0; i < rows; i++) {
        result(i,0) = 1.0;
    }
    return result;
}

MatrixXd reshape(MatrixXd tab, int rows, int col) {
    Map<MatrixXd> temp(tab.data(), col, rows);
    return temp.transpose();
}

MatrixXd hstack(MatrixXd one, MatrixXd X) {
    MatrixXd C(X.rows(), one.cols()+X.cols());
    C << one, X;
    return C;
}

DLLEXPORT void train_regression_pseudo_inverse_linear_model(float* model, float* flattened_dataset_inputs, float* flattened_dataset_expected_outputs,int model_len, int flattened_dataset_inputs_len, int flattened_dataset_expected_outputs_len) {
    int input_dim = model_len - 1;
    int samples_count = flattened_dataset_inputs_len / input_dim;
    MatrixXd X = transformTab(flattened_dataset_inputs, flattened_dataset_inputs_len);
    MatrixXd Y = transformTab(flattened_dataset_expected_outputs, flattened_dataset_expected_outputs_len);

    X = reshape(X, samples_count, input_dim);
    MatrixXd one = ones(samples_count);
    X = hstack(one, X);
    Y = reshape(Y, samples_count, 1);
    MatrixXd W = (((X.transpose() * X).inverse()) * X.transpose()) * Y;

    for(int i = 0; i < model_len; i++){
        model[i] = W(i,0);
    }
}

DLLEXPORT void destroy_linear_model(float* model) {
    delete (model);
}

/*
 * Définitions PMC
 */

DLLEXPORT typedef struct MLP_m {
    vector<vector<vector<float>>> W;
    vector<int> d;
    vector<vector<float>> X;
    vector<vector<float>> deltas;

    void forward_pass(float* sample_inputs, bool is_classification) {
        int L = d.size() - 1;

        for (int j = 1; j < (d[0] + 1); j++) {
            X[0][j] = sample_inputs[j-1];
        }

        for (int l = 1; l < (L+1); l++) {
            for (int j = 1; j < d[l] + 1; j++) {
                float sum_result = 0.0;
                for (int i = 0; i < (d[l-1] + 1); i++) {
                    sum_result += W[l][i][j] * X[l-1][i];
                }
                X[l][j] = sum_result;
                if (is_classification || l < L) {
                    X[l][j] = tanh(X[l][j]);
                }
            }
        }
    }

    void train_stochastic_gradient_backpropagation(float* flattened_dataset_inputs, int flattened_dataset_inputs_len, float* flattened_dataset_expected_outputs, bool is_classification, float alpha=0.001, int iterations_count=100000){
        int input_dim = d[0];
        int output_dim = d[d.size()-1];
        int sample_count = flattened_dataset_inputs_len / input_dim;
        int L = d.size() - 1;
        srand(time(NULL));
        rand();

        for(int it =1; it < iterations_count; it ++){
            int k = rand()%(sample_count);

            float* sample_input = cut_tab(flattened_dataset_inputs, (k * input_dim), ((k+1) * input_dim));
            float* sample_expected_output = cut_tab(flattened_dataset_expected_outputs, (k * output_dim), ((k+1) * output_dim));
            forward_pass(sample_input, is_classification);

            for(int j = 1; j < d[L] + 1; j++){
                deltas[L][j]=X[L][j] - sample_expected_output[j-1];
                if(is_classification)
                    deltas[L][j] *= (1 - X[L][j] * X[L][j]);
            }

            for(int l = L + 1; l > 0 ; l--){
                for(int i = 1; i < d[l - 1] + 1; i++){
                    float sum_result = 0.0;
                    for(int j = 1; j < d[l] + 1; j++){
                        sum_result += W[l][i][j] * deltas[l][j];
                    }
                    deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * sum_result;
                }
            }

            for(int l = 0; l < L + 1; l++){
                for(int i = 0; i < d[l-1] + 1; i++){
                    for(int j = 1; j < d[l] + 1; j++){
                        W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                    }

                }
            }
        }
    }
    
}MLP;

DLLEXPORT MLP* create_mlp_model(int* npl, int npl_len) {
    MLP* model = new MLP[1];
    srand(time(NULL));
    rand();
    for(int l = 0; l < npl_len; l++) {
        model->W.push_back(vector<vector<float>>(0));
        if (l == 0) {
            continue;
        } else {
            for(int i = 0; i < (npl[l-1] + 1); i++) {
                model->W[l].push_back(vector<float>(npl[l] + 1));
                for(int j = 0; j < npl[l] + 1; j++) {
                    model->W[l][i][j] = ((float)rand()/(RAND_MAX/2)) - 1.0;
                }
            }
        }

    }

    for(int i = 0; i < npl_len; i++) {
        model->d.push_back(npl[i]);
    }

    for(int l = 0; l < npl_len; l++) {
        model->X.push_back(vector<float>(0));
        for(int j = 0; j < (npl[l] + 1); j++) {
            if (j == 0) {
                model->X[l].push_back(1.0);
            } else {
                model->X[l].push_back(0.0);
            }
        }
    }

    for(int l = 0; l < npl_len; l++) {
        model->deltas.push_back(vector<float>(0));
        for(int j = 0; j < (npl[l] + 1); j++) {
            model->deltas[l].push_back(0.0);
        }
    }
    
    return model;
}

DLLEXPORT float* predict_mlp_model_regression(MLP model, float* sample_inputs) {
    model.forward_pass(sample_inputs, false);
    int dernier_indice = (model.X.size()) - 1;
    int taille = (model.X[dernier_indice].size()) - 1 ;
    auto newtab = new float[taille];

    for (int i = 0; i < taille; i++) {
        newtab[i] = model.X[dernier_indice][i+1];
    }

    return newtab;

}

DLLEXPORT float* predict_mlp_model_classification(MLP model, float* sample_inputs) {
    model.forward_pass(sample_inputs, true);
    int dernier_indice = (model.X.size()) - 1;
    int taille = (model.X[dernier_indice].size()) - 1 ;
    auto newtab = new float[taille];

    for (int i = 0; i < taille; i++) {
        newtab[i] = model.X[dernier_indice][i+1];
    }

    return newtab;

}

DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP model, float* flattened_dataset_inputs, int flattened_dataset_inputs_len, float* flattened_dataset_expected_outputs, float alpha = 0.001, int iterations_count = 100000) {
    model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_dataset_inputs_len, flattened_dataset_expected_outputs, true, alpha, iterations_count);
}

DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP model, float* flattened_dataset_inputs, int flattened_dataset_inputs_len, float* flattened_dataset_expected_outputs, float alpha = 0.001, int iterations_count = 100000) {
    model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_dataset_inputs_len,flattened_dataset_expected_outputs, false, alpha, iterations_count);
}


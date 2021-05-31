#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;
using Eigen::MatrixXd;
using Eigen::MatrixXi;


MatrixXd transformTab(double* tab,int tab_len) {
    int y = tab_len;
    MatrixXd a(1,y);
    for (int i = 0; i < y; i++) {
        a(0,i) = tab[i];
    }
    return a;
}


int getLength(double* model) {
    int a = sizeof(model);
    cout << "a: " << a << endl;
    int b = sizeof(model[0]);
    cout << "b: " << b << endl;
    return a/b;
}

MatrixXd reshape(MatrixXd tab, int rows, int col) {
    Map<MatrixXd> temp(tab.data(), col, rows);
    return temp.transpose();
}

MatrixXd ones(int rows) {
    if (rows >= 0) {
        MatrixXd result(rows, 1);
        for (int i = 0; i < rows; i++) {
            result(i, 0) =  (double) 5.0 ;
        }
        return result;
    } else {
        MatrixXd result(0,0);
        return result;
    }

}

MatrixXd hstack(MatrixXd one, MatrixXd X) {
    MatrixXd C(X.rows(), one.cols()+X.cols());
    C << one, X;
    return C;
}

typedef struct MLP_m {
    vector<vector<vector<float>>> W;
    vector<int> d;
    vector<vector<float>> X;
    vector<vector<float>> deltas;


}MLP;

MLP* create_mlp_model(int* npl, int npl_len) {
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
                cout << "here " << endl;
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


int main(){
    double tab[6] = {2.2,3.4,1.4,5.1,2.7,7.3};

    int npl[3] = {2,2,1};


    MLP* models = create_mlp_model(npl, 3);
    cout << "Premier :" << models->W[1][1][0] << endl;
    cout << "npl :" << models->d[2] << endl;
    cout << "X :" << models->X[1][0] << endl;

    return 0;
};

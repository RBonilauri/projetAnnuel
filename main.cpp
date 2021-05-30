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
                model->W[1].push_back(vector<float>(npl[l] + 1));
                for(int j = 0; j < npl[l] + 1; j++) {
                    model->W[l][i][j] = ((float)rand()/(RAND_MAX/2)) - 1.0;
                }
            }
        }

    }

//    model->W.push_back(vector<vector<float>>(0));
//    model->W.push_back(vector<vector<float>>(0));
//    model->W[1].push_back(vector<float>(2));
//    model->W[1][0][0] = 1;
//    model->W[1][0][1] = 2;
    return model;

}


//dans le npl :
/* sa longueur fait reference au nombre de layer: ex: len = 3  [[],[],[]],
 * le premier chiffre dans le tableau npl fait réference au nombre de
 *
 *
 */


int main(){
    double tab[6] = {2.2,3.4,1.4,5.1,2.7,7.3};
//    MatrixXd a = transformTab(tab, 6);
//    cout << "Premier tab =\n" << a << endl;
//    MatrixXd reshaped = reshape(a, 3,2);
//    cout << "Premier tab = \n" << reshaped << endl;
//    MatrixXd test = ones(3);
//    std::cout << "One : \n"<< test << endl;
//    MatrixXd C = hstack(test, reshaped);
//    std::cout << "result : \n"<< C << endl;
//    MatrixXd A(2,2);
//    A << 4,2,2,3;
//    MatrixXd B(2,2);
//    B << 3,5,4,8;
//
//    cout << A << endl;
//    cout << B <<endl;
//    cout << A*B << endl;
    int npl[2];
    npl[0] = 2;
    npl[1] = 2;

    MLP* models = create_mlp_model(npl, 2);
    cout << "Premier :" << models->W[1][1][2] << endl;

    return 0;
};

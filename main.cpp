#include "Surface_response.hpp"
#include <iostream>
#include <chrono>

#include <fstream>
#include <Eigen/Dense>

int main() {

    Eigen::Matrix<double, 3*Nstations, NSubFaults> G;

    std::ifstream file("../../FortranCodes_Myriam/GreensFunctionsV1/G_matrix.txt");
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier." << std::endl;
        return 1;
    }

    for (int i = 0; i < G.rows(); ++i) {
            for (int j = 0; j < G.cols(); ++j) {
                if (!(file >> G(i, j))) {
                    std::cerr << "Erreur : Format de fichier incorrect ou données insuffisantes." << std::endl;
                    return 1;
                }
            }
        }

    file.close();




    /*
    // Calcul optimisé
    C.noalias() = A * B;

    // Affichage du premier élément pour valider (doit valider 450 * 1 * 2 = 900)
    std::cout << "Element C(0,0) = " << C(0,0) << " (Attendu: 900)" << std::endl;
        */
    return 0;
}
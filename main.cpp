#include "Surface_response.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main() {

    std::cout<<"extract G : " ;
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

    std::cout<<"done \n";
    
    std::cout<<"define data storage : ";
    const int nb_threads = omp_get_max_threads();
    
    //omp_set_num_threads(1);
    //int nb_threads=1;
    

    std::cout<<"\nnb_threads : "<<nb_threads<<"\n";

    std::vector<sunrealtype> t_list(300);
    for (int i=0; i<300; ++i){
        t_list[i]=5.0/300.0*i;
    }



    std::vector<Param> pP;
    pP.reserve(NSubFaults);

    
    for (int i=0; i<NSubFaults ; i++){
            double k_variable = 0.01 + 0*(i * 0.001); 
            pP.emplace_back(k_variable, 0.4, 0.17, 0.1, 0.08 * 100.0 / (365.0 * 24.0), 2.0);
    }

    std::vector<Fault> Fault_per_thread;
    Fault_per_thread.resize(nb_threads);

    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
    Eigen::Matrix<double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> storage_matrix(NSubFaults, t_list.size());
    

    std::cout<<" done \n";
    std::cout<<"compute surface displacement : ";
    auto timeStart = std::chrono::high_resolution_clock::now();
    
    for (int j=0;j<1000;j++){
    surface_response(pP,  t_list,  Fault_per_thread, G, RES_matrix,  storage_matrix);}

    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = timeEnd - timeStart;
    double timeTotal = duration.count();
    std::cout<<timeTotal;

    // --- Début du bloc d'exportation CSV --- (écrit par une ia j'avais la flemme)
    std::ofstream csv_file("surface_responses.csv");
    if (csv_file.is_open()) {
        // En-tête décrivant chaque colonne
        csv_file << "Time,St1_North,St1_East,St1_Depth,St2_North,St2_East,St2_Depth\n";
        
        // Écriture ligne par ligne (chaque ligne = un pas de temps)
        for (Eigen::Index j = 0; j < RES_matrix.cols(); ++j) {
            csv_file << t_list[j];
            for (Eigen::Index i = 0; i < RES_matrix.rows(); ++i) {
                csv_file << "," << RES_matrix(i, j);
            }
            csv_file << "\n";
        }
        csv_file.close();
        std::cout << "\nFichier CSV généré avec succès.\n";
    } else {
        std::cerr << "\nErreur : Impossible de créer le fichier CSV.\n";
    }
    // --- Fin du bloc d'exportation CSV ---
    return 0;
}
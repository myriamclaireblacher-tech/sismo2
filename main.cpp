#include "inversion.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main() {

    /*
    -----------------------------------------------------------------------------------------------------------------------
                        EXTRACT GREEN MATRIX
    -----------------------------------------------------------------------------------------------------------------------
    */
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

    /*
    -----------------------------------------------------------------------------------------------------------------------
                        PREPARE STORAGE
    -----------------------------------------------------------------------------------------------------------------------
    */
    
    std::cout<<"define data storage : ";
    //const int nb_threads = omp_get_max_threads();
    const int nb_threads = NCPU ;
    
    omp_set_num_threads(NCPU);
    //int nb_threads=1;
    
    std::cout<<"\nnb_threads : "<<nb_threads<<"\n";

    //t_list
    std::vector<sunrealtype> t_list(300);
    for (int i=0; i<300; ++i){
        t_list[i]=5.0/300.0*i;
    }

    std::vector<Param> pP;
    pP.reserve(NSubFaults);


    Fault Faille;


    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
    Eigen::Matrix<double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> storage_matrix(NSubFaults, t_list.size());
    

    std::cout<<" done \n";


    /*
    -----------------------------------------------------------------------------------------------------------------------
                        COMPUTE SURFACE DISPACEMENT 
    -----------------------------------------------------------------------------------------------------------------------
    */

    
    
    for (int i=0; i<NSubFaults ; i++){
            pP.emplace_back(0.01+0.001*i, 0.4, 0.17, 0.1, 0.08 * 100.0 / (365.0 * 24.0), 2.0);
        }
        
        
    

    surface_response(pP,  t_list, Faille , G, RES_matrix,  storage_matrix);
    
    


    /*
    -------------------------------------------------------------------------------------------------------------------
                    EXPORT DATA FOR VIZUALISATION
    -------------------------------------------------------------------------------------------------------------------
    */
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


    /*
    -------------------------------------------------------------------------------------------------------------------
                        EXPORTATION DES PARAMÈTRES DU MODÈLE (pP)
    -------------------------------------------------------------------------------------------------------------------
    */
    std::ofstream param_file("model_parameters.csv");
    if (param_file.is_open()) {
        // En-tête décrivant les colonnes pour chaque sous-faille
        // (Ajuste les noms de colonnes selon les variables membres exactes de ta classe Param)
        param_file << "SubFault_Index,k_a_sigma,b_a,D_c_inv,Dtau_asigma\n";
        
        // Écriture ligne par ligne (chaque ligne = une sous-faille)
        for (size_t i = 0; i < pP.size(); ++i) {
            param_file << i << ","
                       << pP[i].k_a_sigma << ","
                       << pP[i].b_a << ","
                       << pP[i].D_c_inv << ","
                       << pP[i].Dtau_asigma << "\n";
        }
        param_file.close();
        std::cout << "\nFichier CSV des paramètres du modèle généré avec succès.\n";
    } else {
        std::cerr << "\nErreur : Impossible de créer le fichier CSV des paramètres.\n";
    }
    

        /*
    -------------------------------------------------------------------------------------------------------------------
                        PREPARE INVERSION STORAGE
    -------------------------------------------------------------------------------------------------------------------
    */


    //PT_param ParametersPT(0, 10.0, 0.1, 3.0, 0.0, 1000.0, 0.0, 20.0, 100.0, 10, 4);

    PT_param ParametersPT(
        0.5, 3.0,    // k_a_sigma : évite le comportement proche de 0
        0.5, 1.5,    // b_a : limite la forte instabilité
        0.0, 50.0,   // D_c_inv : MAXIMUM 50 (donc Dc minimum de 2cm), au lieu de 1000 !
        0.0, 5.0,    // Dtau_asigma : un saut de contrainte modéré
        100.0, 10, 4 // T_max descendu à 100.0, nchains=10, ncold=4
    );

    std::cout<<"\n begin parallel tempering : " ; 

    std::cout<<"compute surface displacement : ";
    auto timeStart = std::chrono::high_resolution_clock::now();

    parallel_tempering_miror(10000, ParametersPT, G, RES_matrix, t_list) ;

    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = timeEnd - timeStart;
    double timeTotal = duration.count();
    std::cout<<"temps total : "<<timeTotal;

    return 0;

    std::cout<<"\nparallel tempering done ";

}
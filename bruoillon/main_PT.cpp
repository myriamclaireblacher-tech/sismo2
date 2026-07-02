#include "inv_new.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main() {

    //visualiser avec viz2.py
    //Corriger adresse de G

    //1 sous faille, 12 station randoms 

    #pragma region model
    /*
    -----------------------------------------------------------------------------------------------------------------------
                        EXTRACT GREEN MATRIX
    -----------------------------------------------------------------------------------------------------------------------
    */
    Eigen::Matrix<double, 3*Nstations, NSubFaults> G;
    //adresse de G
    std::ifstream file("../../FortranCodes_Myriam/GreensFunctionsV1/G_matrix_4x4subfault_12statsions.txt");
    if (!file.is_open()) {
        std::cerr << "Error : Open file." << std::endl;
        return 1;
    }

    for (int i = 0; i < G.rows(); ++i) {
            for (int j = 0; j < G.cols(); ++j) {
                if (!(file >> G(i, j))) {
                    std::cerr << "Error : not enough data." << std::endl;
                    return 1;
                }
            }
        }
    file.close();

    std::cout<<"done \n";

    
    /*
    -----------------------------------------------------------------------------------------------------------------------
                        Compute  surface displacement
    -----------------------------------------------------------------------------------------------------------------------
    */
    
    omp_set_num_threads(NCPU);

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


    for (int i=0; i<NSubFaults ; i++){
            pP.emplace_back(0.01, 0.4, 0.17, 0.1, 0.08 * 100.0 / (365.0 * 24.0), 2.0);
        }

    surface_response(pP,  t_list, Faille , G, RES_matrix,  storage_matrix);
    

    /*
    -------------------------------------------------------------------------------------------------------------------
                    EXPORT DATA FOR VIZUALISATION
    -------------------------------------------------------------------------------------------------------------------
    */
   

    std::ofstream csv_file("surface_responses.csv");
    if (csv_file.is_open()) {
        

        csv_file << "Time";
        for (int k = 1; k <= Nstations; ++k) {
            csv_file << ",St" << k << "_North,St" << k << "_East,St" << k << "_Depth";
        }
        csv_file << "\n";
        
        for (Eigen::Index j = 0; j < RES_matrix.cols(); ++j) {
            csv_file << t_list[j];
            for (Eigen::Index i = 0; i < RES_matrix.rows(); ++i) {
                csv_file << "," << RES_matrix(i, j);
            }
            csv_file << "\n";
        }
        
        csv_file.close();
        std::cout << "\nFichier CSV généré avec succès pour " << Nstations << " station(s).\n";
    } else {
        std::cerr << "\nErreur : Impossible de créer le fichier CSV.\n";
    }

    /*
    -------------------------------------------------------------------------------------------------------------------
                        EXPORTATION DES PARAMÈTRES DU MODÈLE (pP)
    -------------------------------------------------------------------------------------------------------------------
    */

    std::ofstream param_file2("model_parameters.csv");
    if (param_file2.is_open()) {
        param_file2 << "SubFault_Index,k_a_sigma,b_a,D_c_inv,Dtau_asigma,V0_\n";
        
        for (size_t i = 0; i < pP.size(); ++i) { 
            param_file2 << i << ","
                       << pP[i].k_a_sigma << ","
                       << pP[i].b_a << ","
                       << pP[i].D_c_inv << ","
                       << pP[i].Dtau_asigma << ","
                       << pP[i].V0_ << "\n"; 
        }
        param_file2.close();
        std::cout << "\n Model parameters exported\n";
    } else {
        std::cerr << "\nError : Cannot export model parameters.\n";
    }


        /*
    -------------------------------------------------------------------------------------------------------------------
                        PREPARE INVERSION STORAGE
    -------------------------------------------------------------------------------------------------------------------
    */
    #pragma endregion
    
    //PT_param ParametersPT(0, 10.0, 0.1, 3.0, 0.0, 1000.0, 0.0, 20.0, 100.0, 10, 4);

    PT_param ParametersPT(
        0.001, 10.0,    // k_a_sigma
        0.3, 10,    // b_a : 1/10 - 3
        0.0, 100.0,   // D_c_inv 
        0.0, 10.0,    // Dtau_asigma 
        2000.0, 80, 80/4,  // T_max descendu à 100.0, nchains=10, ncold=4
        100000 //burn-in-steps
    );

    Bounds_Param bds(
        0.0, 1, //a_sigma/k
        0.0, 1.0,    // Dtau_asigma 
        0, 1,  //V_V
        0, 1, //big param
        10000.0, 6, 1,
        100000
    );


    double timeStart = omp_get_wtime();
    //std::vector<Param> best_model = parallel_tempering_lapl2(500000, ParametersPT, G, RES_matrix, t_list, 42, false) ;
    //std::vector<Param> best_model = parallel_tempering_lapl2(500000, ParametersPT, G, RES_matrix, t_list, 42, false) ;
    //std::vector<Param> best_model = SA(5000, 20 ,t_list, RES_matrix, G, ParametersPT, 0.0099) ;
    std ::vector<Easy_Param> best_model  = inversion_easy_PT(500000, bds,  RES_matrix, t_list, G , 42) ; 

    double timeEnd = omp_get_wtime();
    double timeTotal = timeEnd - timeStart;
    std::cout<<"\ntemps total : "<<int(timeTotal)<<" s ";


    #pragma region export_best_model

    /*
    -------------------------------------------------------------------------------------------------------------------
                        EXPORT BEST MCMC MODEL
    -------------------------------------------------------------------------------------------------------------------
    */
    /*
    surface_response(best_model, t_list, Faille, G, RES_matrix, storage_matrix);

    std::ofstream pred_file("best_results.csv"); 
    if (pred_file.is_open()) {
        
        pred_file << "Time";
        for (int k = 1; k <= Nstations; ++k) {
            pred_file << ",St" << k << "_North,St" << k << "_East,St" << k << "_Depth";
        }
        pred_file << "\n";

        for (Eigen::Index j = 0; j < RES_matrix.cols(); ++j) {
            pred_file << t_list[j];
            for (Eigen::Index i = 0; i < RES_matrix.rows(); ++i) {
                pred_file << "," << RES_matrix(i, j);
            }
            pred_file << "\n";
        }
        pred_file.close();
        std::cout << "\n File with best model generated \n";
        
    } else {
        std::cerr << "Error : Cannot creat bestmodelfile\n";
    }
        */

    #pragma endregion
    
    
    
    return 0;
}
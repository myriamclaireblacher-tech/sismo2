#include "inv_new.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main() {

    //visualiser avec viz2.py
    //Corriger adresse de G
    
    #pragma region model
    /*
    -----------------------------------------------------------------------------------------------------------------------
                        EXTRACT GREEN MATRIX
    -----------------------------------------------------------------------------------------------------------------------
    */
    Eigen::Matrix<double, 3*Nstations, NSubFaults> G;
    //adresse de G
    std::ifstream file("../../FortranCodes_Myriam/GreensFunctionsV1/G_matrix_4x8subfault_12statsions.txt");
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
        t_list[i]=50/300.0*i;
    }

    std::vector<Easy_Param> pP; // Vecteur vide
    pP.reserve(NSubFaults);


    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
    Eigen::Matrix<double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> storage_matrix(NSubFaults, t_list.size());


    for (int i=0; i<NSubFaults ; i++){
            pP.emplace_back(0.4, std::exp(2/0.4)* 1,  0.08/0.4 );
        }
    

    direct(t_list, pP, G, RES_matrix, storage_matrix) ; 


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

    if (RES_matrix.hasNaN()) std::cout << "\n[ALERTE] La matrice DATA contient des NaN !";

    #pragma endregion
    
    //PT_param ParametersPT(0, 10.0, 0.1, 3.0, 0.0, 1000.0, 0.0, 20.0, 100.0, 10, 4);

    
    
    Bounds_Param bds(
        0.001, 5.0, //a_sigma/k
        1e-6f, 1000,  //super_big_param
        1e-6f, 1,      //big_param
        10000.0, 100, 100/4,
        50000*3/2*0.2/2
    );

    


    double timeStart = omp_get_wtime();
    //std::vector<Param> best_model = parallel_tempering_lapl2(500000, ParametersPT, G, RES_matrix, t_list, 42, false) ;
    //std::vector<Param> best_model = parallel_tempering_lapl2(500000, ParametersPT, G, RES_matrix, t_list, 42, false) ;
    //std::vector<Param> best_model = SA(5000, 20 ,t_list, RES_matrix, G, ParametersPT, 0.0099) ;
    std::vector<Easy_Param> best_model  = inversion_easy_PT_swap(50000*3/2, bds,  RES_matrix, t_list, G , 234) ; 


    double timeEnd = omp_get_wtime();
    double timeTotal = timeEnd - timeStart;
    std::cout<<"\ntemps total : "<<int(timeTotal)<<" s ";

    std::cout<<"\n Param_initiaux : ("<< pP[0].a_sigma_k<<" , "<< pP[0].super_big_param<< " , "<<pP[0].big_param<<" ) ";
    
    for (int j=0; j<NSubFaults; j++){
        Easy_Param little_param = best_model[j];
        std::cout<<"\n Param_best     : (" <<little_param.a_sigma_k<<" , "<< little_param.super_big_param<< " , "<<little_param.big_param<<" ) ";
    }



    #pragma region export_best_model

    /*
    -------------------------------------------------------------------------------------------------------------------
                        EXPORT BEST MCMC MODEL
    -------------------------------------------------------------------------------------------------------------------
    */
    
    direct(t_list, best_model, G, RES_matrix, storage_matrix) ; 
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
        

    #pragma endregion
    
    
    
    return 0;
}
#include "inversion.hpp"
#include <iostream>
#include <chrono>
#include <fstream>


int main() {

    /*
    -------------------------------------------------------------------------------------------
                    GENERATE & STORE DATA
    -------------------------------------------------------------------------------------------
    */
    std::vector<sunrealtype> t_list(300);
    for (int i=0; i<300; ++i){
        t_list[i]=5.0/300.0*i;
    }

    //prepare data storage and tools
    Fault Faille;
    Eigen::RowVectorXd slip_data(300);
    Eigen::Ref<Eigen::RowVectorXd> data(slip_data);

    //parameters of te Fault of the data
    Param fault_param(0.01, 0.4, 0.17, 0.1, 0.08 * 100.0 / (365.0 * 24.0), 2.0);
    Faille.ODE_solver(t_list, data, fault_param);
    

    /*
    -----------------------------------------------------------------------------------------------------------------------
                        Parallel tempering
    -----------------------------------------------------------------------------------------------------------------------
    */
    
    PT_param ParametersPT(
        0.001, 10.0,    // k_a_sigma
        0.3, 10,    // b_a : 1/10 - 3
        0.0, 1000.0,   // D_c_inv 
        0.0, 10.0,    // Dtau_asigma 
        2000.0, 6, 4,  // T_max descendu à 100.0, nchains=10, ncold=4
        10000 //burn-in-steps
    );

    auto timeStart = std::chrono::high_resolution_clock::now();

    Param PP=parallel_tempering_1faille( 1000000, 6, 1, 200000, ParametersPT, data, t_list, 20);

    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = timeEnd - timeStart;
    double timeTotal = duration.count();
    std::cout<<"\ntemps total : "<<timeTotal;

    std::cout<<PP;

    /*
    -----------------------------------------------------------------------------------------------------------------------
                        Export results
    -----------------------------------------------------------------------------------------------------------------------
    */

    Eigen::RowVectorXd slip_data2(300);
    Eigen::Ref<Eigen::RowVectorXd> data2(slip_data2);

    Faille.ODE_solver(t_list, data2, PP);

    
    std::ofstream final_csv("final_results.csv");
    if (final_csv.is_open()) {
        // En-tête : Time, Cible, Prediction
        final_csv << "Time,Target,Prediction\n";
        
        for (int i = 0; i < 300; ++i) {
            final_csv << t_list[i] << "," 
                      << data(i) << "," 
                      << data2(i) << "\n";
        }
        
        final_csv.close();
        std::cout << "\nResultats exportes dans 'final_results.csv' avec succes." << std::endl;
    } else {
        std::cerr << "\nErreur : Impossible de creer final_results.csv" << std::endl;
    }

    return 0;
    
} 


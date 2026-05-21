#include "ODE.hpp"
#include <iostream>
#include <chrono>





int main(){
    Param P_test(0.01, 0.4, 0.17, 0.1, 0.08*100/(365*24), 2.0);
    // double k_enter, double a_sigma_enter, double b_sigma_enter, double D_c_enter, double V0_enter, double Dtau_enter
    std::vector<sunrealtype> t_list(300);
    for (int i=0; i<300; ++i){
        t_list[i]=5.0/300.0*i;
    }

    std::ofstream fichier_sortie("test.csv");
    
    auto timeStart = std::chrono::high_resolution_clock::now();
            
    

    //solver
    ODE_solver(fichier_sortie, t_list, P_test);

    if (!fichier_sortie.is_open()) {
        std::cerr << "Erreur : Impossible de créer le fichier test.csv" << std::endl;
        return 1;
        fichier_sortie.close();
    }
    

    auto timeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = timeEnd - timeStart;
    double timeTotal = duration.count();
    std::cout<<"\nTIME  : "<<timeTotal<<"\n";
    
        
    // 5. Fermeture du fichier
    
    
    std::cout << "Simulation terminee avec succes !" << std::endl;
    return 0;
}
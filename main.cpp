#include "inversion.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

double pi(const Param & P, const std::vector<sunrealtype> & t_list, const Eigen::Ref<Eigen::RowVectorXd> & data, Fault & F, Eigen::Ref<Eigen::RowVectorXd> & slip_list){
    int ret = F.ODE_solver(t_list, slip_list, P);
    if (ret<0) return -10e6;
    return -(data - slip_list).array().square().sum();}


Param parallel_tempering_1faille(int n_steps, int n_markow, int n_cold, const PT_param PT, const Eigen::Ref<Eigen::RowVectorXd> & data, const std::vector<sunrealtype> & t_list, double sigma= 0.005){

    int accept_count = 0;
    int crash_count = 0;
    //create  Temperture ladder
    int seed=42 ;
    std::mt19937 gen(seed); 
    std::vector<double> T(n_markow); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0); 
    for (int i=0;i<n_markow;i++){
        if (i<n_cold) T[i]=1.0 ;
        else T[i]= std :: exp( (i-n_cold+1)*1.0/(n_markow-n_cold) * std::log(PT.T_max) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    
    //random interger
    std::uniform_int_distribution<int> rand_chain(0, n_markow - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(n_markow);

    //model storage
    std::vector<Param> P;
    P.resize(n_markow);


    //open storage file
    std::vector<std::ofstream> files;
    files.resize(n_cold) ;
    for (int j=0;j<n_cold; j++){
        files[j].open("test_chain_cold_" + std::to_string(j) + ".bin", std::ios::binary);
    }

    std::cout << "Taille exacte ecrite par etape : " << sizeof(Param) ;

    #pragma omp parallel 
    {
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);
        Fault Faille ;
        Eigen::RowVectorXd slip_data(data.size());
        Eigen::Ref<Eigen::RowVectorXd> slip_list(slip_data);


        //initial models
        #pragma omp single
        {
        for (int j=0;j<n_markow;j++){

            double k_a_sigma= (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0; 
            double b_a = (PT.b_a_inf + PT.b_a_sup) / 2.0;
            double D_c_inv = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
            double Dtau_asigma= (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;
            double V0_= (PT.V0_inf + PT.V0_sup) / 2.0;
            P[j] = Param(k_a_sigma, b_a, D_c_inv, Dtau_asigma,V0_);
            llk[j] = pi(P[j], t_list, data, Faille, slip_list) ;} 

    
            //save initial model
        for (int j=0; j<n_cold;j++){
                files[j].write(reinterpret_cast<const char*>(&llk[j]), sizeof(double));
                files[j].write(reinterpret_cast<const char*>(&P[j]), sizeof(Param));}}
                    


        //begin parallel tempering

        for (int it = 0; it < n_steps ; it ++ ){

            #pragma omp single
            {
            if (it % (n_steps / 10) == 0) {
                std::cout << std::setw(3) << (100 * it / n_steps) << "% done" << std::endl;
            }}  

            #pragma omp single
            {
                if (it > 0 && it % (n_steps / 10) == 0) {
                    std::cout << std::setw(3) << (100 * it / n_steps) << "% done | "
                              << "Chain 0 -> Accept: " << accept_count 
                              << " | Crash EDO: " << crash_count 
                              << " | Last llk: " << llk[0] << std::endl;
                    
                    // Remise à zéro pour la prochaine tranche de 10%
                    accept_count = 0;
                    crash_count = 0;
                }
            }
            //update Markow chains
            #pragma omp for schedule(dynamic)
            for (int ichain = 0; ichain < n_markow ; ichain++){
                //new model proposal


                double prop;
                double k_a_sigma;
                prop = unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
                if (prop + P[ichain].k_a_sigma > PT.k_a_sigma_sup)   k_a_sigma = 2 * PT.k_a_sigma_sup - P[ichain].k_a_sigma - prop ; 
                else if (prop + P[ichain].k_a_sigma < PT.k_a_sigma_inf)  k_a_sigma = 2 * PT.k_a_sigma_inf - P[ichain].k_a_sigma - prop ;
                else  k_a_sigma   =  P[ichain].k_a_sigma +  prop;
                double b_a;
                prop = unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigma;
                if (prop + P[ichain].b_a > PT.b_a_sup)   b_a = 2 * PT.b_a_sup - P[ichain].b_a - prop ; 
                else if (prop + P[ichain].b_a < PT.b_a_inf)  b_a = 2 * PT.b_a_inf - P[ichain].b_a - prop ;
                else b_a  = P[ichain].b_a + prop;
                double D_c_inv;
                
                prop = unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma;
                if (prop + P[ichain].D_c_inv > PT.D_c_inv_sup)   D_c_inv = 2 * PT.D_c_inv_sup - P[ichain].D_c_inv - prop ; 
                else if (prop + P[ichain].D_c_inv < PT.D_c_inv_inf)  D_c_inv = 2 * PT.D_c_inv_inf - P[ichain].D_c_inv - prop ;
                else  D_c_inv   = P[ichain].D_c_inv  + prop;

                double Dtau_asigma;
                prop = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
                if (prop + P[ichain].Dtau_asigma > PT.Dtau_asigma_sup)   Dtau_asigma = 2 * PT.Dtau_asigma_sup - P[ichain].Dtau_asigma - prop ; 
                else if (prop + P[ichain].Dtau_asigma < PT.Dtau_asigma_inf)  Dtau_asigma = 2 * PT.Dtau_asigma_inf - P[ichain].Dtau_asigma - prop ;
                else Dtau_asigma   = P[ichain].Dtau_asigma  + prop;

                double V0_;
                prop = unif_dist_plus(gen2) * (PT.V0_sup - PT.V0_inf) * sigma;
                if (prop + P[ichain].V0_ > PT.V0_sup)   V0_ = 2 * PT.V0_sup - P[ichain].V0_ - prop ; 
                else if (prop + P[ichain].V0_ < PT.V0_inf)  V0_ = 2 * PT.V0_inf - P[ichain].V0_ - prop ;
                else V0_   = P[ichain].V0_  + prop;

                Param P_new(k_a_sigma, b_a, D_c_inv, Dtau_asigma, V0_);

                double Enew = pi(P_new, t_list, data, Faille, slip_list);

                bool accept = false ;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen2) );
                accept = (u <= alpha);



                if (ichain == 0) {
                    if (Enew <= -9e6) {
                        #pragma omp atomic
                        crash_count++;
                    }
                    if (accept) {
                        #pragma omp atomic
                        accept_count++;
                    }
                }






                if (accept){
                    llk[ichain] = Enew;
                    P[ichain] = P_new ;
                    
                }
                if (ichain< n_cold) {
                        double val_pi = llk[ichain];
                        
                        #pragma omp critical(file_write)
                        {
                            files[ichain].write(reinterpret_cast<const char*>(&val_pi), sizeof(double));
                            files[ichain].write(reinterpret_cast<const char*>(&P[ichain]), sizeof(Param));
                        }
                }

            }

            #pragma omp single
                { 
                for (int s = 0; s < n_markow - 1; s++) {
                    int p = rand_chain(gen);
                    int q = rand_chain(gen);
                    if ((p == q) ||  (T[p] == T[q])) continue;

                    // Formule théorique du Parallel Tempering
                    double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
                    double u_swap     = std::log(unif_dist(gen));

                    if (u_swap <= alpha_swap) {
                        std::swap(P[p], P[q]);
                        std::swap(llk[p], llk[q]);
                    }
                }}


        }
        
    }
    
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nlast llk computed"<< llk[1];
    //close files
    for (int j=0; j<n_cold; j++){
        if(files[j].is_open()) files[j].close();
    }

    return P[0];
    }



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
        0.3, 10,    // b_a :
        0.0, 1000.0,   // D_c_inv 
        0.0, 10.0,    // Dtau_asigma 
        100000.0, 6, 1 // T_max descendu à 100.0, nchains=10, ncold=4
    );

    Param PP=parallel_tempering_1faille( 100000, 6, 1, ParametersPT, data, t_list, 0.01);

    

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


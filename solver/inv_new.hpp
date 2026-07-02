#ifndef Easy_PARALLEL_TEMPERING
#define EasyPARALLEL_TEMPERING

#include "inversion.hpp"


struct Easy_Param
{
    double a_sigma_k ;
    double dtau_asigma ;
    double V_V;
    double big_param ;
    Easy_Param(double a_sigma_k_ent, double dtau_asigma_ent, double V_V_ent, double big_pram_ent ) : 
    a_sigma_k(a_sigma_k_ent), dtau_asigma (dtau_asigma_ent), V_V ( V_V_ent), big_param (big_pram_ent){} ;
    Easy_Param() = default ;
} ;

struct Bounds_Param{
    double a_sigma_k_inf; double a_sigma_k_sup;
    double dtau_asigma_inf ; double dtau_asigma_sup;
    double V_V_inf ; double V_V_sup ;
    double big_param_inf; double big_param_sup;
    double T ; int nchains; int ncold;
    int burn_in ;
    Bounds_Param(double a_ent, double a_ent_sup, double d_ent, double d_ent_sup, double V_V_ent, double V_V_ent_sup, double big_ent, double big_ent_sup, double T_ent, int nchains_ent, int ncold_ent, int burn_in_ent) :
    a_sigma_k_inf(a_ent), a_sigma_k_sup(a_ent_sup), dtau_asigma_inf(d_ent), dtau_asigma_sup(d_ent_sup), V_V_inf(V_V_ent), V_V_sup(V_V_ent_sup), 
    big_param_inf(big_ent), big_param_sup(big_ent_sup), T(T_ent), nchains(nchains_ent), ncold (ncold_ent), burn_in(burn_in_ent) {} ; 
} ;

void direct(std::vector<sunrealtype> t_list, std::vector<Easy_Param> PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    for (int j=0; j<NSubFaults; j++){
        for (int i=0; i<t_list.size(); i++)
            storage_matrix(j, i) = PT[j].a_sigma_k * std::log(1.0 + std::exp(PT[j].dtau_asigma) * PT[j].V_V * (std::exp(PT[j].big_param * t_list[i]) - 1.0));
    }
    RES_matrix.noalias() = G * storage_matrix ;
}

double llk_easy (Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, std::vector<sunrealtype> t_list, std::vector<Easy_Param> PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

        direct(t_list, PT, G, RES_matrix, storage_matrix) ;
        double llk=  - (data - RES_matrix).array().abs().sum();
        if (std::isnan(llk)) return -1e9f;
        return llk ;
    } 





std::vector<Easy_Param> inversion_easy_PT(int maxint, Bounds_Param bds, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, int seed ){

    #pragma region initializing

    std::vector<double> sigmas (bds.nchains, 0.01) ; 

    int adapt_window=200;

    std::vector<Easy_Param> P(bds.nchains*NSubFaults) ;

    double best_llk = -1e9f ;
    std::vector<double> llk (bds.nchains);
    std::vector<Easy_Param> best_model (NSubFaults) ;

    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::vector<int> sfindex(bds.nchains, 0) ;

    std::vector<double> accepts_chain(bds.nchains) ;

    std::vector<double> T (bds.nchains) ;
    for (int i=0;i<bds.nchains;i++){
        if (i<bds.ncold) T[i]=1.0 ;
        //loi de puissance
        else T[i]= std :: exp( std::pow((i-bds.ncold+1)*1.0/(bds.nchains-bds.ncold),2.0) * std::log(bds.T) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    
    std::uniform_int_distribution<int> rand_chain(0, bds.nchains - 1); 
    #pragma endregion
    
    #pragma omp parallel
    {
        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
        Eigen::Matrix<double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> storage_matrix(NSubFaults, t_list.size());

        std::mt19937 gen(seed+omp_get_thread_num());
        std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);

        #pragma omp single
        {
        for (int j=0; j<bds.nchains * NSubFaults ; j++){
                double p1 = unif_dist_intern (gen) * (bds.a_sigma_k_sup -bds.a_sigma_k_inf) + bds.a_sigma_k_inf;
                double p2 = unif_dist_intern(gen) *(bds.dtau_asigma_sup- bds.dtau_asigma_inf) + bds.dtau_asigma_inf ;
                double p3 = unif_dist_intern(gen) * (bds.V_V_sup-bds.V_V_inf) +bds.V_V_inf ;
                double p4 = unif_dist_intern(gen) * (bds.big_param_sup - bds.big_param_inf ) + bds.big_param_inf ;
                P[j] = Easy_Param(p1, p2, p3, p4); }
        
        for (int j=0; j<bds.nchains; j++){
            std::vector<Easy_Param> P_new (NSubFaults);
            std::copy(P.begin() + (j * NSubFaults), P.begin() + (j+1) * NSubFaults, P_new.begin()) ;
            llk[j] = llk_easy (data, t_list, P_new, G, RES_matrix, storage_matrix) ;
            if (llk[j]>best_llk) best_llk=llk[j];
            
        }
        std::cout<<"\n best llk : "<<best_llk ;

        
        }


        for (int it=0; it<maxint ; it++ ){
            /*
            #pragma region affichage

            if (it > 0 && (it % 25000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : " << sigmas[c] 
                                  << " |  T : " << T[c] ;}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                    // Remise à zéro pour la prochaine tranche de 10%
                    
                }
            

            #pragma endregion

            */
            #pragma region adaptation 

            if (it > 0 && it % adapt_window == 0) {
                     
                    
                    // On adapte uniquement si on est dans la période de Burn-in
                    if (it < bds.burn_in) {
                        for (int c = 0; c < bds.nchains; c++) {
                            //sigma
    
                            double acc_rate = (double)accepts_chain[c] / adapt_window;
                            
                            if (it < bds.burn_in/2){
                                if ((acc_rate < 0.20 ) && (sigmas[c]>1e-7)) {
                                sigmas[c] *= 0.9 ;
                                } else if ((acc_rate > 0.30) &&(sigmas[c]<0.1) )  sigmas[c] *= 1.1; 
                                
                            }
                            else
                                {if ((acc_rate < 0.20 )&&(sigmas[c]>1e-7)) {
                                sigmas[c] *= 0.9 ;
                                } else if ((acc_rate > 0.30) &&(sigmas[c]<0.1) )  sigmas[c] *= 1.1; }
                            ;}
                        
                    for (int c = 0; c < bds.nchains; c++) {
                        accepts_chain[c] = 0; 
                    }
                }
            #pragma endregion

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i =sfindex[ichain];}

                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P_new.begin(), P_new.end(), P.begin() + (ichain * NSubFaults));

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * sigmas[ichain];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                
                prop = unif_dist_plus(gen) * (bds.dtau_asigma_sup   - bds.dtau_asigma_inf) * sigmas[ichain];
                if (prop + P_new[i].dtau_asigma > bds.dtau_asigma_sup)   P_new[i].dtau_asigma = 2 * bds.dtau_asigma_sup - P_new[i].dtau_asigma - prop ; 
                else if (prop + P_new[i].dtau_asigma < bds.dtau_asigma_inf)  P_new[i].dtau_asigma = 2 * bds.dtau_asigma_inf - P_new[i].dtau_asigma - prop ;
                else  P_new[i].dtau_asigma   =  P_new[i].dtau_asigma +  prop;

                
                prop = unif_dist_plus(gen) * (bds.V_V_sup   - bds.V_V_inf) * sigmas[ichain];
                if (prop + P_new[i].V_V > bds.V_V_sup)   P_new[i].V_V = 2 * bds.V_V_sup - P_new[i].V_V - prop ; 
                else if (prop + P_new[i].V_V < bds.V_V_inf)  P_new[i].V_V = 2 * bds.V_V_inf - P_new[i].V_V - prop ;
                else  P_new[i].V_V   =  P_new[i].V_V +  prop;

                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * sigmas[ichain];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                double Enew = llk_easy (data, t_list, P_new, G, RES_matrix, storage_matrix);
                
                #pragma endregion

                #pragma region acept

                bool accept = false ;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen) );
                accept = (u <= alpha);

                if (accept){
                    std::copy(P_new.begin(), P_new.end(), P.begin() + (ichain * NSubFaults));
                    llk[ichain] = Enew;
                    accepts_chain[ichain]++;
                    if (Enew>best_llk) {
                        #pragma omp critical (updat_best_model)
                        {
                        best_llk=Enew ;
                        best_model = P_new ; }
                    }
                }

                #pragma endregion
            }

            #pragma region swap
            #pragma omp single
            {
                for (int s = 0; s < bds.nchains - 1; s++) {
                    int p = rand_chain(gen);
                    int q = rand_chain(gen);
                    if ((p == q) ||  (T[p] == T[q])) continue;

                    // Formule théorique du Parallel Tempering
                    double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
                    double u_swap     = std::log(unif_dist_intern(gen));
                    //swap_try[p]++;
                    //swap_try[q]++;

                    if (u_swap <= alpha_swap) {
                        //swap_rate[p]++;
                        //swap_rate[q]++;
                        std::swap_ranges(P.begin() + (p * NSubFaults), P.begin() + ((p + 1) * NSubFaults), P.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                    }
            }}
            #pragma endregion

        }





    }
    #pragma endregion

    }
    return best_model ; 
}




#endif
#ifndef Easy_PARALLEL_TEMPERING
#define EasyPARALLEL_TEMPERING

#include "inversion.hpp"


struct Easy_Param
{
    double a_sigma_k ;
    double super_big_param ;
    double big_param ;
    Easy_Param(double a_sigma_k_ent, double super_big_param_ent, double big_pram_ent ) : 
    a_sigma_k(a_sigma_k_ent), super_big_param (super_big_param_ent), big_param (big_pram_ent){} ;
    Easy_Param() = default ;
} ;


struct Bounds_Param{
    double a_sigma_k_inf; double a_sigma_k_sup;
    double super_big_param_inf ; double super_big_param_sup;
    double big_param_inf ; double big_param_sup ; 
    double T ; int nchains; int ncold;
    int burn_in ;
    Bounds_Param(double a_ent, double a_ent_sup, double super_big_param_ent, double super_big_param_ent_sup, double big_ent, double big_ent_sup, double T_ent, int nchains_ent, int ncold_ent, int burn_in_ent) :
    a_sigma_k_inf(a_ent), a_sigma_k_sup(a_ent_sup), super_big_param_inf(super_big_param_ent), super_big_param_sup(super_big_param_ent_sup),
    big_param_inf(big_ent), big_param_sup(big_ent_sup), T(T_ent), nchains(nchains_ent), ncold (ncold_ent), burn_in(burn_in_ent) {} ; 
} ;

void directi( int index, std::vector<sunrealtype> t_list, std::vector<Easy_Param> PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    
        for (int i=0; i<t_list.size(); i++)
            storage_matrix(index, i) = PT[index].a_sigma_k * std::log(1.0 + PT[index].super_big_param * (std::exp(PT[index].big_param * t_list[i]) - 1.0));
    
    RES_matrix.noalias() = G * storage_matrix ;
    }

void direct(std::vector<sunrealtype> t_list, std::vector<Easy_Param> PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

    for (int j=0; j<NSubFaults; j++){
        for (int i=0; i<t_list.size(); i++)
            storage_matrix(j, i) = PT[j].a_sigma_k * std::log(1.0 +  PT[j].super_big_param * (std::exp(PT[j].big_param * t_list[i]) - 1.0));
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

double llk_easy_i (int index, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, std::vector<sunrealtype> t_list, std::vector<Easy_Param> PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G,
    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& RES_matrix,
    Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor> & storage_matrix){

        directi(index, t_list, PT, G, RES_matrix, storage_matrix) ;
        double llk=  - (data - RES_matrix).array().abs().sum();
        if (std::isnan(llk)) return -1e9f;
        return llk ;
    } 

/*
std::vector<Easy_Param> inversion_easy_PT(int maxint, Bounds_Param bds, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, int seed ){

    #pragma region initializing

    //-------------------------------------------------------------------------------------------------
    int swap_rate=0;
    std::vector<int> neighbour_swap (bds.nchains, 0);
    std::vector<int> subfault_index(NSubFaults);
    std::vector<double> accepts_chain(bds.nchains) ;
    //--------------------------------------------------------------------------------------------------

    std::vector<Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor>> storage_matrixes (bds.nchains);
    for(int c=0; c<bds.nchains; c++){
                storage_matrixes[c].resize(NSubFaults, t_list.size());
    }

    int adapt_window=200;
    std::vector<double> sigmas (bds.nchains, 0.01) ; 

    std::vector<Easy_Param> P(bds.nchains*NSubFaults) ;

    double best_llk = -1e9f ;
    std::vector<double> llk (bds.nchains);
    std::vector<Easy_Param> best_model (NSubFaults) ;

    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::vector<int> sfindex(bds.nchains, 0) ;
    

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
        

        std::mt19937 gen(seed+omp_get_thread_num());
        std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);

        #pragma omp single
        {

        for (int j=0; j<bds.nchains * NSubFaults ; j++){
                double p1 = unif_dist_intern (gen) * (bds.a_sigma_k_sup -bds.a_sigma_k_inf) + bds.a_sigma_k_inf;
                double p2 = unif_dist_intern(gen) *(bds.super_big_param_sup- bds.super_big_param_inf) + bds.super_big_param_inf ;
                double p4 = unif_dist_intern(gen) * (bds.big_param_sup - bds.big_param_inf ) + bds.big_param_inf ;
                P[j] = Easy_Param(p1, p2, p4); }
        
        for (int j=0; j<bds.nchains; j++){
            std::vector<Easy_Param> P_new (NSubFaults);
            std::copy(P.begin() + (j * NSubFaults), P.begin() + (j+1) * NSubFaults, P_new.begin()) ;
            llk[j] = llk_easy (data, t_list, P_new, G, RES_matrix, storage_matrixes[j]) ;
            if (llk[j]>best_llk) best_llk=llk[j];
            
        }

        std::cout<<"\n best  initial llk : "<<best_llk ;

        
        }

        
        for (int it=0; it<maxint ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : " << sigmas[c] 
                                  << " |  T : " << T[c] ;}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            

            #pragma endregion

            
            #pragma region adaptation 

            if (it > 0 && it % adapt_window == 0) {
                     
                    std::vector<double> acc_rate (bds.nchains);
                    for (int c = 0; c < bds.nchains; c++) {
                        acc_rate[c] = (double)accepts_chain[c] / adapt_window;
                        accepts_chain[c] = 0; 
                    }
                    // On adapte uniquement si on est dans la période de Burn-in
                    if (it < bds.burn_in) {
                        for (int c = 0; c < bds.nchains; c++) {
                            //sigma                        
                            
                            if (it < bds.burn_in/2){
                                if ((acc_rate[c] < 0.20 ) && (sigmas[c]>1e-15)) {
                                sigmas[c] *= 0.9 ;
                                } else if ((acc_rate[c] > 0.30) &&(sigmas[c]<0.1) )  sigmas[c] *= 1.1; 
                                
                            }
                            else
                                {if ((acc_rate[c] < 0.20 )&&(sigmas[c]>1e-15)) {
                                sigmas[c] *= 0.9 ;
                                } else if ((acc_rate[c] > 0.30) &&(sigmas[c]<0.1) )  sigmas[c] *= 1.1; }
                            ;}
                        
                    
                }
            #pragma endregion

            }
            }

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * sigmas[ichain];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                
                prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * sigmas[ichain];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;


                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * sigmas[ichain];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;

                #pragma endregion
            }

            #pragma region swap
            #pragma omp single
            {
                for (int hey=0; hey<2; hey++){
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
                        swap_rate++;
                        std::swap_ranges(P.begin() + (p * NSubFaults), P.begin() + ((p + 1) * NSubFaults), P.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                        std::swap(storage_matrixes[p], storage_matrixes[q]); /////////////////////////ici
                    }
            }}}

            
            
            #pragma endregion

        }

    }
    #pragma endregion

    

    std::cout<<"\nFinal best llk : "<<best_llk;
    std::cout<<"\n average number of swap per chain: "<<swap_rate*1.0/bds.nchains;
    std::cout<<"\n swap every "<<maxint/(swap_rate*1.0/bds.nchains)<<"   iterations";
    
    return best_model ; 
}
*/

struct ChainSaver {
    std::ofstream file;
    ChainSaver(int cold_idx)
        {
            file.open("Easy_chain_cold_" + std::to_string(cold_idx) + ".bin", std::ios::binary);
        } ;
    void save_step(const std::vector<Easy_Param>& pP, double energy)
        {file.write(reinterpret_cast<const char*>(&energy), sizeof(double));
        file.write(reinterpret_cast<const char*>(pP.data()), pP.size() * sizeof(Easy_Param));} ;
};



std::vector<Easy_Param> inversion_easy_PT_sigma(int maxint, Bounds_Param bds, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, int seed ){

    #pragma region initializing

    //-------------------------------------------------------------------------------------------------
    int swap_rate=0;
    std::vector<int> neighbour_swap (bds.nchains, 0);
    std::vector<int> subfault_index(NSubFaults);
    std::vector<double> accepts_chain(bds.nchains) ;
    //--------------------------------------------------------------------------------------------------

    std::vector<ChainSaver> savers;
    for (int i = 0; i < bds.ncold; ++i) {
        savers.emplace_back(i);
    }

    int nparam=3;
    int param_num=0;

    std::vector<Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor>> storage_matrixes (bds.nchains);
    for(int c=0; c<bds.nchains; c++){
                storage_matrixes[c].resize(NSubFaults, t_list.size());
    }

    int adapt_window=200;
    std::vector<std::vector<double>> SIGMAS (bds.nchains) ; 
    for(int c=0; c<bds.nchains; c++){
                SIGMAS[c]={0.01 , 0.01 , 0.01, 1 };
    }

    std::vector<Easy_Param> P(bds.nchains*NSubFaults) ;

    double best_llk = -1e9f ;
    std::vector<double> llk (bds.nchains);
    std::vector<Easy_Param> best_model (NSubFaults) ;

    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::vector<int> sfindex(bds.nchains, 0) ;
    

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
        #pragma region model_i
        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
        

        std::mt19937 gen(seed+omp_get_thread_num());
        std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);
        std::uniform_int_distribution<int> random_param(0, nparam-1);

        #pragma omp single
        {

        for (int j=0; j<bds.nchains * NSubFaults ; j++){
                double p1 = unif_dist_intern (gen) * (bds.a_sigma_k_sup -bds.a_sigma_k_inf) + bds.a_sigma_k_inf;
                double p2 = unif_dist_intern(gen) *(bds.super_big_param_sup- bds.super_big_param_inf) + bds.super_big_param_inf ;
                double p4 = unif_dist_intern(gen) * (bds.big_param_sup - bds.big_param_inf ) + bds.big_param_inf ;
                P[j] = Easy_Param(p1, p2, p4); }
        
        for (int j=0; j<bds.nchains; j++){
            std::vector<Easy_Param> P_new (NSubFaults);
            std::copy(P.begin() + (j * NSubFaults), P.begin() + (j+1) * NSubFaults, P_new.begin()) ;
            llk[j] = llk_easy (data, t_list, P_new, G, RES_matrix, storage_matrixes[j]) ;
            if (llk[j]>best_llk) best_llk=llk[j];
            if (j<bds.ncold) savers[j].save_step(P_new, llk[j]);
            
        }

        std::cout<<"\n best  initial llk : "<<best_llk ;

        

        
        }
        #pragma endregion

        for (int it=0; it<bds.burn_in*NSubFaults*2/3 ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : (" << SIGMAS[c][0]<<" , "<<SIGMAS[c][1]<<" , "<< SIGMAS[c][2]<< " ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c];}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            

            #pragma endregion

            
            #pragma region adaptation 
            
            if (it > 0 && it % adapt_window == 0) {

                     
                std::vector<double> acc_rate (bds.nchains);
                for (int c = 0; c < bds.nchains; c++) {
                    acc_rate[c] = (double)accepts_chain[c] / adapt_window;
                    accepts_chain[c] = 0; 
                }
                // On adapte uniquement si on est dans la période de Burn-in
                
                for (int c = 0; c < bds.nchains; c++) {
                    //sigma                        
                    if ((acc_rate[c] < 0.20 ) && (SIGMAS[c][param_num]>1e-7)) {
                    SIGMAS[c][param_num] *= 0.9 ;
                    } else if ((acc_rate[c] > 0.30) &&(SIGMAS[c][param_num]<0.1) )  SIGMAS[c][param_num] *= 1.1; 
                }
                param_num= random_param(gen) ;
            }
            #pragma endregion

            
            }

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                //if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                if (param_num==0){
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;}

                else if (param_num==1){
                double prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;}

                else {
                double prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;}

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;

                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

                #pragma endregion
            }

            #pragma region swap
            #pragma omp single
            {
                for (int hey=0; hey<2; hey++){
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
                        swap_rate++;
                        std::swap_ranges(P.begin() + (p * NSubFaults), P.begin() + ((p + 1) * NSubFaults), P.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                        std::swap(storage_matrixes[p], storage_matrixes[q]); /////////////////////////ici
                    }
            }}}

            
            
            #pragma endregion

        }


        #pragma omp single
        {std::cout<<"\n END OF PHASE 1 ";}
        for (int it=bds.burn_in*NSubFaults*2/3; it<bds.burn_in*NSubFaults ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma adjust : (" << SIGMAS[c][nparam]<<" ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c];}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            

            #pragma endregion

            
            #pragma region adaptation 

            

            if (it > 0 && it % adapt_window == 0) {

                     
                std::vector<double> acc_rate (bds.nchains);
                for (int c = 0; c < bds.nchains; c++) {
                    acc_rate[c] = (double)accepts_chain[c] / adapt_window;
                    accepts_chain[c] = 0; 
                }
                // On adapte uniquement si on est dans la période de Burn-in
                
                for (int c = 0; c < bds.nchains; c++) {
                    //sigma                        
                    if (acc_rate[c] < 0.20 )  {
                    SIGMAS[c][nparam] *= 0.9 ;
                    } else if (acc_rate[c] > 0.30)  SIGMAS[c][nparam] *= 1.1; 
                }
            #pragma endregion

            }
            }

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                //if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;

                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;
                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

                #pragma endregion
            }

            #pragma region swap
            #pragma omp single
            {
                for (int hey=0; hey<2; hey++){
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
                        swap_rate++;
                        std::swap_ranges(P.begin() + (p * NSubFaults), P.begin() + ((p + 1) * NSubFaults), P.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                        std::swap(storage_matrixes[p], storage_matrixes[q]); /////////////////////////ici
                    }
            }}}

            
            
            #pragma endregion

        }


        #pragma region adapt
        #pragma omp single
        {std::cout<<"\n END OF PHASE 2 ";
        double mean1=0;
        double mean2=0;
        double mean3=0;
        double mean4=0;
        for (int i=0;i<bds.ncold; i++){
            mean1+= SIGMAS[i][0];
            mean2+= SIGMAS[i][1];
            mean3+= SIGMAS[i][2];
            mean4+= SIGMAS[i][3];
        }
        mean1/=bds.ncold;
        mean2/=bds.ncold;
        mean3/=bds.ncold;
        mean4/=bds.ncold;
        for (int i=0; i<bds.nchains; i++){
            if (i<bds.ncold){SIGMAS[i][0]= mean1; SIGMAS[i][1]=mean2; SIGMAS[i][2] = mean3; SIGMAS[i][4] = mean4;}
            else{SIGMAS[i][0]= mean1*std::sqrt(T[i]);SIGMAS[i][1]= mean2*std::sqrt(T[i]);SIGMAS[i][2]= mean3*std::sqrt(T[i]);SIGMAS[i][3]= mean4*std::sqrt(T[i]); }
        }
        #pragma endregion

        
        }
        for (int it=bds.burn_in*NSubFaults; it<maxint*NSubFaults ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : (" << SIGMAS[c][0]<<" , "<<SIGMAS[c][1]<<" , "<< SIGMAS[c][2]<<" , "<< SIGMAS[c][3] << " ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c] ;}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            if (it%adapt_window==0) {for (int c=0; c<bds.nchains; c++) accepts_chain[c]=0 ;}
            

            #pragma endregion


            }
            

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                
                prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;


                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;

                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

                #pragma endregion
            }

            #pragma region swap
            #pragma omp single
            {
                for (int hey=0; hey<2; hey++){
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
                        swap_rate++;
                        std::swap_ranges(P.begin() + (p * NSubFaults), P.begin() + ((p + 1) * NSubFaults), P.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                        std::swap(storage_matrixes[p], storage_matrixes[q]); /////////////////////////ici
                    }
            }}}

            
            
            #pragma endregion

        }

    }
    #pragma endregion

    

    std::cout<<"\nFinal best llk : "<<best_llk;
    std::cout<<"\n swap every "<<maxint*NSubFaults/(swap_rate*1.0/bds.nchains)<<"   iterations";
    
    return best_model ; 
}




std::vector<Easy_Param> inversion_easy_PT_swap(int maxint, Bounds_Param bds, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, int seed ){

    #pragma region initializing

    //-------------------------------------------------------------------------------------------------
    int swap_rate=0;
    std::vector<int> neighbour_swap (bds.nchains, 0);
    std::vector<int> subfault_index(NSubFaults);
    std::vector<double> accepts_chain(bds.nchains) ;
    //--------------------------------------------------------------------------------------------------

    std::vector<ChainSaver> savers;
    for (int i = 0; i < bds.ncold; ++i) {
        savers.emplace_back(i);
    }

    int nparam=3;
    int param_num=0;

    std::vector<Eigen::Matrix <double, NSubFaults, Eigen::Dynamic, Eigen::RowMajor>> storage_matrixes (bds.nchains);
    for(int c=0; c<bds.nchains; c++){
                storage_matrixes[c].resize(NSubFaults, t_list.size());
    }

    int adapt_window=200;
    std::vector<std::vector<double>> SIGMAS (bds.nchains) ; 
    for(int c=0; c<bds.nchains; c++){
                SIGMAS[c]={0.01 , 0.01 , 0.01, 1 };
    }

    std::vector<Easy_Param> P(bds.nchains*NSubFaults) ;

    double best_llk = -1e9f ;
    std::vector<double> llk (bds.nchains);
    std::vector<Easy_Param> best_model (NSubFaults) ;

    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::vector<int> sfindex(bds.nchains, 0) ;
    

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
        #pragma region model_i
        Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic> RES_matrix(3*Nstations, t_list.size());
        

        std::mt19937 gen(seed+omp_get_thread_num());
        std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);
        std::uniform_int_distribution<int> random_param(0, nparam-1);

        #pragma omp single
        {

        for (int j=0; j<bds.nchains * NSubFaults ; j++){
                double p1 = unif_dist_intern (gen) * (bds.a_sigma_k_sup -bds.a_sigma_k_inf) + bds.a_sigma_k_inf;
                double p2 = unif_dist_intern(gen) *(bds.super_big_param_sup- bds.super_big_param_inf) + bds.super_big_param_inf ;
                double p4 = unif_dist_intern(gen) * (bds.big_param_sup - bds.big_param_inf ) + bds.big_param_inf ;
                P[j] = Easy_Param(p1, p2, p4); }
        
        for (int j=0; j<bds.nchains; j++){
            std::vector<Easy_Param> P_new (NSubFaults);
            std::copy(P.begin() + (j * NSubFaults), P.begin() + (j+1) * NSubFaults, P_new.begin()) ;
            llk[j] = llk_easy (data, t_list, P_new, G, RES_matrix, storage_matrixes[j]) ;
            if (llk[j]>best_llk) best_llk=llk[j];
            if (j<bds.ncold) savers[j].save_step(P_new, llk[j]);
            
        }

        std::cout<<"\n best  initial llk : "<<best_llk ;

        

        
        }
        #pragma endregion

        for (int it=0; it<bds.burn_in*NSubFaults*2/3 ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : (" << SIGMAS[c][0]<<" , "<<SIGMAS[c][1]<<" , "<< SIGMAS[c][2]<< " ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c];}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            

            #pragma endregion

            
            #pragma region adaptation 
            
            if (it > 0 && it % adapt_window == 0) {

                     
                std::vector<double> acc_rate (bds.nchains);
                for (int c = 0; c < bds.nchains; c++) {
                    acc_rate[c] = (double)accepts_chain[c] / adapt_window;
                    accepts_chain[c] = 0; 
                }
                // On adapte uniquement si on est dans la période de Burn-in
                
                for (int c = 0; c < bds.nchains; c++) {
                    //sigma                        
                    if ((acc_rate[c] < 0.20 ) && (SIGMAS[c][param_num]>1e-7)) {
                    SIGMAS[c][param_num] *= 0.9 ;
                    } else if ((acc_rate[c] > 0.30) &&(SIGMAS[c][param_num]<0.1) )  SIGMAS[c][param_num] *= 1.1; 
                }
                param_num= random_param(gen) ;
            }
            #pragma endregion

            
            }

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                //if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                if (param_num==0){
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;}

                else if (param_num==1){
                double prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;}

                else {
                double prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;}

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;

                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

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
                    int pP = random_index(gen);
                    std::vector<Easy_Param> P_new (NSubFaults);
                    //copy p model in P_new
                    std::copy(P.begin() + (p * NSubFaults), P.begin() + (p+1) * NSubFaults, P_new.begin());
                    //exchange q parameters subbfault with p
                    std::copy(P.begin() + (q * NSubFaults) + pP, P.begin() + q * NSubFaults +pP + 1 , P_new.begin() + pP);

                    Eigen::VectorXd old_line = storage_matrixes[p].row(pP);
                    double llk1 = llk_easy_i (pP, data, t_list, P_new, G, RES_matrix, storage_matrixes[p]);


                    std::vector<Easy_Param> P_new2 (NSubFaults);
                    //copy q model in P_new
                    std::copy(P.begin() + (q * NSubFaults), P.begin() + (q+1) * NSubFaults, P_new2.begin());
                    //exchange p parameters subbfault with p
                    std::copy(P.begin() + (p * NSubFaults) + pP, P.begin() + p * NSubFaults +pP + 1 , P_new2.begin() + pP);
                    Eigen::VectorXd old_line2 = storage_matrixes[q].row(pP);
                    double llk2 = llk_easy_i (pP, data, t_list, P_new2, G, RES_matrix, storage_matrixes[q]);


                    

                    double alpha_swap = std::min(0.0, (llk1 - llk[p])/T[p] + (llk2 - llk[q])/T[q]);
                    double u_swap     = std::log(unif_dist_intern(gen));
                    //swap_try[p]++;
                    //swap_try[q]++;

                    if (u_swap <= alpha_swap) {
                        //swap_rate[p]++;
                        //swap_rate[q]++;
                        if ((p<bds.ncold) || (q<bds.ncold))
                        swap_rate++;
                        std::copy(P_new.begin() , P_new.end(), P.begin() + (p * NSubFaults));
                        llk[p]=llk1;
                        std::copy(P_new2.begin() , P_new2.end(), P.begin() + (q * NSubFaults));
                        llk[q]=llk2;
                    }
                    else {storage_matrixes[p].row(pP) = old_line;storage_matrixes[q].row(pP) = old_line2;}
            }}

            
            
            #pragma endregion

        }


        #pragma omp single
        {std::cout<<"\n END OF PHASE 1 ";}
        for (int it=bds.burn_in*NSubFaults*2/3; it<bds.burn_in*NSubFaults ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma adjust : (" << SIGMAS[c][nparam]<<" ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c];}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            

            #pragma endregion

            
            #pragma region adaptation 

            

            if (it > 0 && it % adapt_window == 0) {

                     
                std::vector<double> acc_rate (bds.nchains);
                for (int c = 0; c < bds.nchains; c++) {
                    acc_rate[c] = (double)accepts_chain[c] / adapt_window;
                    accepts_chain[c] = 0; 
                }
                // On adapte uniquement si on est dans la période de Burn-in
                
                for (int c = 0; c < bds.nchains; c++) {
                    //sigma                        
                    if (acc_rate[c] < 0.20 )  {
                    SIGMAS[c][nparam] *= 0.9 ;
                    } else if (acc_rate[c] > 0.30)  SIGMAS[c][nparam] *= 1.1; 
                }
            #pragma endregion

            }
            }

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                //if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;

                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;
                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

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
                    int pP = random_index(gen);
                    std::vector<Easy_Param> P_new (NSubFaults);
                    //copy p model in P_new
                    std::copy(P.begin() + (p * NSubFaults), P.begin() + (p+1) * NSubFaults, P_new.begin());
                    //exchange q parameters subbfault with p
                    std::copy(P.begin() + (q * NSubFaults) + pP, P.begin() + q * NSubFaults +pP + 1 , P_new.begin() + pP);

                    Eigen::VectorXd old_line = storage_matrixes[p].row(pP);
                    double llk1 = llk_easy_i (pP, data, t_list, P_new, G, RES_matrix, storage_matrixes[p]);


                    std::vector<Easy_Param> P_new2 (NSubFaults);
                    //copy q model in P_new
                    std::copy(P.begin() + (q * NSubFaults), P.begin() + (q+1) * NSubFaults, P_new2.begin());
                    //exchange p parameters subbfault with p
                    std::copy(P.begin() + (p * NSubFaults) + pP, P.begin() + p * NSubFaults +pP + 1 , P_new2.begin() + pP);
                    Eigen::VectorXd old_line2 = storage_matrixes[q].row(pP);
                    double llk2 = llk_easy_i (pP, data, t_list, P_new2, G, RES_matrix, storage_matrixes[q]);


                    

                    double alpha_swap = std::min(0.0, (llk1 - llk[p])/T[p] + (llk2 - llk[q])/T[q]);
                    double u_swap     = std::log(unif_dist_intern(gen));
                    //swap_try[p]++;
                    //swap_try[q]++;

                    if (u_swap <= alpha_swap) {
                        //swap_rate[p]++;
                        //swap_rate[q]++;
                        swap_rate++;
                        std::copy(P_new.begin() , P_new.end(), P.begin() + (p * NSubFaults));
                        llk[p]=llk1;
                        std::copy(P_new2.begin() , P_new2.end(), P.begin() + (q * NSubFaults));
                        llk[q]=llk2;
                    }
                    else {storage_matrixes[p].row(pP) = old_line;storage_matrixes[q].row(pP) = old_line2;}
            }}

            
            
            #pragma endregion

       }


        #pragma region adapt
        #pragma omp single
        {std::cout<<"\n END OF PHASE 2 ";
        double mean1=0;
        double mean2=0;
        double mean3=0;
        double mean4=0;
        for (int i=0;i<bds.ncold; i++){
            mean1+= SIGMAS[i][0];
            mean2+= SIGMAS[i][1];
            mean3+= SIGMAS[i][2];
            mean4+= SIGMAS[i][3];
        }
        mean1/=bds.ncold;
        mean2/=bds.ncold;
        mean3/=bds.ncold;
        mean4/=bds.ncold;
        for (int i=0; i<bds.nchains; i++){
            if (i<bds.ncold){SIGMAS[i][0]= mean1; SIGMAS[i][1]=mean2; SIGMAS[i][2] = mean3; SIGMAS[i][4] = mean4;}
            else{SIGMAS[i][0]= mean1*std::sqrt(T[i]);SIGMAS[i][1]= mean2*std::sqrt(T[i]);SIGMAS[i][2]= mean3*std::sqrt(T[i]);SIGMAS[i][3]= mean4*std::sqrt(T[i]); }
        }
        #pragma endregion

        
        }
        for (int it=bds.burn_in*NSubFaults; it<maxint*NSubFaults ; it++ ){

            #pragma omp single
            {
            
            #pragma region affichage

            if (it > 0 && (it % 10000 == 0) ) {
                    std::cout <<"\n"<< std::setw(3) << (100 * it / (maxint*NSubFaults)) << "% done" << std::endl;
                    for (int c = 0; c < std::min(bds.nchains,6); c++) {
                        std::cout << "\nChain " << c 
                                  << " -> Accept : " << 100.0 * accepts_chain[c] / adapt_window << " %"
                                  << " |  sigma : (" << SIGMAS[c][0]<<" , "<<SIGMAS[c][1]<<" , "<< SIGMAS[c][2]<<" , "<< SIGMAS[c][3] << " ) "
                                  << " |  T : " << T[c] << " | llk : "<<llk[c] ;}

                    //if  (swap_try[c]!=0)   std::cout  << " |  swap : "<< 100.0 * swap_rate[c] / swap_try[c] <<" % ";}
                    std::cout
                              << "  \nBest llk: " << best_llk << std::endl;
                    
                }
            if (it%adapt_window==0) {for (int c=0; c<bds.nchains; c++) accepts_chain[c]=0 ;}
            

            #pragma endregion


            }
            

            #pragma omp for
            for (int ichain=0; ichain<bds.nchains ;  ichain++){

                #pragma region new_model


                double p = unif_dist_intern(gen);
                int i = sfindex[ichain] ;
                if (p<geometric_proba) { sfindex[ichain] = random_index(gen) ; i = sfindex[ichain];}

                if (ichain==5) subfault_index[i]++;


                std::vector<Easy_Param> P_new (NSubFaults);
                std::copy(P.begin() + (ichain * NSubFaults), P.begin() + (ichain+1) * NSubFaults, P_new.begin());

                //newmodel
                double prop = unif_dist_plus(gen) * (bds.a_sigma_k_sup   - bds.a_sigma_k_inf) * SIGMAS[ichain][0]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].a_sigma_k > bds.a_sigma_k_sup)   P_new[i].a_sigma_k = 2 * bds.a_sigma_k_sup - P_new[i].a_sigma_k - prop ; 
                else if (prop + P_new[i].a_sigma_k < bds.a_sigma_k_inf)  P_new[i].a_sigma_k = 2 * bds.a_sigma_k_inf - P_new[i].a_sigma_k - prop ;
                else  P_new[i].a_sigma_k   =  P_new[i].a_sigma_k +  prop;

                
                prop = unif_dist_plus(gen) * (bds.super_big_param_sup   - bds.super_big_param_inf) * SIGMAS[ichain][1]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].super_big_param > bds.super_big_param_sup)   P_new[i].super_big_param = 2 * bds.super_big_param_sup - P_new[i].super_big_param - prop ; 
                else if (prop + P_new[i].super_big_param < bds.super_big_param_inf)  P_new[i].super_big_param = 2 * bds.super_big_param_inf - P_new[i].super_big_param - prop ;
                else  P_new[i].super_big_param   =  P_new[i].super_big_param +  prop;


                prop = unif_dist_plus(gen) * (bds.big_param_sup   - bds.big_param_inf) * SIGMAS[ichain][2]*SIGMAS[ichain][nparam];
                if (prop + P_new[i].big_param > bds.big_param_sup)   P_new[i].big_param = 2 * bds.big_param_sup - P_new[i].big_param - prop ; 
                else if (prop + P_new[i].big_param < bds.big_param_inf)  P_new[i].big_param = 2 * bds.big_param_inf - P_new[i].big_param - prop ;
                else  P_new[i].big_param   =  P_new[i].big_param +  prop;

                Eigen::VectorXd old_line = storage_matrixes[ichain].row(i);

                double Enew = llk_easy_i (i, data, t_list, P_new, G, RES_matrix, storage_matrixes[ichain]);
                
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
                            if (Enew>best_llk) {
                        best_llk=Enew ;
                        best_model = P_new ; }}
                    }
                }
                else storage_matrixes[ichain].row(i) = old_line;

                if (ichain<bds.ncold) savers[ichain].save_step(P_new, llk[ichain]);

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
                    int pP = random_index(gen);
                    std::vector<Easy_Param> P_new (NSubFaults);
                    //copy p model in P_new
                    std::copy(P.begin() + (p * NSubFaults), P.begin() + (p+1) * NSubFaults, P_new.begin());
                    //exchange q parameters subbfault with p
                    std::copy(P.begin() + (q * NSubFaults) + pP, P.begin() + q * NSubFaults +pP + 1 , P_new.begin() + pP);

                    Eigen::VectorXd old_line = storage_matrixes[p].row(pP);
                    double llk1 = llk_easy_i (pP, data, t_list, P_new, G, RES_matrix, storage_matrixes[p]);


                    std::vector<Easy_Param> P_new2 (NSubFaults);
                    //copy q model in P_new
                    std::copy(P.begin() + (q * NSubFaults), P.begin() + (q+1) * NSubFaults, P_new2.begin());
                    //exchange p parameters subbfault with p
                    std::copy(P.begin() + (p * NSubFaults) + pP, P.begin() + p * NSubFaults +pP + 1 , P_new2.begin() + pP);
                    Eigen::VectorXd old_line2 = storage_matrixes[q].row(pP);
                    double llk2 = llk_easy_i (pP, data, t_list, P_new2, G, RES_matrix, storage_matrixes[q]);


                    

                    double alpha_swap = std::min(0.0, (llk1 - llk[p])/T[p] + (llk2 - llk[q])/T[q]);
                    double u_swap     = std::log(unif_dist_intern(gen));
                    //swap_try[p]++;
                    //swap_try[q]++;

                    if (u_swap <= alpha_swap) {
                        //swap_rate[p]++;
                        //swap_rate[q]++;
                        swap_rate++;
                        std::copy(P_new.begin() , P_new.end(), P.begin() + (p * NSubFaults));
                        llk[p]=llk1;
                        std::copy(P_new2.begin() , P_new2.end(), P.begin() + (q * NSubFaults));
                        llk[q]=llk2;
                    }
                    else {storage_matrixes[p].row(pP) = old_line;storage_matrixes[q].row(pP) = old_line2;}
            }}

            
            
            #pragma endregion

     }

    }
    #pragma endregion

    

    std::cout<<"\nFinal best llk : "<<best_llk;
    std::cout<<"\n swap every "<<maxint*NSubFaults/(swap_rate*1.0/bds.ncold)<<"   iterations for cold chains";
    
    return best_model ; 
}


#endif
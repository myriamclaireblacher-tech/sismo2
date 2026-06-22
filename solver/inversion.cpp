#include "inversion.hpp"

PT_param::PT_param(double k_a_sigma_inf_ent, double k_a_sigma_sup_ent, double b_a_inf_ent, double b_a_sup_ent, double D_c_inv_inf_ent, double D_c_inv_sup_ent, double Dtau_asigma_inf_ent, double Dtau_asigma_sup_ent, double Tmax_ent,
            int nchains_ent, int ncold_ent, int burn_in_steps_ent)
    :k_a_sigma_inf(k_a_sigma_inf_ent),k_a_sigma_sup(k_a_sigma_sup_ent),b_a_inf(b_a_inf_ent), b_a_sup (b_a_sup_ent), D_c_inv_inf(D_c_inv_inf_ent),
    D_c_inv_sup (D_c_inv_sup_ent), Dtau_asigma_inf(Dtau_asigma_inf_ent), Dtau_asigma_sup (Dtau_asigma_sup_ent), V0_inf(0.1*Vinf), V0_sup(1.1*Vinf), T_max(Tmax_ent),
    nchains (nchains_ent), ncold(ncold_ent), burn_in_steps(burn_in_steps_ent)
    {}   


double pi(const Param & P, const std::vector<sunrealtype> & t_list, const Eigen::Ref<Eigen::RowVectorXd> & data, Fault & F, Eigen::Ref<Eigen::RowVectorXd> & slip_list){
    int ret = F.ODE_solver(t_list, slip_list, P);
    if (ret<0) return -10e4;
    return -(data - slip_list).array().square().sum();}


Param parallel_tempering_1faille(int n_steps, int n_markow, int n_cold,int  burn_in_steps, const PT_param PT, const Eigen::Ref<Eigen::RowVectorXd> & data, const std::vector<sunrealtype> & t_list, int seed ){

    int accept_count = 0;
    int crash_count = 0;
    //create  Temperture ladder
    std::mt19937 gen(seed); 
    std::vector<double> T(n_markow); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0); 
    for (int i=0;i<n_markow;i++){
        if (i<n_cold) T[i]=1.0 ;
        else T[i]= std :: exp( (i-n_cold+1)*1.0/(n_markow-n_cold) * std::log(PT.T_max) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    
    //rand\om interger
    std::uniform_int_distribution<int> rand_chain(0, n_markow - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(n_markow);

    //model storage
    std::vector<Param> P;
    P.resize(n_markow);


    //open storage file
    std::vector<std::ofstream> files;
    files.resize(n_markow) ;
    for (int j=0;j<n_markow; j++){
        files[j].open("test_chain_cold_" + std::to_string(j) + ".bin", std::ios::binary);
    }

    double best_llk;
    Param best_param;
    int best_it;

    //sigma

        std::vector<double> sigmas(n_markow, 0.01);
    
        // count accept rain for each chain
        std::vector<int> accepts_chain(n_markow, 0);
        const int adapt_window = 200; // Evaluation every 200 steps


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

            best_llk=llk[0];
            best_param=P[0];

    
            //save initial model
        for (int j=0; j<n_markow;j++){
                files[j].write(reinterpret_cast<const char*>(&llk[j]), sizeof(double));
                files[j].write(reinterpret_cast<const char*>(&P[j]), sizeof(Param));}}
        
        
        //begin parallel tempering

        for (int it = 0; it < n_steps ; it ++ ){

            #pragma omp single
            {
            if (it % (n_steps / 10) == 0) {
                std::cout <<"\n"<< std::setw(3) << (100 * it / n_steps) << "% done" << std::endl;
            }}  

            #pragma omp single
            {
                if (it > 0 && it % (n_steps / 10) == 0) {
                    std::cout << std::setw(3) << (100 * it / n_steps) << "% done | "
                              << "\nChain 0 -> Accept : " << 100*accepts_chain[0]/adapt_window<<" % |   sigma : "<<sigmas[0]
                              << "  \nChain 1 -> Accept: " << 100*accepts_chain[1]/adapt_window<<" % |   sigma : "<<sigmas[1]
                              << "  \nChain 2 -> Accept: " << 100*accepts_chain[2]/adapt_window<<" % |   sigma : "<<sigmas[2]
                              << "  \nChain 3 -> Accept: " << 100*accepts_chain[3]/adapt_window<<" % |   sigma : "<<sigmas[3]
                              << "  \nChain 4 -> Accept: " << 100*accepts_chain[4]/adapt_window<<" % |   sigma : "<<sigmas[4]
                              << "  \nChain 5 -> Accept: " << 100*accepts_chain[5]/adapt_window<<" % |   sigma : "<<sigmas[5]
                              << "  \nCrash EDO: " << crash_count <<" / "<< n_steps / 10 
                              << "  \nLast llk: " << llk[0] << std::endl;
                    
                    // Remise à zéro pour la prochaine tranche de 10%
    
                    crash_count = 0;
                }

                //update sigma
                if (it > 0 && it % adapt_window == 0) {
                    
                    // On adapte uniquement si on est dans la période de Burn-in
                    if (it < burn_in_steps) {
                        for (int c = 0; c < n_markow; c++) {
                            double acc_rate = (double)accepts_chain[c] / adapt_window;
                            
                            if (acc_rate < 0.20) {
                                sigmas[c] *= 0.9; 
                            } else if (acc_rate > 0.30) {
                                sigmas[c] *= 1.1; 
                            }
                        }
                    }
                    
                    // Remise à zéro des compteurs pour la prochaine fenêtre de 200 pas
                    // (On le fait même après le burn_in pour pouvoir afficher les stats)
                    for (int c = 0; c < n_markow; c++) {
                        accepts_chain[c] = 0; 
                    }
                }
            }
            //update Markow chains
            #pragma omp for schedule(dynamic)
            for (int ichain = 0; ichain < n_markow ; ichain++){
                //new model proposal


                double prop;
                double k_a_sigma;
                prop = unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigmas[ichain];
                if (prop + P[ichain].k_a_sigma > PT.k_a_sigma_sup)   k_a_sigma = 2 * PT.k_a_sigma_sup - P[ichain].k_a_sigma - prop ; 
                else if (prop + P[ichain].k_a_sigma < PT.k_a_sigma_inf)  k_a_sigma = 2 * PT.k_a_sigma_inf - P[ichain].k_a_sigma - prop ;
                else  k_a_sigma   =  P[ichain].k_a_sigma +  prop;
                double b_a;
                prop = unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigmas[ichain];
                if (prop + P[ichain].b_a > PT.b_a_sup)   b_a = 2 * PT.b_a_sup - P[ichain].b_a - prop ; 
                else if (prop + P[ichain].b_a < PT.b_a_inf)  b_a = 2 * PT.b_a_inf - P[ichain].b_a - prop ;
                else b_a  = P[ichain].b_a + prop;
                double D_c_inv;
                
                prop = unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigmas[ichain];
                if (prop + P[ichain].D_c_inv > PT.D_c_inv_sup)   D_c_inv = 2 * PT.D_c_inv_sup - P[ichain].D_c_inv - prop ; 
                else if (prop + P[ichain].D_c_inv < PT.D_c_inv_inf)  D_c_inv = 2 * PT.D_c_inv_inf - P[ichain].D_c_inv - prop ;
                else  D_c_inv   = P[ichain].D_c_inv  + prop;

                double Dtau_asigma;
                prop = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigmas[ichain];
                if (prop + P[ichain].Dtau_asigma > PT.Dtau_asigma_sup)   Dtau_asigma = 2 * PT.Dtau_asigma_sup - P[ichain].Dtau_asigma - prop ; 
                else if (prop + P[ichain].Dtau_asigma < PT.Dtau_asigma_inf)  Dtau_asigma = 2 * PT.Dtau_asigma_inf - P[ichain].Dtau_asigma - prop ;
                else Dtau_asigma   = P[ichain].Dtau_asigma  + prop;

                double V0_;
                prop = unif_dist_plus(gen2) * (PT.V0_sup - PT.V0_inf) * sigmas[ichain];
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
                    if (Enew <= -9e4) {
                        #pragma omp atomic
                        crash_count++;
                    }
                
                }






                if (accept){
                    llk[ichain] = Enew;
                    P[ichain] = P_new ;
                    accepts_chain[ichain]++;
                    
                }
                if ((ichain< n_markow)&&(accept=true)) {
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

                if (llk[0]>best_llk){best_llk=llk[0]; best_param=P[0]; best_it=it;}
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
    for (int j=0; j<n_markow; j++){
        if(files[j].is_open()) files[j].close();
    }

    std::cout<<"\n best llk at it  "<<best_it<<" : "<<best_llk;
    return best_param;
    }


std::vector<Param> parallel_tempering_corrected(const int maxint,  const PT_param PT,  const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype> & t_list, const int seed, bool hotchains )
                                    {

    int accept_count = 0;
    int crash_count = 0;

    //check size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Temperatures
    std::mt19937 gen(seed); 
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0); 
    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( (i-PT.ncold+1)*1.0/(PT.nchains-PT.ncold) * std::log(PT.T_max) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    
    //rand\om interger
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //model storage
    std::vector<Param> P;
    P.resize(PT.nchains);

    //store the models of the cold chains
    
    std::vector<ColdChainSaver> savers;
    if (hotchains){for (int i = 0; i < PT.nchains; ++i) {
        savers.emplace_back(i);
    }}
    else{
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }}


    double best_llk=-10e8f;
    std::vector<Param> best_param;
    best_param.resize(NSubFaults);
    int best_it;

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //sigma
    std::vector<double> sigmas(PT.nchains, 0.01);
    
    // count accept rate for each chain
    std::vector<int> accepts_chain(PT.nchains, 0);
    const int adapt_window = 200; // Evaluation every 200 steps


    #pragma omp parallel 
    {
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 


        //initial models
        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            double llk_i ;
            //propose new model 
            
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;
                double p5 = (PT.V0_inf + PT.V0_sup) /2.0 ;

                

                M[j * NSubFaults + i] = Param(p1, p2, p3 , p4 , p5);
            }
            
            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk2(t_list, data, G , work);
            llk[j]=llk_i;
            if (hotchains)
            {
               savers[j].save_step(work.pP, llk_i);
            }
            else {if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);}


        }
        
        
        //begin parallel tempering

        for (int it = 0; it < maxint ; it ++ ){

            #pragma omp single
            {
            if (it % (maxint / 10) == 0) {
                std::cout <<"\n"<< std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
            }}  

            #pragma omp single
            {
                if (it > 0 && it % (maxint / 10) == 0) {
                    std::cout << std::setw(3) << (100 * it / maxint) << "% done | "
                              << "\nChain 0 -> Accept : " << 100*accepts_chain[0]/adapt_window<<" % |   sigma : "<<sigmas[0]
                              << "  \nChain 1 -> Accept: " << 100*accepts_chain[1]/adapt_window<<" % |   sigma : "<<sigmas[1]
                              << "  \nChain 2 -> Accept: " << 100*accepts_chain[2]/adapt_window<<" % |   sigma : "<<sigmas[2]
                              << "  \nChain 3 -> Accept: " << 100*accepts_chain[3]/adapt_window<<" % |   sigma : "<<sigmas[3]
                              << "  \nChain 4 -> Accept: " << 100*accepts_chain[4]/adapt_window<<" % |   sigma : "<<sigmas[4]
                              << "  \nChain 5 -> Accept: " << 100*accepts_chain[5]/adapt_window<<" % |   sigma : "<<sigmas[5]
                              << "  \nCrash EDO: " << crash_count <<" / "<< maxint / 10 
                              << "  \nLast llk: " << llk[0] << std::endl;
                    
                    // Remise à zéro pour la prochaine tranche de 10%
    
                    crash_count = 0;
                }

                //update sigma
                if (it > 0 && it % adapt_window == 0) {
                    
                    // On adapte uniquement si on est dans la période de Burn-in
                    if (it < PT.burn_in_steps) {
                        for (int c = 0; c < PT.nchains; c++) {
                            double acc_rate = (double)accepts_chain[c] / adapt_window;
                            
                            if (acc_rate < 0.20) {
                                sigmas[c] *= 0.9; 
                            } else if (acc_rate > 0.30) {
                                sigmas[c] *= 1.1; 
                            }
                        }
                    }
                    
                    // Remise à zéro des compteurs pour la prochaine fenêtre de 200 pas
                    for (int c = 0; c < PT.nchains; c++) {
                        accepts_chain[c] = 0; 
                    }
                }
            }
            //update Markow chains
            #pragma omp for schedule(dynamic)
            for (int ichain = 0; ichain < PT.nchains ; ichain++){
                //new model proposal
                
                //update values of pP
                std::copy(M.begin() + (ichain * NSubFaults), 
                          M.begin() + ((ichain + 1) * NSubFaults), 
                          work.pP.begin());

                double prop;
                double k_a_sigma;

                for (int i=0; i<NSubFaults; i++){
                    prop = unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigmas[ichain];
                    if (prop + work.pP[i].k_a_sigma > PT.k_a_sigma_sup)   k_a_sigma = 2 * PT.k_a_sigma_sup - work.pP[i].k_a_sigma - prop ; 
                    else if (prop + work.pP[i].k_a_sigma < PT.k_a_sigma_inf)  k_a_sigma = 2 * PT.k_a_sigma_inf - work.pP[i].k_a_sigma - prop ;
                    else  k_a_sigma   =  work.pP[i].k_a_sigma +  prop;
                    double b_a;
                    prop = unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigmas[ichain];
                    if (prop + work.pP[i].b_a > PT.b_a_sup)   b_a = 2 * PT.b_a_sup - work.pP[i].b_a - prop ; 
                    else if (prop + work.pP[i].b_a < PT.b_a_inf)  b_a = 2 * PT.b_a_inf - work.pP[i].b_a - prop ;
                    else b_a  = work.pP[i].b_a + prop;
                    double D_c_inv;
                    
                    prop = unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigmas[ichain];
                    if (prop + work.pP[i].D_c_inv > PT.D_c_inv_sup)   D_c_inv = 2 * PT.D_c_inv_sup - work.pP[i].D_c_inv - prop ; 
                    else if (prop + work.pP[i].D_c_inv < PT.D_c_inv_inf)  D_c_inv = 2 * PT.D_c_inv_inf - work.pP[i].D_c_inv - prop ;
                    else  D_c_inv   = work.pP[i].D_c_inv  + prop;

                    double Dtau_asigma;
                    prop = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigmas[ichain];
                    if (prop + work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup)   Dtau_asigma = 2 * PT.Dtau_asigma_sup - work.pP[i].Dtau_asigma - prop ; 
                    else if (prop + work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf)  Dtau_asigma = 2 * PT.Dtau_asigma_inf - work.pP[i].Dtau_asigma - prop ;
                    else Dtau_asigma   = work.pP[i].Dtau_asigma  + prop;

                    double V0_;
                    prop = unif_dist_plus(gen2) * (PT.V0_sup - PT.V0_inf) * sigmas[ichain];
                    if (prop + work.pP[i].V0_ > PT.V0_sup)   V0_ = 2 * PT.V0_sup - work.pP[i].V0_ - prop ; 
                    else if (prop + work.pP[i].V0_ < PT.V0_inf)  V0_ = 2 * PT.V0_inf - work.pP[i].V0_ - prop ;
                    else V0_   = work.pP[i].V0_  + prop;
                
                    work.pP[i]=Param(k_a_sigma, b_a, D_c_inv, Dtau_asigma, V0_);
                }


                double Enew = compute_llk2(t_list, data, G , work);

                bool accept = false ;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen2) );
                accept = (u <= alpha);



                if (ichain == 0) {
                    if (Enew <= -9e4) {
                        #pragma omp atomic
                        crash_count++;
                    }
                
                }

                if (accept){
                    //copy new model in M
                    std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                    llk[ichain] = Enew;
                    accepts_chain[ichain]++;
                }

                if (hotchains) savers[ichain].save_step(work.pP, Enew);
                else{if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);}            //plus tard save que sur certaines itérations
            
            }

            #pragma omp single
                { 

                if (llk[0]>best_llk){best_llk=llk[0]; std::copy(M.begin(), M.begin() + NSubFaults, best_param.begin()) ; best_it=it;}

                for (int s = 0; s < PT.nchains - 1; s++) {
                    int p = rand_chain(gen);
                    int q = rand_chain(gen);
                    if ((p == q) ||  (T[p] == T[q])) continue;

                    // Formule théorique du Parallel Tempering
                    double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
                    double u_swap     = std::log(unif_dist(gen));

                    if (u_swap <= alpha_swap) {
                        std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                    }
                }}


        }
        
    }
    
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nlast llk computed"<< llk[0];

    std::cout<<"\n best llk at it  "<<best_it<<" : "<<best_llk;
    return best_param;
    }


std::vector<Param> parallel_tempering_2sf(const int maxint,  const PT_param PT,  const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype> & t_list, const int seed, bool hotchains )
                                    {

    int accept_count = 0;
    int crash_count = 0;

    //check size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Temperatures
    std::mt19937 gen(seed); 
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0); 
    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( (i-PT.ncold+1)*1.0/(PT.nchains-PT.ncold) * std::log(PT.T_max) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    
    //rand\om interger
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //model storage
    std::vector<Param> P;
    P.resize(PT.nchains);

    //store the models of the cold chains
    
    std::vector<ColdChainSaver> savers;
    if (hotchains){for (int i = 0; i < PT.nchains; ++i) {
        savers.emplace_back(i);
    }}
    else{
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }}


    double best_llk=-10e8f;
    std::vector<Param> best_param;
    best_param.resize(NSubFaults);
    int best_it;

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //sigma
    std::vector<double> sigmas(PT.nchains, 0.01);
    
    // count accept rate for each chain
    std::vector<int> accepts_chain(PT.nchains, 0);
    const int adapt_window = 200; // Evaluation every 200 steps


    #pragma omp parallel 
    {
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);
        std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 


        //initial models
        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            double llk_i ;
            //propose new model 
            
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;
                double p5 = (PT.V0_inf + PT.V0_sup) /2.0 ;

                

                M[j * NSubFaults + i] = Param(p1, p2, p3 , p4 , p5);
            }
            
            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk2(t_list, data, G , work);
            llk[j]=llk_i;
            if (hotchains)
            {
               savers[j].save_step(work.pP, llk_i);
            }
            else {if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);}


        }
        
        
        //begin parallel tempering

        for (int it = 0; it < maxint ; it ++ ){

            #pragma omp single
            {
            if (it % (maxint / 10) == 0) {
                std::cout <<"\n"<< std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
            }}  

            #pragma omp single
            {
                if (it > 0 && (it % 50000 == 0) ) {
                    std::cout << std::setw(3) << (100 * it / maxint) << "% done | "
                              << "\nChain 0 -> Accept : " << 100*accepts_chain[0]/adapt_window<<" % |   sigma : "<<sigmas[0]
                              << "  \nChain 1 -> Accept: " << 100*accepts_chain[1]/adapt_window<<" % |   sigma : "<<sigmas[1]
                              << "  \nChain 2 -> Accept: " << 100*accepts_chain[2]/adapt_window<<" % |   sigma : "<<sigmas[2]
                              << "  \nChain 3 -> Accept: " << 100*accepts_chain[3]/adapt_window<<" % |   sigma : "<<sigmas[3]
                              << "  \nChain 4 -> Accept: " << 100*accepts_chain[4]/adapt_window<<" % |   sigma : "<<sigmas[4]
                              << "  \nChain 5 -> Accept: " << 100*accepts_chain[5]/adapt_window<<" % |   sigma : "<<sigmas[5]
                              << "  \nCrash EDO: " << crash_count <<" / "<< maxint / 10 
                              << "  \nLast llk: " << llk[0] << std::endl;
                    
                    // Remise à zéro pour la prochaine tranche de 10%
    
                    crash_count = 0;
                }

                //update sigma
                if (it > 0 && it % adapt_window == 0) {
                    
                    // On adapte uniquement si on est dans la période de Burn-in
                    if (it < PT.burn_in_steps) {
                        for (int c = 0; c < PT.nchains; c++) {
                            double acc_rate = (double)accepts_chain[c] / adapt_window;
                            
                            if (acc_rate < 0.20) {
                                sigmas[c] *= 0.9; 
                            } else if (acc_rate > 0.30) {
                                sigmas[c] *= 1.1; 
                            }
                        }
                    }
                    
                    // Remise à zéro des compteurs pour la prochaine fenêtre de 200 pas
                    for (int c = 0; c < PT.nchains; c++) {
                        accepts_chain[c] = 0; 
                    }
                }
            }
            //update Markow chains
            #pragma omp for schedule(dynamic)
            for (int ichain = 0; ichain < PT.nchains ; ichain++){
                //new model proposal
                
                //update values of pP
                std::copy(M.begin() + (ichain * NSubFaults), 
                          M.begin() + ((ichain + 1) * NSubFaults), 
                          work.pP.begin());

                double prop;
                double k_a_sigma;

                int i = random_index(gen2) ;
                
                prop = unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigmas[ichain];
                if (prop + work.pP[i].k_a_sigma > PT.k_a_sigma_sup)   k_a_sigma = 2 * PT.k_a_sigma_sup - work.pP[i].k_a_sigma - prop ; 
                else if (prop + work.pP[i].k_a_sigma < PT.k_a_sigma_inf)  k_a_sigma = 2 * PT.k_a_sigma_inf - work.pP[i].k_a_sigma - prop ;
                else  k_a_sigma   =  work.pP[i].k_a_sigma +  prop;
                double b_a;
                prop = unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigmas[ichain];
                if (prop + work.pP[i].b_a > PT.b_a_sup)   b_a = 2 * PT.b_a_sup - work.pP[i].b_a - prop ; 
                else if (prop + work.pP[i].b_a < PT.b_a_inf)  b_a = 2 * PT.b_a_inf - work.pP[i].b_a - prop ;
                else b_a  = work.pP[i].b_a + prop;
                double D_c_inv;
                
                prop = unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigmas[ichain];
                if (prop + work.pP[i].D_c_inv > PT.D_c_inv_sup)   D_c_inv = 2 * PT.D_c_inv_sup - work.pP[i].D_c_inv - prop ; 
                else if (prop + work.pP[i].D_c_inv < PT.D_c_inv_inf)  D_c_inv = 2 * PT.D_c_inv_inf - work.pP[i].D_c_inv - prop ;
                else  D_c_inv   = work.pP[i].D_c_inv  + prop;

                double Dtau_asigma;
                prop = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigmas[ichain];
                if (prop + work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup)   Dtau_asigma = 2 * PT.Dtau_asigma_sup - work.pP[i].Dtau_asigma - prop ; 
                else if (prop + work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf)  Dtau_asigma = 2 * PT.Dtau_asigma_inf - work.pP[i].Dtau_asigma - prop ;
                else Dtau_asigma   = work.pP[i].Dtau_asigma  + prop;

                double V0_;
                prop = unif_dist_plus(gen2) * (PT.V0_sup - PT.V0_inf) * sigmas[ichain];
                if (prop + work.pP[i].V0_ > PT.V0_sup)   V0_ = 2 * PT.V0_sup - work.pP[i].V0_ - prop ; 
                else if (prop + work.pP[i].V0_ < PT.V0_inf)  V0_ = 2 * PT.V0_inf - work.pP[i].V0_ - prop ;
                else V0_   = work.pP[i].V0_  + prop;
            
                work.pP[i]=Param(k_a_sigma, b_a, D_c_inv, Dtau_asigma, V0_);
            


                double Enew = compute_llk2(t_list, data, G , work);

                bool accept = false ;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen2) );
                accept = (u <= alpha);



                if (ichain == 0) {
                    if (Enew <= -9e4) {
                        #pragma omp atomic
                        crash_count++;
                    }
                
                }

                if (accept){
                    //copy new model in M
                    std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                    llk[ichain] = Enew;
                    accepts_chain[ichain]++;
                }

                if (hotchains) savers[ichain].save_step(work.pP, Enew);
                else{if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);}            //plus tard save que sur certaines itérations
            
            }

            #pragma omp single
                { 

                if (llk[0]>best_llk){best_llk=llk[0]; std::copy(M.begin(), M.begin() + NSubFaults, best_param.begin()) ; best_it=it;}

                for (int s = 0; s < PT.nchains - 1; s++) {
                    int p = rand_chain(gen);
                    int q = rand_chain(gen);
                    if ((p == q) ||  (T[p] == T[q])) continue;

                    // Formule théorique du Parallel Tempering
                    double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
                    double u_swap     = std::log(unif_dist(gen));

                    if (u_swap <= alpha_swap) {
                        std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                        std::swap(llk[p], llk[q]);
                    }
                }}


        }
        
    }
    
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nlast llk computed"<< llk[0];

    std::cout<<"\n best llk at it  "<<best_it<<" : "<<best_llk;
    return best_param;
    }


ThreadWorkspace::ThreadWorkspace(int t_list_size) {
    pP.resize(NSubFaults);
    RES_matrix.resize(3 * Nstations, t_list_size);
    storage_matrix.resize(NSubFaults, t_list_size);
}

ThreadWorkspace::ThreadWorkspace() = default; 



ColdChainSaver::ColdChainSaver(int cold_idx) {
    file.open("chain_cold_" + std::to_string(cold_idx) + ".bin", std::ios::binary);
}

void ColdChainSaver::save_step(const std::vector<Param>& pP, double energy) {
    file.write(reinterpret_cast<const char*>(&energy), sizeof(double));
    file.write(reinterpret_cast<const char*>(pP.data()), pP.size() * sizeof(Param));
}

double compute_llk(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work){

    int SR = surface_response(work.pP, t_list, work.fault, G, work.RES_matrix, work.storage_matrix);
    if (SR!=0) return -std::numeric_limits<double>::infinity();
    double llk = - (data - work.RES_matrix).array().square().sum();
    return llk;
}

double compute_llk2(const std::vector<sunrealtype>& t_list, const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data,
                    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, ThreadWorkspace& work){

    int SR = surface_response(work.pP, t_list, work.fault, G, work.RES_matrix, work.storage_matrix);
    if (SR!=0) return -1e6f;
    double llk = - (data - work.RES_matrix).array().square().sum();
    return llk;
}

/*
int parallel_tempering(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed,const double sigma){

    //check t_list size, data size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( unif_dist(gen) * std::log(PT.T_max) ) ; //loguniform repartition

    }

    //random integer 
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 2); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);
    #pragma omp parallel 
    {
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0); ///////////////////////


        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            bool reject = true;
            double llk_i ;
            //propose new model 
            while (reject)
            {
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;

                // On applique une infime variation de 1% maximum propre à chaque chaîne (j) et sous-faille (i)
                // pour que les 10 chaînes ne partent pas exactement du même pixel
                double bruit = 0.99 + 0.02 * unif_dist_intern(gen2); // Nombre entre 0.99 et 1.01

                M[j * NSubFaults + i] = Param(p1 * bruit, p2 * bruit, p3 * bruit, p4 * bruit);
            }

            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk(t_list, data, G , work);
            if (llk_i != -std::numeric_limits<double>::infinity() ) {reject = false;}
            }
            llk[j]=llk_i;
            if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);

        }
        


    //Parallel tempering


    for (int it = 0; it < maxint ; it ++ ){

        #pragma omp single
        {
        if (it % (maxint / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
        }}

        #pragma omp for schedule(dynamic)
        for (int ichain = 0; ichain < PT.nchains ; ichain++){

            //copy model values
            work.pP.assign(M.begin() + (ichain * NSubFaults), M.begin() + ((ichain + 1) * NSubFaults));

            //New model proposal
            for (int i=0; i<NSubFaults; i++){
                work.pP[i].k_a_sigma   += unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
                work.pP[i].b_a         += unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigma ;
                work.pP[i].D_c_inv     += unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma ;
                work.pP[i].Dtau_asigma += unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
            }

            // Check if out of bounds
            bool InBound = true;
            for (int i=0; i<NSubFaults; i++){
                if (work.pP[i].k_a_sigma   < PT.k_a_sigma_inf   || work.pP[i].k_a_sigma   > PT.k_a_sigma_sup   ||
                    work.pP[i].b_a         < PT.b_a_inf         || work.pP[i].b_a         > PT.b_a_sup         ||
                    work.pP[i].D_c_inv     < PT.D_c_inv_inf     || work.pP[i].D_c_inv     > PT.D_c_inv_sup     ||
                    work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf || work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup) 
                {
                    InBound = false;
                    break; 
                }
                }

            //accept rate of the new model
            bool accept = false ;
            double Enew = -std::numeric_limits<double>::infinity() ; 

            if (InBound) {
                Enew = compute_llk(t_list, data, G , work);
                if (Enew!=-std::numeric_limits<double>::infinity()){
                    double delta  = (Enew - llk[ichain])/T[ichain] ;
                    double alpha  = std::min(0.0, delta);
                    double u      = std::log(unif_dist_intern(gen2) );
                    accept = (u <= alpha);
                }
            }

            if (accept){
                std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                llk[ichain] = Enew;
                if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);            //plus tard save que sur certaines itérations
            }

        } //parallel end

        #pragma omp single
        { 
        for (int s = 0; s < PT.nchains - 1; s++) {
            int p = rand_chain(gen);
            int q = rand_chain(gen);
            if (p == q) continue;

            // Formule théorique du Parallel Tempering
            double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
            double u_swap     = std::log(unif_dist(gen));

            if (u_swap <= alpha_swap) {
                std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                std::swap(llk[p], llk[q]);
            }
        }
        
    }}}
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    return 0;
}

//à corriger, conditions initiales


int parallel_tempering_test_out_of_bounds(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed,const double sigma){

    int inbouds=0 ;
    int newmodel=0;
    int swapnum=0 ;
    int swapcold=0;
    //check t_list size, data size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( unif_dist(gen) * std::log(PT.T_max) ) ; //loguniform repartition

    }

    //random integer 
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 2); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);
    #pragma omp parallel 
    {
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0); ///////////////////////


        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            bool reject = true;
            double llk_i ;
            //propose new model 
            while (reject)
            {
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;

                // On applique une infime variation de 1% maximum propre à chaque chaîne (j) et sous-faille (i)
                // pour que les 10 chaînes ne partent pas exactement du même pixel
                double bruit = 0.99 + 0.02 * unif_dist_intern(gen2); // Nombre entre 0.99 et 1.01

                M[j * NSubFaults + i] = Param(p1 * bruit, p2 * bruit, p3 * bruit, p4 * bruit);
            }

            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk(t_list, data, G , work);
            if (llk_i != -std::numeric_limits<double>::infinity() ) {reject = false;}
            }
            llk[j]=llk_i;
            if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);

        }
        


    //Parallel tempering


    for (int it = 0; it < maxint ; it ++ ){

        #pragma omp single
        {
        if (it % (maxint / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
        }}

        #pragma omp for schedule(dynamic)
        for (int ichain = 0; ichain < PT.nchains ; ichain++){

            //copy model values
            work.pP.assign(M.begin() + (ichain * NSubFaults), M.begin() + ((ichain + 1) * NSubFaults));

            //New model proposal
            for (int i=0; i<NSubFaults; i++){
                work.pP[i].k_a_sigma   += unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
                work.pP[i].b_a         += unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigma ;
                work.pP[i].D_c_inv     += unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma ;
                work.pP[i].Dtau_asigma += unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
            }

            // Check if out of bounds
            bool InBound = true;
            for (int i=0; i<NSubFaults; i++){
                if (work.pP[i].k_a_sigma   < PT.k_a_sigma_inf   || work.pP[i].k_a_sigma   > PT.k_a_sigma_sup   ||
                    work.pP[i].b_a         < PT.b_a_inf         || work.pP[i].b_a         > PT.b_a_sup         ||
                    work.pP[i].D_c_inv     < PT.D_c_inv_inf     || work.pP[i].D_c_inv     > PT.D_c_inv_sup     ||
                    work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf || work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup) 
                {
                    InBound = false;
                    break; 
                }
                }

            
            #pragma omp atomic
            inbouds+=InBound;

            //accept rate of the new model
            bool accept = false ;
            double Enew = -std::numeric_limits<double>::infinity() ; 

            if (InBound) {
                Enew = compute_llk(t_list, data, G , work);
                if (Enew!=-std::numeric_limits<double>::infinity()){
                    double delta  = (Enew - llk[ichain])/T[ichain] ;
                    double alpha  = std::min(0.0, delta);
                    double u      = std::log(unif_dist_intern(gen2) );
                    accept = (u <= alpha);
                }
            }

            #pragma omp atomic
            newmodel+=accept;

            if (accept){
                std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                llk[ichain] = Enew;
                if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);            //plus tard save que sur certaines itérations
            }

        } //parallel end

        #pragma omp single
        { 
        for (int s = 0; s < PT.nchains - 1; s++) {
            int p = rand_chain(gen);
            int q = rand_chain(gen);
            if (p == q) continue;

            // Formule théorique du Parallel Tempering
            double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
            double u_swap     = std::log(unif_dist(gen));

            if (u_swap <= alpha_swap) {
                swapnum+=1 ;
                if (p<PT.ncold || q<PT.ncold) swapcold+=1;
                std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                std::swap(llk[p], llk[q]);
            }
        }
        
    }}}
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nInbounds : "<<inbouds;
    std::cout<<"\nNew Models explored : "<<newmodel;
    std::cout<<"\nswaps : "<<swapnum;
    std::cout<<"\nswaps with coldchains : "<<swapcold;

    return 0;
}



int parallel_tempering_miror(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed,const double sigma){

    int inbouds=0 ;
    int newmodel=0;
    int swapnum=0 ;
    int swapcold=0;
    int llk_computed=0 ;

    std::cout << "Taille de Param : " << sizeof(Param) << " octets\n";
    std::cout << "Taille exacte ecrite par etape : " << sizeof(double) + (NSubFaults * sizeof(Param)) << " octets\n";
    //check t_list size, data size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( unif_dist(gen) * std::log(PT.T_max) ) ; //loguniform repartition

    }

    //random integer 
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);
    #pragma omp parallel 
    {
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0); ///////////////////////


        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            bool reject = true;
            double llk_i ;
            //propose new model 
            while (reject)
            {
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;

                // On applique une infime variation de 1% maximum propre à chaque chaîne (j) et sous-faille (i)
                // pour que les 10 chaînes ne partent pas exactement du même pixel
                double bruit = 0.99 + 0.02 * unif_dist_intern(gen2); // Nombre entre 0.99 et 1.01

                M[j * NSubFaults + i] = Param(p1 * bruit, p2 * bruit, p3 * bruit, p4 * bruit);
            }

            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk(t_list, data, G , work);
            if (llk_i != -std::numeric_limits<double>::infinity() ) {reject = false;}
            }
            llk[j]=llk_i;
            if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);

        }
        


    //Parallel tempering


    for (int it = 0; it < maxint ; it ++ ){

        #pragma omp single
        {
        if (it % (maxint / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
        }}

        #pragma omp for schedule(dynamic)
        for (int ichain = 0; ichain < PT.nchains ; ichain++){

            //copy model values
            work.pP.assign(M.begin() + (ichain * NSubFaults), M.begin() + ((ichain + 1) * NSubFaults));

            //New model proposal with miror boundaries

            // Remplacement propre du bloc de proposition et du check Out-of-Bounds
            for (int i = 0; i < NSubFaults; i++) {
                
                // --- 1. Réflexion pour k_a_sigma ---
                double prop_k = unif_dist_plus(gen2) * (PT.k_a_sigma_sup - PT.k_a_sigma_inf) * sigma;
                double val_k = work.pP[i].k_a_sigma + prop_k;
                while (val_k < PT.k_a_sigma_inf || val_k > PT.k_a_sigma_sup) {
                    if (val_k < PT.k_a_sigma_inf) {
                        val_k = 2 * PT.k_a_sigma_inf - val_k; // Rebond sur la borne inf
                    } else if (val_k > PT.k_a_sigma_sup) {
                        val_k = 2 * PT.k_a_sigma_sup - val_k; // Rebond sur la borne sup
                    }
                }
                work.pP[i].k_a_sigma = val_k;

                // --- 2. Réflexion pour b_a ---
                double prop_b = unif_dist_plus(gen2) * (PT.b_a_sup - PT.b_a_inf) * sigma;
                double val_b = work.pP[i].b_a + prop_b;
                while (val_b < PT.b_a_inf || val_b > PT.b_a_sup) {
                    if (val_b < PT.b_a_inf) {
                        val_b = 2 * PT.b_a_inf - val_b;
                    } else if (val_b > PT.b_a_sup) {
                        val_b = 2 * PT.b_a_sup - val_b;
                    }
                }
                work.pP[i].b_a = val_b;

                // --- 3. Réflexion pour D_c_inv ---
                double prop_c = unif_dist_plus(gen2) * (PT.D_c_inv_sup - PT.D_c_inv_inf) * sigma;
                double val_c = work.pP[i].D_c_inv + prop_c;
                while (val_c < PT.D_c_inv_inf || val_c > PT.D_c_inv_sup) {
                    if (val_c < PT.D_c_inv_inf) {
                        val_c = 2 * PT.D_c_inv_inf - val_c;
                    } else if (val_c > PT.D_c_inv_sup) {
                        val_c = 2 * PT.D_c_inv_sup - val_c;
                    }
                }
                work.pP[i].D_c_inv = val_c;

                // --- 4. Réflexion pour Dtau_asigma ---
                double prop_t = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
                double val_t = work.pP[i].Dtau_asigma + prop_t;
                while (val_t < PT.Dtau_asigma_inf || val_t > PT.Dtau_asigma_sup) {
                    if (val_t < PT.Dtau_asigma_inf) {
                        val_t = 2 * PT.Dtau_asigma_inf - val_t;
                    } else if (val_t > PT.Dtau_asigma_sup) {
                        val_t = 2 * PT.Dtau_asigma_sup - val_t;
                    }
                }
                work.pP[i].Dtau_asigma = val_t;
            }


            //accept rate of the new model
            bool accept = false ;
            double Enew = -std::numeric_limits<double>::infinity() ; 

            
            Enew = compute_llk(t_list, data, G , work);
            if (Enew!=-std::numeric_limits<double>::infinity()){
                llk_computed+=1;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen2) );
                accept = (u <= alpha);
            }
            

            #pragma omp atomic
            newmodel+=accept;

            if (accept){
                std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                llk[ichain] = Enew;
                if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);            //plus tard save que sur certaines itérations
            }

        } //parallel end

        #pragma omp single
        { 
        for (int s = 0; s < PT.nchains - 1; s++) {
            int p = rand_chain(gen);
            int q = rand_chain(gen);
            if ((p == q) ||  (T[p] == T[q])) continue;

            // Formule théorique du Parallel Tempering
            double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
            double u_swap     = std::log(unif_dist(gen));

            if (u_swap <= alpha_swap) {
                swapnum+=1 ;
                if (p<PT.ncold || q<PT.ncold) swapcold+=1;
                std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                std::swap(llk[p], llk[q]);
            }
        }
        
    }}}
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nNew Models explored : "<<newmodel;
    std::cout<<"\nswaps : "<<swapnum;
    std::cout<<"\nswaps with coldchains : "<<swapcold;
    std::cout<<"\ncomputed llk : "<<llk_computed;

    return 0;
}



int parallel_tempering_new(const int maxint, const PT_param PT, const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, const
                    Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, const std::vector<sunrealtype>& t_list,  const int seed,const double sigma){

    int inbouds=0 ;
    int newmodel=0;
    int swapnum=0 ;
    int swapcold=0;
    int llk_computed=0 ;

    std::cout << "Taille de Param : " << sizeof(Param) << " octets\n";
    std::cout << "Taille exacte ecrite par etape : " << sizeof(double) + (NSubFaults * sizeof(Param)) << " octets\n";
    //check t_list size, data size
    if (t_list.size()!=static_cast<size_t>(data.cols())) std::cout<<"\nt_list and data size are not matching \n";

    //Create storage space for each CPU
    std::vector<std::unique_ptr<ThreadWorkspace>> workspaces;
    for (int t = 0; t < NCPU; ++t) {
        workspaces.push_back(std::make_unique<ThreadWorkspace>(t_list.size()));
    }

    //storage of the models of the cold chains
    std::vector<ColdChainSaver> savers;
    for (int i = 0; i < PT.ncold; ++i) {
        savers.emplace_back(i);
    }

    ColdChainSaver hotchainsaver(PT.nchains);
    
    //Temperatures of the chains
    std::mt19937 gen(seed); //generate random
    std::vector<double> T(PT.nchains); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);

    for (int i=0;i<PT.nchains;i++){
        if (i<PT.ncold) T[i]=1.0 ;
        else T[i]= std :: exp( i*1.0/(PT.ncold-1) * std::log(PT.T_max) ) ; 

    }

    //random integer 
    std::uniform_int_distribution<int> rand_chain(0, PT.nchains - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(PT.nchains);

    //Initial models of the chains    
    std::vector<Param> M;
    M.resize(PT.nchains*NSubFaults);

    
    #pragma omp parallel 
    {
        ThreadWorkspace& work = *workspaces[omp_get_thread_num()]; 
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0); ///////////////////////

        
        #pragma omp for schedule(dynamic)
        for (int j=0;j<PT.nchains;j++){
            bool reject = true;
            double llk_i ;
            //propose new model 
            while (reject)
            {
            for (int i = 0; i < NSubFaults; i++) {
                // Centre exact des bornes
                double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 2.0;
                double p2 = (PT.b_a_inf + PT.b_a_sup) / 2.0;
                double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 2.0;
                double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;

                // On applique une infime variation de 1% maximum propre à chaque chaîne (j) et sous-faille (i)
                // pour que les 10 chaînes ne partent pas exactement du même pixel
                double bruit = 0.99 + 0.02 * unif_dist_intern(gen2); // Nombre entre 0.99 et 1.01

                M[j * NSubFaults + i] = Param(p1 * bruit, p2 * bruit, p3 * bruit, p4 * bruit);
            }
            
            //test llk
            work.pP.assign(M.begin() + (j * NSubFaults), 
                           M.begin() + ((j + 1) * NSubFaults));
            llk_i = compute_llk2(t_list, data, G , work);
            if (llk_i > -9e5f ) {reject = false;}
            }
            llk[j]=llk_i;
            if (j<PT.ncold) savers[j].save_step(work.pP, llk_i);
            if (j==PT.nchains - 1) hotchainsaver.save_step(work.pP, llk_i);


        }
        


    //Parallel tempering

    for (int it = 0; it < maxint ; it ++ ){

        #pragma omp single
        {
        if (it % (maxint / 10) == 0) {
            std::cout << std::setw(3) << (100 * it / maxint) << "% done" << std::endl;
        }}

        #pragma omp for schedule(dynamic)
        for (int ichain = 0; ichain < PT.nchains ; ichain++){

            //copy model values
            work.pP.assign(M.begin() + (ichain * NSubFaults), M.begin() + ((ichain + 1) * NSubFaults));

            //New model proposal with miror boundaries

            
            double prop=0.0;

            for (int i=0; i<NSubFaults; i++){
                prop = unif_dist_plus(gen2) * (PT.k_a_sigma_sup   - PT.k_a_sigma_inf) * sigma;
                if (prop + work.pP[i].k_a_sigma > PT.k_a_sigma_sup)  work.pP[i].k_a_sigma = 2 * PT.k_a_sigma_sup - work.pP[i].k_a_sigma - prop ; 
                else if (prop + work.pP[i].k_a_sigma < PT.k_a_sigma_inf) work.pP[i].k_a_sigma = 2 * PT.k_a_sigma_inf - work.pP[i].k_a_sigma - prop ;
                else work.pP[i].k_a_sigma   += prop;

                prop = unif_dist_plus(gen2) * (PT.b_a_sup         - PT.b_a_inf) * sigma;
                if (prop + work.pP[i].b_a > PT.b_a_sup)  work.pP[i].b_a = 2 * PT.b_a_sup - work.pP[i].b_a - prop ; 
                else if (prop + work.pP[i].b_a < PT.b_a_inf) work.pP[i].b_a = 2 * PT.b_a_inf - work.pP[i].b_a - prop ;
                else work.pP[i].b_a   += prop;
                
                prop = unif_dist_plus(gen2) * (PT.D_c_inv_sup     - PT.D_c_inv_inf) * sigma;
                if (prop + work.pP[i].D_c_inv > PT.D_c_inv_sup)  work.pP[i].D_c_inv = 2 * PT.D_c_inv_sup - work.pP[i].D_c_inv - prop ; 
                else if (prop + work.pP[i].D_c_inv < PT.D_c_inv_inf) work.pP[i].D_c_inv = 2 * PT.D_c_inv_inf - work.pP[i].D_c_inv - prop ;
                else work.pP[i].D_c_inv   += prop;

                prop = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
                if (prop + work.pP[i].Dtau_asigma > PT.Dtau_asigma_sup)  work.pP[i].Dtau_asigma = 2 * PT.Dtau_asigma_sup - work.pP[i].Dtau_asigma - prop ; 
                else if (prop + work.pP[i].Dtau_asigma < PT.Dtau_asigma_inf) work.pP[i].Dtau_asigma = 2 * PT.Dtau_asigma_inf - work.pP[i].Dtau_asigma - prop ;
                else work.pP[i].Dtau_asigma   += prop;
            }
                

            // Remplacement propre du bloc de proposition et du check Out-of-Bounds
            for (int i = 0; i < NSubFaults; i++) {
                
                // --- 1. Réflexion pour k_a_sigma ---
                double prop_k = unif_dist_plus(gen2) * (PT.k_a_sigma_sup - PT.k_a_sigma_inf) * sigma;
                double val_k = work.pP[i].k_a_sigma + prop_k;
                while (val_k < PT.k_a_sigma_inf || val_k > PT.k_a_sigma_sup) {
                    if (val_k < PT.k_a_sigma_inf) {
                        val_k = 2 * PT.k_a_sigma_inf - val_k; // Rebond sur la borne inf
                    } else if (val_k > PT.k_a_sigma_sup) {
                        val_k = 2 * PT.k_a_sigma_sup - val_k; // Rebond sur la borne sup
                    }
                }
                work.pP[i].k_a_sigma = val_k;

                // --- 2. Réflexion pour b_a ---
                double prop_b = unif_dist_plus(gen2) * (PT.b_a_sup - PT.b_a_inf) * sigma;
                double val_b = work.pP[i].b_a + prop_b;
                while (val_b < PT.b_a_inf || val_b > PT.b_a_sup) {
                    if (val_b < PT.b_a_inf) {
                        val_b = 2 * PT.b_a_inf - val_b;
                    } else if (val_b > PT.b_a_sup) {
                        val_b = 2 * PT.b_a_sup - val_b;
                    }
                }
                work.pP[i].b_a = val_b;

                // --- 3. Réflexion pour D_c_inv ---
                double prop_c = unif_dist_plus(gen2) * (PT.D_c_inv_sup - PT.D_c_inv_inf) * sigma;
                double val_c = work.pP[i].D_c_inv + prop_c;
                while (val_c < PT.D_c_inv_inf || val_c > PT.D_c_inv_sup) {
                    if (val_c < PT.D_c_inv_inf) {
                        val_c = 2 * PT.D_c_inv_inf - val_c;
                    } else if (val_c > PT.D_c_inv_sup) {
                        val_c = 2 * PT.D_c_inv_sup - val_c;
                    }
                }
                work.pP[i].D_c_inv = val_c;

                // --- 4. Réflexion pour Dtau_asigma ---
                double prop_t = unif_dist_plus(gen2) * (PT.Dtau_asigma_sup - PT.Dtau_asigma_inf) * sigma;
                double val_t = work.pP[i].Dtau_asigma + prop_t;
                while (val_t < PT.Dtau_asigma_inf || val_t > PT.Dtau_asigma_sup) {
                    if (val_t < PT.Dtau_asigma_inf) {
                        val_t = 2 * PT.Dtau_asigma_inf - val_t;
                    } else if (val_t > PT.Dtau_asigma_sup) {
                        val_t = 2 * PT.Dtau_asigma_sup - val_t;
                    }
                }
                work.pP[i].Dtau_asigma = val_t;
            }


            //accept rate of the new model
            bool accept = false ;
            double Enew = -std::numeric_limits<double>::infinity() ; 

            
            Enew = compute_llk2(t_list, data, G , work);
            
            llk_computed+=1;
            double delta  = (Enew - llk[ichain])/T[ichain] ;
            double alpha  = std::min(0.0, delta);
            double u      = std::log(unif_dist_intern(gen2) );
            accept = (u <= alpha);
            
            

            #pragma omp atomic
            newmodel+=accept;

            if (accept){
                std::copy(work.pP.begin(), work.pP.end(), M.begin() + (ichain * NSubFaults));
                llk[ichain] = Enew;
                if (ichain<PT.ncold) savers[ichain].save_step(work.pP, Enew);            //plus tard save que sur certaines itérations
                if (ichain==PT.nchains-1) hotchainsaver.save_step(work.pP, Enew);
            }

        } //parallel end

        #pragma omp single
        { 
        for (int s = 0; s < PT.nchains - 1; s++) {
            int p = rand_chain(gen);
            int q = rand_chain(gen);
            if ((p == q) ||  (T[p] == T[q])) continue;

            // Formule théorique du Parallel Tempering
            double alpha_swap = std::min(0.0, (1.0/T[p] - 1.0/T[q]) * (llk[q] - llk[p]));
            double u_swap     = std::log(unif_dist(gen));

            if (u_swap <= alpha_swap) {
                swapnum+=1 ;
                if (p<PT.ncold || q<PT.ncold) swapcold+=1;
                std::swap_ranges(M.begin() + (p * NSubFaults), M.begin() + ((p + 1) * NSubFaults), M.begin() + (q * NSubFaults));
                std::swap(llk[p], llk[q]);
            }
        }
        
    }}}
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";
    std::cout<<"\nNew Models explored : "<<newmodel;
    std::cout<<"\nswaps : "<<swapnum;
    std::cout<<"\nswaps with coldchains : "<<swapcold;
    std::cout<<"\ncomputed llk : "<<llk_computed;
    std::cout<<"\nlast llk"<< llk[1];

    return 0;
}


*/

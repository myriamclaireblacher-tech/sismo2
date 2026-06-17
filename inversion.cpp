#include "inversion.hpp"

PT_param::PT_param(double k_a_sigma_inf_ent, double k_a_sigma_sup_ent, double b_a_inf_ent, double b_a_sup_ent, double D_c_inv_inf_ent, double D_c_inv_sup_ent, double Dtau_asigma_inf_ent, double Dtau_asigma_sup_ent, double Tmax_ent,
            int nchains_ent, int ncold_ent)
    :k_a_sigma_inf(k_a_sigma_inf_ent),k_a_sigma_sup(k_a_sigma_sup_ent),b_a_inf(b_a_inf_ent), b_a_sup (b_a_sup_ent), D_c_inv_inf(D_c_inv_inf_ent),
    D_c_inv_sup (D_c_inv_sup_ent), Dtau_asigma_inf(Dtau_asigma_inf_ent), Dtau_asigma_sup (Dtau_asigma_sup_ent), V0_inf(0.1*Vinf), V0_sup(1.1*Vinf), T_max(Tmax_ent),
    nchains (nchains_ent), ncold(ncold_ent)
    {}   

/*
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





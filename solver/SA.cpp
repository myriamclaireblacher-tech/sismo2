#include "inversion.hpp"
#include <algorithm>


void SA (int niter, int npert,  const std::vector<sunrealtype>& t_list,  const  Eigen::Matrix<double, 3*Nstations, Eigen::Dynamic>& data, 
    const Eigen::Matrix<double,3*Nstations,NSubFaults>& G, PT_param PT,
      int npert  = 20, double T = 0.01, double seed= 3456789){

    #pragma region initialisation
    double alp    = (0.000004/0.01) ** (1.0/niter) ;
    const double nparam = 5 ;

    //Workspace
    ThreadWorkspace work(t_list.size()) ;

    //modèle initial 

    double p1 = (PT.k_a_sigma_inf + PT.k_a_sigma_sup) / 10.0;
    double p2 = (PT.b_a_inf + PT.b_a_sup) / 10.0;
    double p3 = (PT.D_c_inv_inf + PT.D_c_inv_sup) / 20.0;
    double p4 = (PT.Dtau_asigma_inf + PT.Dtau_asigma_sup) / 2.0;
    double p5 = (PT.V0_inf + PT.V0_sup) /2.0 ;

    for (int j=0; j<NSubFaults ; j++) work.pP[j] = Param(p1, p2, p3 , p4 , p5);

    //best llk & params
    double llk=compute_llk2(t_list, data, G, work) ;
    double best_llk=llk ;

    std::vector<Param> best_param;
    best_param.resize(NSubFaults);
    best_param = work.pP ;
    int best_it=0;

    //subfault shuffle && random
    int i=1 ;
    double geometric_proba = 1.0 - std::exp(std::log(0.5)/1) ;

    std::mt19937 gen(seed+omp_get_thread_num());
    std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
    /////////////////////////////////
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);
    std::uniform_int_distribution<int> random_index(0, NSubFaults-1); 
    std::random_device rd;
    std::default_random_engine rng(rd());

    #pragma endregion

    for (int it=0; i<niter ; i++){

        double p = unif_dist(gen);

        //if (p<geometric_proba) {i=random_index(gen);}

        std::vector <int> num_sf (NSubFaults);
        for (int j=0; j< NSubFaults; j++) num_sf[j]=j;
        std::shuffle(num_sf.begin(), num_sf.end(), rng);
        std::vector <int> num_param = {0, 1, 2, 3 , 4};
        std::shuffle(num_param.begin(), num_param.end(), rng);

        for (int id_sf; id_sf<NSubFaults; id_sf++){
        
        double *pm1= &work.pP[id_sf].k_a_sigma ;
        double *pm2= &work.pP[id_sf].b_a ;
        double *pm3= &work.pP[id_sf].D_c_inv ;
        double *pm4= &work.pP[id_sf].Dtau_asigma ;
        double *pm5= &work.pP[id_sf].V0_ ;


        for (int j=0; j<nparam ; j++){

            double * old_param =-1;
            double ub = -1 ;
            double lb = - 1 ;

            #pragma region param_choice
            if (num_param[j]==0) 
                {old_param = &work.pP[num_sf[id_sf]].k_a_sigma;
                ub = PT.k_a_sigma_sup;
                lb = PT.k_a_sigma_inf ;}
            else if (num_param[j]==1) 
                {old_param = &work.pP[num_sf[id_sf]].b_a ;
                ub = PT.b_a_sup ;
                lb = PT.b_a_inf ;}
            else if (num_param[j] ==2) 
                {old_param = &work.pP[num_sf[id_sf]].D_c_inv ;
                ub = PT.D_c_inv_sup ;
                lb = PT.D_c_inv_inf ;}
            else if (num_param[j] == 3) 
                {old_param = &work.pP[num_sf[id_sf]].V0_ ;
                ub = PT.V0_sup ;
                lb = PT.V0_inf ;}
                    
            else if (num_param[j] == 4 ) 
                {old_param = & work.pP[num_sf[id_sf]].Dtau_asigma ;
                ub = PT.Dtau_asigma_sup ;
                lb =PT.Dtau_asigma_inf ;}

            #pragma endregion
            

            for (int ipert=0; i<npert; i++){
                double P_new ;
                do {double alpha = unif_dist(gen);
                // -> Perturbation distance 
                double y = T * std::tan(std::pi * (alpha - 0.5)) ;
                // direction
                double v = unif_dist_plus(gen) ;
                v= v/std::abs(v);
                //new model
                P_new =  old_param + y * v* (ub - lb) ; }

                while ((P_new> ub)|| (P_new<lb) )

                 &(*old_param) = P_new ;


                work.pP = Param(*pm1, *pm2, *pm3 , *pm4 , *pm5);
                llk =compute_llk3(id_sf, t_list, data, G, work, work.storage_matrix) ;
                if (llk>best_llk) best_llk=llk ;
                
            }
        }
    }
    T * alp ;

    std::cout << "\niteration = "<<it<< "/ cost = "<<llk<< " / cost min. = "<<best_llk ;
}




}
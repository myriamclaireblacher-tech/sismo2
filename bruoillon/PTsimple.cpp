#include <cmath>
#include <random>
#include <omp.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector> 
#include <fstream>
#include <algorithm>

double pi(double x){return std::log(std::pow(2,-x)+std::pow(2,x-100));}


void parallel_tempering(double T_max, int n_steps, int n_markow, int n_cold, double sigma= 0.005){

    //create  Temperture ladder
    int seed=42 ;
    std::mt19937 gen(seed); 
    std::vector<double> T(n_markow); 
    std::uniform_real_distribution<double> unif_dist(0.0 , 1.0);
    for (int i=0;i<n_markow;i++){
        if (i<n_cold) T[i]=1.0 ;
        else T[i]= std :: exp( i*1.0/(n_markow-1) * std::log(T_max) ) ; ///////////////////////:
        std::cout<< "\nT["<<i<<"] : "<<T[i];
    }
    

    //random interger
    std::uniform_int_distribution<int> rand_chain(0, n_markow - 1); 

    //likelihood data storage
    std::vector<double> llk ;
    llk.resize(n_markow);

    //model storage
    std::vector<double> X;
    X.resize(n_markow);

    //open storage file
    std::vector<std::ofstream> files;
    files.resize(n_cold) ;
    for (int j=0;j<n_cold; j++){
        files[j].open("test_chain_cold_" + std::to_string(j) + ".bin", std::ios::binary);
    }

    std::cout << "Taille exacte ecrite par etape : " << sizeof(double) ;

    #pragma omp parallel 
    {
        std::mt19937 gen2(seed+omp_get_thread_num());
        std::uniform_real_distribution<double> unif_dist_plus(-1.0 , 1.0);//////////////////////////////////////
        /////////////////////////////////
        std::uniform_real_distribution<double> unif_dist_intern(0.0 , 1.0);

        //initial models
        #pragma omp single
        {
        for (int j=0;j<n_markow;j++){
            X[j]= unif_dist_intern(gen2) * 100; //number between 0 and 1
            llk[j] = pi(X[j]) ;
            std::cout<<" \nX"<<j<<" : "<< X[j]; }

    
            //save initiial model
        for (int j=0; j<n_cold;j++){
                double val_pi = pi(X[j]);
                files[j].write(reinterpret_cast<const char*>(&val_pi), sizeof(double));
                files[j].write(reinterpret_cast<const char*>(&X[j]), sizeof(double));}}
                    


        //begin parallel tempering

        for (int it = 0; it < n_steps ; it ++ ){

            //update Markow chains
            #pragma omp for schedule(dynamic)
            for (int ichain = 0; ichain < n_markow ; ichain++){
                //new model proposal
                double X_new = X[ichain]+ unif_dist_plus(gen2) * sigma * 100 ;
                if (X_new>100) X_new = 200 - X_new ; //miror
                if (X_new<0) X_new = - X_new ; //miror 
                double Enew = pi(X_new);
                bool accept = false ;
                double delta  = (Enew - llk[ichain])/T[ichain] ;
                double alpha  = std::min(0.0, delta);
                double u      = std::log(unif_dist_intern(gen2) );
                accept = (u <= alpha);

                if (accept){
                    llk[ichain] = Enew;
                    X[ichain] = X_new ;
                    
                }
                if (ichain< n_cold) {
                        double val_pi = pi(X[ichain]);
                        
                        #pragma omp critical(file_write)
                        {
                            files[ichain].write(reinterpret_cast<const char*>(&val_pi), sizeof(double));
                            files[ichain].write(reinterpret_cast<const char*>(&X[ichain]), sizeof(double));
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
                        std::swap(X[p], X[q]);
                        std::swap(llk[p], llk[q]);
                    }
                }}


        }
        
    }
    
    std::cout<<"\n 100% done \n results ( log likelihood + models ) are saved in model_parameters.csv";

    //close files
    for (int j=0; j<n_cold; j++){
        if(files[j].is_open()) files[j].close();
    }

    }



int main(){
    parallel_tempering(1000.0, 10000, 2, 1, 0.01);
    return 0;
}




#ifndef VARIABLES_HPP
#define VARIABLES_HPP

constexpr double Vinf = 0.08*100/(365*24) ; //m/an

constexpr bool rapport_EDO = false ;


//constexpr int Nstations = 2 ;
constexpr int Nstations = 12 ;

//constexpr int NSubFaults = 450 ;

constexpr int n_m =8;
constexpr int m_n= 16;

constexpr int NSubFaults = n_m*m_n ;

constexpr int NCPU = 6 ;
                    //6

constexpr COCO= - (std::log(0.05)+0.5*std::log(2*std::acos(-1.0)))*300*2*NSubFaults ;

constexpr  double coco=1/NSubFaults ;         

#endif
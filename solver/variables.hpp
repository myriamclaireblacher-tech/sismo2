#ifndef VARIABLES_HPP
#define VARIABLES_HPP

constexpr double Vinf = 0.08*100/(365*24) ; //m/an

constexpr bool rapport_EDO = false ;


//constexpr int Nstations = 2 ;
constexpr int Nstations = 12 ;

//constexpr int NSubFaults = 450 ;
constexpr int NSubFaults = 4*4 ;

constexpr int NCPU = 6 ;
                    //6

constexpr  double coco=1/NSubFaults ;         

#endif
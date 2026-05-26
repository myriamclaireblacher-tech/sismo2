#ifndef READ_GREEN
#define READ_GREEN

#include <iostream>
#include <fstream>
#include <vector>


int main() {
    std::ifstream file("matrice_G.bin", std::ios::binary);
    if (!file.is_open()) return -1;

    int total_elements = 3 * 450 * 2; // nc * ns * nr = 2700 doubles
    std::vector<double> G_flat(total_elements);

    // Lecture directe du bloc mémoire d'un seul coup
    file.read(reinterpret_cast<char*>(G_flat.data()), total_elements * sizeof(double));
    file.close();

    // Accès aux données G(c, s, r) équivalent à l'indexation NumPy :
    // index = c * (ns * nr) + s * nr + r
    return 0;
}

#endif
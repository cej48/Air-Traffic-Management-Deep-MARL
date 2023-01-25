#include "atmosphere.h"

Atmosphere::Atmosphere(const int x_size, const int y_size, const int layers)
{
        weather = std::vector<std::vector<std::vector<Weather>>>(x_size, std::vector<std::vector<Weather>>
                    (y_size, std::vector<Weather>(layers)));
}
 
// TODO: Implment weather transiton dynamics.
void Atmosphere::step()
{

}

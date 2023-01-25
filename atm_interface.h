#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"

class ATMInterface{

    sf::RenderWindow* window;
    sf::View view;

    int width;
    int height;
    float zoom=1;
    public:
        ATMInterface(std::vector<Airport> airports);
        bool step(std::vector<Traffic> &traffic);
};
#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"

class ATMInterface{

    sf::RenderWindow* window;

    public:
        ATMInterface(std::vector<Airport> *airports);
        void step(std::vector<Traffic> *traffic);
};
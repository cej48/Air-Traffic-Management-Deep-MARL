#include "atm_interface.h"



ATMInterface::ATMInterface(std::vector<Airport> *airports)
{
    window = new sf::RenderWindow(sf::VideoMode(2000, 2000), "SFML works!");
}

void ATMInterface::step(std::vector<Traffic> *traffic)
{
}

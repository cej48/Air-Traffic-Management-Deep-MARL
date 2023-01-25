#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"

class ATMInterface{

    sf::RenderWindow* window;
    sf::View view;
    float view_width;
    float view_height;


    // Track mouse movement to drag display
    int mouse_previous_x=0;
    int mouse_previous_y=0;
    bool left_mouse_pressed = false;

    int width;
    int height;
    float zoom=1;

    bool input_handler();

    public:
        ATMInterface(std::vector<Airport> airports);
        bool step(std::vector<Traffic> &traffic);
};
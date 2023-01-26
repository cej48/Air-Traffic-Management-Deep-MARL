#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"

class ATMInterface{

    sf::Color radar_green = sf::Color(0,255,0);
    sf::Color radar_white = sf::Color(255,255,255);

    sf::RenderWindow* window;
    sf::View view;
    float view_width;
    float view_height;

    int scale_factor;

    std::vector<Airport*> *airports;
    std::vector<Traffic*> *traffic;
    sf::Font font;
    // Track mouse movement to drag display
    int mouse_previous_x=0;
    int mouse_previous_y=0;

    int width;
    int height;
    float zoom=1;

    bool input_handler();
    void draw_airports();
    void draw_traffic();

    public:
        ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor);
        bool step();
};
#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"
#include "rint.h"

class ATMInterface{

    sf::Color radar_green = sf::Color(0,255,0);
    sf::Color radar_white = sf::Color(255,255,255);
    sf::Color radar_yellow = sf::Color(255,255,0);
    sf::Color radar_red = sf::Color(255, 0, 0);
    sf::Color radar_grey = sf::Color(125,125,125);
    sf::Color transparent = sf::Color(0,0,0,0);
    sf::Color radar_blue = sf::Color(0,0,255);

    sf::Vector2f mouse_previous = sf::Vector2f(0,0);
    sf::Vector2f center_fix;

    sf::RenderWindow* window;
    sf::View scene_view;
    sf::View gui_view;

    float view_width;
    float view_height;

    int scale_factor;

    std::vector<Airport*> *airports;
    std::vector<Traffic*> *traffic;
    sf::Font font;

    RangeInt *acceleration;

    std::string selector_code;
    bool selector_bool=false;
    std::string input_text = "";
    // Track mouse movement to drag display

    int width;
    int height;
    float zoom=1;

    void selector(sf::Event &event);
    bool input_handler();
    void draw_airports();
    void draw_traffic();
    void draw_gui(std::string text);
    void action_parser(std::string text);

    std::string alt_to_string(double value);
    // std::string alt_display();

    public:
        ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor, RangeInt *acceleration);
        bool step();
};
#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"

class ATMInterface{

    sf::Color radar_green = sf::Color(0,255,0);
    sf::Color radar_white = sf::Color(255,255,255);
    sf::Color radar_yellow = sf::Color(255,255,0);

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

    // std::string alt_display();

    public:
        ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor);
        bool step();
};
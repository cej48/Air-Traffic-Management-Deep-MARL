#pragma once
#include <SFML/Graphics.hpp>
#include "airport.h"
#include "traffic.h"
#include "rint.h"
#include <eigen3/Eigen/Dense>

class ATMInterface{

    sf::Color radar_green = sf::Color(0,255,0);
    sf::Color radar_white = sf::Color(255,255,255);
    sf::Color radar_yellow = sf::Color(255,255,0);
    sf::Color radar_red = sf::Color(255, 0, 0);
    sf::Color radar_grey = sf::Color(125,125,125);
    sf::Color transparent = sf::Color(0,0,0,0);
    sf::Color radar_blue = sf::Color(0,0,255);

    sf::Color airport_0 = sf::Color(255,255,0);
    sf::Color airport_1 = sf::Color(0,255,255);
    sf::Color airport_2 = sf::Color(125,125,255);
    sf::Color airport_3 = sf::Color(255,0,0);
    sf::Color airport_4 = sf::Color(255,125, 255);
    sf::Color airport_5 = sf::Color(125,255,125);




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
    bool display_trails = false;
    bool display_mode = 0; // 0  is airport, 1 is altitude.
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
    void draw_trails();

    std::string alt_to_string(float value);

    bool* skip_render;

    // std::string alt_display();

    public:
        ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor, RangeInt *acceleration, bool *skip_render);

        bool step();
};
#include "atm_interface.h"
#include <SFML/Graphics.hpp>
#include <math.h>

ATMInterface::ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor)
{
    this->scale_factor = scale_factor;
    this->airports = airports;
    this->traffic = traffic;

    font.loadFromFile("arial.ttf");

    sf::Vector2 center(airports->at(0)->longitude*this->scale_factor, airports->at(0)->lattitude*-1*this->scale_factor);

    width = sf::VideoMode::getDesktopMode().width;
    height = sf::VideoMode::getDesktopMode().height;
    window = new sf::RenderWindow(sf::VideoMode(width/2, height/2), "Air traffic management sim");

    view = sf::View(sf::FloatRect(center.x, center.y, width/2, height/2));
    window->setView(view);
    window->setFramerateLimit(60);
}

float get_distance_vec2(sf::Vector2f a, sf::Vector2f b){
    return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y, 2));
}

void ATMInterface::selector(sf::Event &event){
    bool selected=false;
    for (int i = 0; i < traffic->size(); i++) 
        {
            Traffic* traffic_draw = traffic->at(i);
            float distance = get_distance_vec2(sf::Vector2f(traffic_draw->position[0]*scale_factor, traffic_draw->position[1]*-1*scale_factor),
                                            window->mapPixelToCoords(
                                            sf::Vector2i(event.mouseButton.x, event.mouseButton.y)));
            std::cout<<distance<<'\n';
            if (distance<400 && !selected){
                selected=true;
                this->selector_code = traffic_draw->callsign;
            }
        }
    if (!selected){
        this->selector_code = "";
    }
}

bool ATMInterface::input_handler()
{
    sf::Event event;
    bool update_view = false;

    int new_center_y = view.getCenter().y;
    int new_center_x = view.getCenter().x;


    while (window->pollEvent(event))
    {
        switch(event.type){
            case (sf::Event::Closed):
            {
                window->close();
                return 0;
            }
        
            case (sf::Event::Resized):
                {
                this->view_width = event.size.width;
                this->view_height = event.size.height;
                update_view = true;
                }
            case (sf::Event::KeyPressed):
            {
                switch (event.key.code){
                    case (sf::Keyboard::Escape):
                    {
                        window->close();
                        return 0;
                    }

                }
            } break;
            case (sf::Event::MouseWheelScrolled):
            {
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
                {
                    this->zoom+=event.mouseWheelScroll.delta;
                    update_view=true;
                }        
            } break;
            case(sf::Event::MouseButtonPressed):
            {
                switch (event.mouseButton.button)
                {
                    case (sf::Mouse::Left):
                    {   
                        mouse_previous = sf::Vector2f(event.mouseButton.x, event.mouseButton.y);
                        center_fix = view.getCenter();
                        selector(event);
                    }
                }
            } break;
            case(sf::Event::MouseButtonReleased):
            {

            }
            case (sf::Event::MouseMoved):
            {   

                if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){

                    new_center_x = center_fix.x + (mouse_previous.x-event.mouseMove.x)*this->zoom;
                    new_center_y = center_fix.y + (mouse_previous.y-event.mouseMove.y)*this->zoom;
                    update_view=true;
                }
            } break;
            default:
                break;
        }
    }
    if (update_view){
        view.setCenter(new_center_x, new_center_y);
        view.setSize(this->view_width*this->zoom, this->view_height*this->zoom);
        window->setView(view);
    }
    return 1;
}

void ATMInterface::draw_airports()
{
    for (int i = 0; i < airports->size(); i++) {
        Airport* airport = airports->at(i);
        sf::Text text;

        text.setFont(font);
        text.setString(airport->code);
        text.setCharacterSize(240);
        text.setFillColor(radar_green);
        text.setPosition(airport->longitude*this->scale_factor-500, airport->lattitude*-1*this->scale_factor-300);

        sf::RectangleShape rectangle(sf::Vector2f(100, 100));
        rectangle.setFillColor(radar_green);
        rectangle.setPosition(airport->longitude*this->scale_factor, airport->lattitude*-1*this->scale_factor);
        window->draw(text);
        window->draw(rectangle);    
    }
}

void ATMInterface::draw_traffic()
    {
    for (int i = 0; i < traffic->size(); i++) 
        {
        Traffic* traffic_draw = traffic->at(i);
        sf::Text text;

        sf::Color* colour = &this->radar_white;
        if (traffic_draw->callsign == this->selector_code){
            colour = &this->radar_yellow;
        }

        text.setFont(font);
        text.setString(traffic_draw->callsign);
        text.setCharacterSize(240);
        text.setFillColor(*colour);
        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor-300);
        window->draw(text);
        
        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+100);
        text.setCharacterSize(180);
        text.setString(traffic_draw->destination);
        window->draw(text);

        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+250);
        text.setString(std::to_string(int(traffic_draw->speed)));
        window->draw(text);

        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+400);
        text.setString(std::to_string(int(traffic_draw->position[2])));
        window->draw(text);

        sf::RectangleShape rectangle(sf::Vector2f(100, 100));
        rectangle.setFillColor(*colour);
        rectangle.setPosition(traffic_draw->position[0]*this->scale_factor, traffic_draw->position[1]*-1*this->scale_factor);

        window->draw(rectangle);    
    
        }
        
    }

    // std::string ATMInterface::alt_display(double altitude)
    // {
    //     std::string = 
    //     return std::string();
    // }

bool ATMInterface::step()
{
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    
    bool returnval = this->input_handler();

    window->clear();
    this->draw_airports();
    this->draw_traffic();
    window->display();
    return returnval;
}

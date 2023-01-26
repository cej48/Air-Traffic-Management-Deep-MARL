#include "atm_interface.h"
#include <SFML/Graphics.hpp>

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

bool ATMInterface::input_handler()
{
    sf::Event event;
    bool update_view = false;
    int mouse_delta_x=0;
    int mouse_delta_y=0;

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
            case (sf::Event::MouseMoved):
            {   
                if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
                    mouse_delta_x = (mouse_previous_x-event.mouseMove.x);
                    mouse_delta_y = (mouse_previous_y-event.mouseMove.y);
                    update_view=true;
                }
                mouse_previous_x = event.mouseMove.x;
                mouse_previous_y = event.mouseMove.y;
            } break;
            default:
                break;
        }
    }
    if (update_view){
        std::cout<<mouse_delta_x<<'\n';
        view.move(mouse_delta_x*this->zoom, mouse_delta_y*this->zoom);
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

        text.setFont(font);
        text.setString(traffic_draw->callsign);
        text.setCharacterSize(240);
        text.setFillColor(radar_white);
        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor-300);

        sf::RectangleShape rectangle(sf::Vector2f(100, 100));
        rectangle.setFillColor(radar_white);
        rectangle.setPosition(traffic_draw->position[0]*this->scale_factor, traffic_draw->position[1]*-1*this->scale_factor);
        window->draw(text);
        window->draw(rectangle);    
    
        }
        
    }

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

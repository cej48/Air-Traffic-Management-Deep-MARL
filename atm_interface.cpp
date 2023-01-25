#include "atm_interface.h"
#include <SFML/Graphics.hpp>

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
                    this->zoom+=event.mouseWheelScroll.delta*0.1;
                    update_view=true;
                }        
            } break;
            case (sf::Event::MouseButtonPressed):
            {
                switch (event.mouseButton.button){
                    case (sf::Mouse::Left):
                    {
                        mouse_previous_x = event.mouseButton.x;
                        mouse_previous_y = event.mouseButton.y;
                    }
                }
            } break;
            case (sf::Event::MouseMoved):
            {
                if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
                    mouse_delta_x = mouse_previous_x-event.mouseMove.x;
                    mouse_delta_y = mouse_previous_y-event.mouseMove.y;
                    mouse_previous_x = event.mouseMove.x;
                    mouse_previous_y = event.mouseMove.y;
                    update_view=true;
                }
            } break;
            default:
                break;
        }
    }
    if (update_view){
        view.move(mouse_delta_x*this->zoom, mouse_delta_y*this->zoom);
        view.setSize(this->view_width*this->zoom, this->view_height*this->zoom);
        window->setView(view);
    }
    return 1;
}

ATMInterface::ATMInterface(std::vector<Airport> airports)
{
    width = sf::VideoMode::getDesktopMode().width;
    height = sf::VideoMode::getDesktopMode().height;
    window = new sf::RenderWindow(sf::VideoMode(width/2, height/2), "Air traffic management sim");

    view = sf::View(sf::FloatRect(0, 0, width/2, height/2));
    window->setView(view);
}

bool ATMInterface::step(std::vector<Traffic> &traffic)
{
    // std::cout<<traffic[0].callsign<<'\n';
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);
    
    bool returnval = this->input_handler();

    window->clear();
    window->draw(shape);
    window->display();
    return returnval;

}

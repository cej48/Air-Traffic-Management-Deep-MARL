#include "atm_interface.h"
#include <SFML/Graphics.hpp>



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
    sf::Event event;

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
                // float height = view.getViewport
                // sf::Vector2f size = view.getSize();
                // float xscale = event.size.width/size.x;
                // float yscale = event.size.height/size.y;
                view.setSize(event.size.width*this->zoom, event.size.height*this->zoom);
                // view.getCenter()
                // view = sf::View(sf::FloatRect(0, 0, event.size.width*this->zoom, event.size.height*this->zoom));

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
            }
            case (sf::Event::MouseWheelScrolled):
            {
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
                {
                    // view.zoom(event.mouseWheelScroll.delta);
                    // view.zoom(1+event.mouseWheelScroll.delta*0.01);
                    this->zoom+=event.mouseWheelScroll.delta*0.01;
                    view.setSize(view.getSize()*float(event.mouseWheelScroll.delta*0.01+1));
                }        
            }
        }
    }
    // std::cout << sf::VideoMode::getDesktopMode().width << ", " << sf::VideoMode::getDesktopMode().height<<'\n';
    // view.zoom(this->zoom);
    window->setView(view);
    window->clear();
    window->draw(shape);
    window->display();
    return 1;

}

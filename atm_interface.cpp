#include "atm_interface.h"
#include <SFML/Graphics.hpp>
#include <math.h>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <math.h>

#define PI 3.14159265

ATMInterface::ATMInterface(std::vector<Airport*> *airports, std::vector<Traffic*> *traffic, int scale_factor, RangeInt *acceleration)
{
    this->scale_factor = scale_factor;
    this->airports = airports;
    this->traffic = traffic;

    this->acceleration = acceleration;

    this->view_width = width/2;
    this->view_height = height/2;

    font.loadFromFile("arial.ttf");

    sf::Vector2 center(airports->at(0)->position[0]*this->scale_factor, airports->at(0)->position[1]*-1*this->scale_factor);

    width = sf::VideoMode::getDesktopMode().width;
    height = sf::VideoMode::getDesktopMode().height;
    window = new sf::RenderWindow(sf::VideoMode(width/2, height/2), "Air traffic management sim");

    this->scene_view = sf::View(sf::FloatRect(center.x, center.y, width/2, height/2));
    gui_view = sf::View(sf::FloatRect(this->view_width/2, this->view_height/2, this->view_width, this->view_height));
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
            if (distance<400 && !selected){
                selected=true;
                this->selector_code = traffic_draw->callsign;
                this->selector_bool = true;
            }
        }
    if (!selected){
        this->selector_code = "";
        this->selector_bool=false;
    }
}

void ATMInterface::action_parser(std::string text)
{
    if (text.empty()){
        return;
    }
    std::vector<std::string> words;
    boost::split(words, text, boost::is_any_of(", "), boost::token_compress_on);
    for (int i = 0; i < traffic->size(); i++) 
    {
        // We havent selected anything
        if (!this->selector_bool){
            if (traffic->at(i)->callsign == words[0]){
                this->selector_code = words[0];
                this->selector_bool = true;
            };
        }
        
        // This isn't the correct aircraft.
        if (this->selector_code != traffic->at(i)->callsign){
            continue;
        }

        for (int w_index = 0; w_index<words.size(); w_index++){ 
            if (w_index>=words.size()){
                return;
                };
            if (words[w_index]=="hdg"){
                traffic->at(i)->target_heading = std::stof(words[w_index+1]);
            }            
            if (words[w_index]=="alt"){
                traffic->at(i)->target_altitude = std::stof(words[w_index+1]);
            }
            if (words[w_index]=="spd"){
                traffic->at(i)->target_speed = std::stof(words[w_index+1]);
            }
        
        }

    }
}

bool ATMInterface::input_handler()
{
    sf::Event event;
    bool update_view = false;

    int new_center_y = this->scene_view.getCenter().y;
    int new_center_x = this->scene_view.getCenter().x;


    while (window->pollEvent(event))
    {
        switch(event.type){
            case (sf::Event::Closed):{
                window->close();
                return 0;
            }
        
            case (sf::Event::Resized):{
                this->view_width = event.size.width;
                this->view_height = event.size.height;
                update_view = true;
                }break;
            
            case (sf::Event::TextEntered):{
                if (this->selector_bool){
                    switch (event.text.unicode){
                        case(8):{
                            if (!this->input_text.empty()){
                                if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)){
                                    this->input_text = "";
                                }else{
                                    this->input_text.pop_back();
                                }
                            }
                        }break;
                        case(27):{
                            this->selector_bool=false;
                            this->selector_code="";
                        case(13):{
                            action_parser(input_text);
                            this->input_text="";
                        }break;
                        }break;
                        default:{
                            this->input_text+=event.text.unicode;
                        }
                    }
                }
            } break;

            case (sf::Event::KeyPressed):{
                    switch (event.key.code){
                        case (sf::Keyboard::Escape):
                        {
                            if (!this->selector_bool){
                                window->close();
                                return 0;
                            }
                        }break;
                        case (sf::Keyboard::Right):
                        {
                            *this->acceleration-=1;
                        }break;
                        case (sf::Keyboard::Left):{
                            *this->acceleration+=1;
                        }break;
                    }
            } break;

            case (sf::Event::MouseWheelScrolled):{
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
                {
                    this->zoom+=event.mouseWheelScroll.delta;
                    update_view=true;
                }        
            } break;

            case(sf::Event::MouseButtonPressed):{
                switch (event.mouseButton.button)
                {
                    case (sf::Mouse::Left):
                    {   
                        mouse_previous = sf::Vector2f(event.mouseButton.x, event.mouseButton.y);
                        center_fix = this->scene_view.getCenter();
                        selector(event);
                    }
                }
            } break;

            case (sf::Event::MouseMoved):{   

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
        this->scene_view.setCenter(new_center_x, new_center_y);
        this->scene_view.setSize(this->view_width*this->zoom, this->view_height*this->zoom);
        this->gui_view.setSize(this->view_width, this->view_height);
        this->gui_view.setCenter(this->view_width/2,this->view_height/2);
    }
    return 1;
}

std::string ATMInterface::alt_to_string(double value)
{
    return (value<10000) ? std::to_string((int)value) : "FL"+std::to_string((int)value/100);
}

void ATMInterface::draw_gui(std::string in)
{   
    window->setView(this->gui_view);
    
    sf::Text text(in, font, this->view_width*0.1);
    text.setPosition(0, 0);
    text.setFillColor(radar_yellow);

    window->draw(text);
}

void ATMInterface::draw_airports()
{
    for (int i = 0; i < airports->size(); i++) {
        Airport* airport = airports->at(i);

        sf::RectangleShape rwhdg(sf::Vector2f(15.f, 0.0833*this->scale_factor));
        rwhdg.rotate(airport->runway_heading.value);
        rwhdg.setPosition((airport->position[0] + 0.0416*sin((airport->runway_heading.value)*PI/180))*this->scale_factor+50, 
                          (airport->position[1]*-1 - 0.0416*cos((airport->runway_heading.value)*PI/180))*this->scale_factor+50);
        rwhdg.setFillColor(this->radar_blue);
        window->draw(rwhdg);


        sf::Text text(airport->code, font, 240);
        text.setFillColor(radar_green);
        text.setPosition(airport->position[0]*this->scale_factor-500, airport->position[1]*-1*this->scale_factor-300);

        sf::RectangleShape rectangle(sf::Vector2f(100, 100));
        rectangle.setFillColor(radar_green);
        rectangle.setPosition(airport->position[0]*this->scale_factor, airport->position[1]*-1*this->scale_factor);
        window->draw(text);
        window->draw(rectangle);    
        
        float radius = 0.0833;
        for (int i=0; i<3; i++){
            sf::CircleShape shape(radius*this->scale_factor);
            shape.setPosition((airport->position[0]-radius)*this->scale_factor, (airport->position[1]*-1-radius)*this->scale_factor);
            shape.setOutlineThickness(10);
            shape.setFillColor(this->transparent);
            shape.setOutlineColor(this->radar_grey);
            window->draw(shape);
            radius+=0.0833;
        }
    }
}


void ATMInterface::draw_traffic()
    {
    for (int i = 0; i < traffic->size(); i++) 
        {
        Traffic* traffic_draw = traffic->at(i);

        sf::Color* colour = &this->radar_white;
        if (traffic_draw->infringement){
            colour = &this->radar_red;
        }
        if (traffic_draw->callsign == this->selector_code){
            colour = &this->radar_yellow;
        }

        sf::Text text(traffic_draw->callsign, font, 240);
        text.setFillColor(*colour);
        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor-300);
        window->draw(text);
        
        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+100);
        text.setCharacterSize(180);
        text.setString(traffic_draw->destination->code);
        window->draw(text);

        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+250);
        text.setString(std::to_string(int(traffic_draw->speed)));
        window->draw(text);

        text.setPosition(traffic_draw->position[0]*this->scale_factor-500, traffic_draw->position[1]*-1*this->scale_factor+400);
        text.setString(alt_to_string(traffic_draw->position[2]));
        window->draw(text);

        sf::RectangleShape rectangle(sf::Vector2f(100, 100));
        rectangle.setFillColor(*colour);
        rectangle.setPosition(traffic_draw->position[0]*this->scale_factor, traffic_draw->position[1]*-1*this->scale_factor);

        window->draw(rectangle);    
        }
        
    }
bool ATMInterface::step()
{
    window->setView(this->scene_view);

    window->clear();
    bool returnval = this->input_handler();
    this->draw_airports();
    this->draw_traffic();

    draw_gui(this->input_text);

    window->display();
    return returnval;
}

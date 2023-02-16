#include "atm_sim.h"
#include "traffic.h"
#include "utils.h"
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define PI 3.14159265


ATMSim::ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length)
{
    this->framerate = framerate;
    this->frame_length = frame_length;

    this->acceleration.value=10; // adjusted by interface if req'd

    std::ifstream file (environment_meta);
    json boundaries_json = json::parse(file);
    file.close();
    lattitude_min = boundaries_json["lattitude_min"];
    lattitude_max = boundaries_json["lattitude_max"];
    longitude_min = boundaries_json["longitude_min"];
    longitude_max = boundaries_json["longitude_max"];

    file = std::ifstream(airport_information);
    json airports_json = json::parse(file);
    file.close();
    for (json::iterator airport = airports_json.begin(); airport != airports_json.end(); ++airport){
        airports.push_back(new Airport(airport.value()["lattitude"],airport.value()["longitude"], 
                                   airport.value()["heading"], airport.key()));
    }

    environment = new Atmosphere(100,100,10);

    traffic.push_back(new Traffic(0.f, 51.f, 150.f, 0.f, 1000.f, airports[0], "BAW32P", this->frame_length));
    traffic.push_back(new Traffic(0.f, 52.f, 150.f, 0.f, 1000.f, airports[0], "BAW34P", this->frame_length));
    traffic.push_back(new Traffic(0.f, 52.f, 150.f, 0.f, 1000.f, airports[0], "BAW35P", this->frame_length));
    this->render = render;
    if (render){
        interface = new ATMInterface(&airports, &traffic, 10000, &acceleration);
    }

}


void ATMSim::detect_closure_infringement()
{
    int j = this->traffic.size()-1;
    for (int i=0; i<this->traffic.size();i++){
        this->traffic.at(i)->infringement=false;
    }
    for (int i = 0; i < this->traffic.size(); i++){
        for (int k = j; k>0; k--){
            float distance_xy = Utils::calculate_distance(this->traffic.at(i)->position, this->traffic.at(i+k)->position);
            float distance_z = abs(this->traffic.at(i)->position[2] - this->traffic.at(i+k)->position[2]);
            if (distance_xy<0.0833 && distance_z<900){ // 5 miles
                this->traffic.at(i)->infringement = true;
                this->traffic.at(i+k)->infringement = true;
            }        
        }
        j--;
    }
}

void ATMSim::detect_traffic_arrival()
{
    for (int i=0; i<this->traffic.size(); i++){
        Heading min(this->traffic[i]->heading-30);
        Heading max(this->traffic[i]->heading+30);

        // first check traffic is pointing in the correct direction.
        if (!this->traffic[i]->heading.in_range(60, this->traffic[i]->destination->runway_heading.value)){
            return;
        }

        if (Utils::calculate_distance(this->traffic[i]->position, this->traffic[i]->destination->position) < 0.0833 
            && abs(this->traffic[i]->position[2]- this->traffic[i]->destination->position[2]<2500)){
            this->traffic.erase(this->traffic.begin()+i);
        }
    }
}

// TODO: implement weather at position.
void ATMSim::spawn_aircraft()
{

    int value = 10+rand()%89;
    int destination = rand()%this->airports.size();
    int altitude = 20000+rand()%20000;

    float y_length = this->lattitude_max-lattitude_min;
    float x_length = this->longitude_max-longitude_min;

    float latti;
    float longi;
    switch(rand()%4){
    // switch(0){
        //TOP
        case(0):{
            latti = this->lattitude_max;
            longi = this->longitude_min + float((rand()%int(x_length*1e7))/1e7);
        } break;
        //LEFT
        case(1):{
            longi = this->longitude_min;
            latti = this->lattitude_min + float((rand()%int(y_length*1e7))/1e7);
        } break;
        //RIGHT
        case(2):{
            longi = this->longitude_max;
            latti = this->lattitude_min + float((rand()%int(y_length*1e7))/1e7);
        } break;
        //BOTTOM
        case(3):{
            latti = this->lattitude_min;
            longi = this->longitude_min + float((rand()%int(x_length*1e7))/1e7);
        } break;
    }

    traffic.push_back(new Traffic(longi, latti, 350.f, 0.f, altitude, airports[destination], "BAW"+std::to_string(value), this->frame_length));
}



bool ATMSim::step()
{   
    bool return_val = 1;
    Weather weather = Weather(1,2,3);
    count++;
    
    this->detect_closure_infringement();
    this->detect_traffic_arrival();
    
    if (this->render){
        return_val = interface->step();
    }
    if (count%acceleration.value){ // late guard
        std::cout<<acceleration.value<<'\n';
        return return_val;
    }
    for (auto item : traffic){
        item->step(&weather);
    }
    if (this->traffic.size()<this->max_traffic_count){
        this->spawn_aircraft();
    }
    return return_val;
}
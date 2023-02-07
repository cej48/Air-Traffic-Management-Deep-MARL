#include "atm_sim.h"
#include "traffic.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;


ATMSim::ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length)
{
    this->framerate = framerate;
    this->frame_length = frame_length;

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
        interface = new ATMInterface(&airports, &traffic, 10000);
    }

}


float ATMSim::calculate_distance(arma::vec3 a, arma::vec3 b){
    return sqrt(pow(a[0] - b[0],2)
            +   pow(a[1] - b[1],2));
}

void ATMSim::detect_closure_infringement()
{
    int j = this->traffic.size()-1;
    for (int i=0; i<this->traffic.size();i++){
        this->traffic.at(i)->infringement=false;
    }
    for (int i = 0; i < this->traffic.size(); i++){
        for (int k = j; k>0; k--){

            float distance_xy = this->calculate_distance(this->traffic.at(i)->position, this->traffic.at(i+k)->position);
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
        if (this->calculate_distance(this->traffic[i]->position, this->traffic[i]->destination->position) < 0.0833){
            this->traffic.erase(this->traffic.begin()+i);
        }
    }
    std::cout<<this->traffic.size()<<'\n';
}

// TODO: implement weather at position.
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
    if (count%60){
        return return_val;
    }
    for (auto item : traffic){
        item->step(&weather);
    }
    return return_val;
}
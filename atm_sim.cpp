#include "atm_sim.h"
#include "traffic.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

ATMSim::ATMSim(std::string environment_meta, std::string airport_information, bool render)
{
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

    traffic.push_back(new Traffic(0.f, 51.f, 150.f, 0.f, 1000.f, "EGKK", "TEST"));

    this->render = render;
    if (render){
        interface = new ATMInterface(&airports, &traffic, 10000);
    }

}


// TODO: implement weather at position.
bool ATMSim::step()
{   
    bool return_val = 1;
    Weather weather = Weather(1,2,3);
    
    if (this->render){
        return_val = interface->step();
    }
    for (auto item : traffic){
        item->step(&weather);
    }
    return return_val;
}
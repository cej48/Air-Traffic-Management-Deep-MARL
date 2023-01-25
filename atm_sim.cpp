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
        airports.push_back(Airport(airport.value()["lattitude"],airport.value()["longitude"], 
                                   airport.value()["heading"], airport.key()));
    }

    environment = new Atmosphere(100,100,10);
    traffic.push_back(Traffic(10.f, 10.f, 150.f, 0.f, 1000.f, "EGKK", "TEST"));

    this->render = render;
    if (render){
        interface = new ATMInterface(airports);
    }

}


// TODO: implement weather at position.
bool ATMSim::step()
{
    Weather weather = Weather(1,2,3);
    
    if (this->render){
        return interface->step(traffic);
    }

    for (Traffic item : traffic){
        item.step(&weather);
    }
    return 1;
}
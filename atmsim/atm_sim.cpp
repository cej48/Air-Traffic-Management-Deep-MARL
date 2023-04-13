#include "macro.h"
#include "atm_sim.h"
#include "traffic.h"
#include "utils.h"
#include <execution>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

ATMSim::ATMSim(ATMSim *other, bool render)
{
    this->render = render;
    Utils::deepcopy_pointy_vectors(&other->airports, &this->airports);
    Utils::deepcopy_pointy_vectors(&other->traffic, &this->traffic);
    this->lattitude_max = other->lattitude_max;
    this->lattitude_min = other->lattitude_min;
    this->longitude_min = other->longitude_min;
    this->longitude_max = other->longitude_max;
    this->max_traffic_count = other->max_traffic_count;
    this->count = other->count;
}

ATMSim::ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length)
{
    this->framerate = framerate;
    this->frame_length = frame_length;

    this->acceleration.value=10; // adjusted by interface if req'd

    std::ifstream file(environment_meta);
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
    this->render = render;
    if (render){
        interface = new ATMInterface(&airports, &traffic, 10000, &acceleration, &this->skip_render);
    }
    while (this->traffic.size()<this->max_traffic_count){
        this->spawn_aircraft();
    }

}



// void ATMSim::detect_nearest_traffic(Traffic* traff, float angle, float distance_xy){
    
//     for (long unsigned int j =0; j<traff->closest_distances.size();j++){
     
//         if (distance_xy < traff->closest_distances.at(j)){
//             traff->closest_distances.push_back(distance_xy);
//             traff->closest_angles.push_back(angle);
//         }
//     }
// }

void ATMSim::detect_closure_infringement()
{
    int j = this->traffic.size()-1;
    for (unsigned int i=0; i<this->traffic.size();i++){
        this->traffic.at(i)->infringement=false;
        this->traffic.at(i)->clear_nearest();
    }
    for (unsigned int i = 0; i < this->traffic.size(); i++){
        for (int k = j; k>0; k--){
            float distance_xy = Utils::calculate_distance(this->traffic.at(i)->position, this->traffic.at(i+k)->position);
            float distance_z = abs(this->traffic.at(i)->position[2] - this->traffic.at(i+k)->position[2]);
            
            this->traffic.at(i)->closest.push_back(std::make_pair(distance_xy, this->traffic.at(i+k)));
            this->traffic.at(i+k)->closest.push_back(std::make_pair(distance_xy, this->traffic.at(i)));
            // detect_nearest_traffic(this->traffic.at(i), 40.f, distance_xy);
            // detect_nearest_traffic(this->traffic.at(i+k), 40.f, distance_xy);
           
            if (distance_xy<MILE_5 && distance_z<900){ // 5 miles
                this->traffic.at(i)->infringement = true;
                this->traffic.at(i+k)->infringement = true;
            }        
        }
        j--;
    }


}

void ATMSim::detect_traffic_arrival()
{
    for (unsigned int i=0; i<this->traffic.size(); i++){

        // first check traffic is pointing in the correct direction.
        if (Utils::calculate_distance(this->traffic[i]->position, this->traffic[i]->destination->position) < 3 * MILE_5 
            && abs(this->traffic[i]->position[2]- this->traffic[i]->destination->position[2])<10000){
            this->traffic[i]->reward+=10;
        }

        if (this->traffic[i]->heading.in_range(60, this->traffic[i]->destination->runway_heading.value)){
            // std::cout<<this->traffic[i]->heading.value<<'\n';
            if (Utils::calculate_distance(this->traffic[i]->position, this->traffic[i]->destination->position) < MILE_5 
                && abs(this->traffic[i]->position[2]- this->traffic[i]->destination->position[2])<2500){
                this->traffic[i]->reward+=30;
            }
            if (Utils::calculate_distance(this->traffic[i]->position, this->traffic[i]->destination->position) < MILE_5/2 
                && abs(this->traffic[i]->position[2]- this->traffic[i]->destination->position[2])<1500){
                this->traffic[i]->reward+=300;
                std::cout<<"Arrived"<<'\n';
                this->traffic[i]->silent_terminated = true;

            }
        }
    }
}

// TODO: implement weather at position.

void ATMSim::verify_boundary_constraints(){
    for (unsigned int i=0; i<this->traffic.size(); i++){
        if (this->lattitude_min > this->traffic[i]->position[1] 
        || this->lattitude_max < this->traffic[i]->position[1]
        || this->longitude_min > this->traffic[i]->position[0]
        || this->longitude_max < this->traffic[i]->position[0]
        || std::isnan(this->traffic[i]->position[0])
        || std::isnan(this->traffic[i]->position[1]) //|| true 
        )
        {

            // this->traffic[i]->reward-=100;
            this->traffic[i]->terminated = true;
        }
    }
}

void ATMSim::spawn_aircraft()
{

    int value = 10+rand()%89;
    int destination = rand()%this->airports.size();
    int altitude = 10000+(rand()%10)*1000;

    float y_length = this->lattitude_max-lattitude_min;
    float x_length = this->longitude_max-longitude_min;

    float latti;
    float longi;
    switch(rand()%4){
    // switch(0){
        //TOP
        case(0):{
            latti = this->lattitude_max-0.1;
            longi = this->longitude_min + float((rand()%int(x_length*1e7))/1e7);
        } break;
        //LEFT
        case(1):{
            longi = this->longitude_min+0.1;
            latti = this->lattitude_min + float((rand()%int(y_length*1e7))/1e7);
        } break;
        //RIGHT
        case(2):{
            longi = this->longitude_max-0.1;
            latti = this->lattitude_min + float((rand()%int(y_length*1e7))/1e7);
        } break;
        //BOTTOM
        case(3):{
            latti = this->lattitude_min+0.1;
            longi = this->longitude_min + float((rand()%int(x_length*1e7))/1e7);
        } break;
    }
    if (std::isnan(latti) || std::isnan(longi)){
        return;
        }
    traffic.push_back(new Traffic(longi, latti, 350.f, 0.f, altitude, airports[destination], "BAW"+std::to_string(value), 
                                    this->frame_length, this->traffic_ID, this->count));
    traffic_ID++;

}

void ATMSim::copy_from_other(ATMSim *other)
{
    Utils::deepcopy_pointy_vectors<Traffic>(&other->traffic, &this->traffic);
    this->environment = other->environment;
}

void ATMSim::calculate_rewards(){
    float sum=0;
    for (auto &traff : this->traffic){
        if (traff->infringement){
            traff->reward-=30;
        }
        traff->reward-= 10*abs(Utils::calculate_distance(traff->position, traff->destination->position));

        sum+=traff->reward;
    }
}
bool ATMSim::step()
{   
    
    bool return_val = 1;
    Weather weather = Weather(1,2,3);
    count++;

    if (!skip_render || count%60==0){
        for (int i=0; i<acceleration.value;i++){
            return_val = interface->step();
        }
    }
    for (unsigned int i=0; i<this->traffic.size(); i++){
        this->traffic[i]->reward=0;

        if (this->traffic[i]->terminated || this->traffic[i]->silent_terminated || (this->count - this->traffic[i]->start_count) > this->traffic_timeout){
            delete traffic[i];
            this->traffic.erase(traffic.begin()+i);
            i--;
        }
    }
    std::for_each(
        std::execution::par,
        this->traffic.begin(),
        this->traffic.end(),
        [&weather](auto&& item)
        {
            item->step(&weather);
        }
    );
    this->detect_traffic_arrival();
    this->verify_boundary_constraints();
    this->detect_closure_infringement();
    for (long unsigned int i =0; i<this->traffic.size(); i++){
        std::sort(this->traffic.at(i)->closest.begin(),this->traffic.at(i)->closest.end());
    }

    while (this->traffic.size()<this->max_traffic_count){
        this->spawn_aircraft();
    }
    this->calculate_rewards();

    return return_val;
}

void ATMSim::reset()
{   
    // for (Traffic* i : this->traffic){
    //     delete i;
    // }
    // // std::cout<<this->traffic[0]->position<<'\n';
    // // std::cout<<"Fuckery"<<'\n';
    // // std::cout<<traffic.empty()<<'\n';
    // this->traffic.clear();


    while (this->traffic.size()<this->max_traffic_count){
        this->spawn_aircraft();
    }
}

// Action passer protocol:
// TARGET HEADING | TARGET_ALTITUDE | TARGET_SPEED 


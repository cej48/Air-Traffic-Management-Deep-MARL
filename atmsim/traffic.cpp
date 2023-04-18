#include "traffic.h"
#include "macro.h"
#include <math.h>
#include "utils.h"


Traffic::Traffic(float longitude, float lattitude,
                 float speed, float rate_of_climb,
                 float altitude, Airport *destination,
                 std::string callsign, int frame_length, int ID
                 , int start_count)
{
    this->position = Eigen::Vector3f(longitude, lattitude, altitude);
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
    this->frame_length = frame_length;

    this->heading = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));
    // std::cout<<'\n';
    // std::cout<<this->heading.value<<'\n';
    // std::cout<<this->target_heading<<'\n';
    // std::cout<<this->destination->code<<'\n';
    this->target_speed=speed;
    this->target_heading=heading.value;
    this->target_altitude=altitude;
    this->ID = ID;
    this->start_count = start_count;

}
Traffic::Traffic(Traffic* other){
        this->position=other->position;
        this->speed=other->speed;
        this->heading=other->heading;
        this->destination=other->destination; 
        this->callsign=other->callsign;

        this->infringement=other->infringement;

        this->rate_of_climb=other->rate_of_climb;
        this->rate_of_turn=other->rate_of_turn;
        this->rate_of_speed=other->rate_of_speed;

        this->target_heading=other->target_heading;
        this->target_speed=other->target_speed;
        this->target_altitude=other->target_altitude;

        this->destination_hdg=other->destination_hdg;

        this->reward=other->reward;

        this->frame_length=other->frame_length;
        this->ID = other->ID;
}

void Traffic::verify_constraints()
{
    if (this->speed<140){
        this->speed = 140;
        this->rate_of_speed = 0;
    }
    else if (this->speed>350){
        this->speed = 350;
        this->rate_of_speed = 0;
    }
    
    if (this->position[2]<250){
        this->position[2] = 250;
        this->rate_of_climb = 0;
    }
    else if (this->position[2]>41000){
        this->position[2] = 41000;
        this->rate_of_climb=0;
    }
}


void Traffic::adjust_params(){
    float det =this->heading.difference(target_heading);// (this->heading-this->target_heading);//*this->scale_speed;
    // std::cout<<"target_heading: "<<target_heading<<'\n';
    // std::cout<<"heading: "<<heading.value<<'\n';
    // std::cout<<"det: "<<det<<'\n';
    det = (det>=180 || det<=-180) ? -det : det;
    this->rate_of_turn = (abs(det)<3*this->scale_speed) ? det : ((det > 0 ) - (det < 0 )) * 3 * this->scale_speed;
    // std::cout<<(abs(det)<3*this->scale_speed)<<'\n';


    det = (this->position[2] - this->target_altitude);
    this->rate_of_climb = (abs(det) < 20*this->scale_speed) ? -det : ((det < 0 ) - (det > 0 )) *20*this->scale_speed;
    
    det = (this->speed - this->target_speed);
    this->rate_of_speed = (abs(det) < 1*this->scale_speed) ? -det : ((det < 0 ) - (det > 0 ))*this->scale_speed;

}


void Traffic::clear_nearest()
{
    this->closest.clear();
}

void Traffic::step(Weather *weather)
{
    // this->reward= 0;

    // std::cout<<"rot: "<<this->rate_of_turn<<'\n';
    // std::cout<<"heading: "<<this->heading.value<<"\n\n";
    // std::cout<<this->closest_distances[0]<<'\n';
    adjust_params();
    this->position[2]+=this->rate_of_climb; //*this->scale_speed;
    this->heading +=this->rate_of_turn; //*this->scale_speed;

    this->position[0]+=(sin(this->heading.value*(PI/180))*1/pow(60,3))*this->speed *this->scale_speed;
    this->position[1]+=(cos(this->heading.value*(PI/180))*1/pow(60,3))*this->speed *this->scale_speed;
    this->speed += this->rate_of_speed; //*this->scale_speed;
    verify_constraints();
    // get_closest_distances();

    if(std::isnan(this->position[0])){
        std::cout<<this->speed<<'\n';
        std::cout<<this->position<<'\n';
    }

    this->destination_hdg.value = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));
    // std::cout<<this->destination_hdg.value<<'\n';
}

// pos x, pos y, pos z, heading, 
std::vector<double> Traffic::get_observation()
{
    // std::cout<<"in"<<'\n';

    float base_size = 7;
    std::vector<double> ret(base_size+(N_closest*3));

    ret.at(0) = this->position[0]/2.5f;
    ret.at(1) = (this->position[1]-51.5)/1.5;
    ret.at(2) = this->position[2]/41000;
    ret.at(3) = (180-this->heading.value)/180;
    ret.at(4) = this->speed/350;
    ret.at(5) = this->destination->position[0]/2.5f;
    ret.at(6) = (this->destination->position[1]-50)/3;

    // Give current target states, used so that the network can consider the fact that 
    // a change has negative reward... So that the network doesn't issue unnecessary commands.
    // ret.at(7) = (180-this->target_heading)/180;
    // ret.at(8) = (this->target_altitude/41000);
    // ret.at(9) = (this->target_speed/350);

    for (long unsigned int i=0; i<N_closest; i++){
        // std::cout<<this->closest_distances.at(i)<<'\n';
        if (i>=this->closest.size()){
            ret.at((3*i)+base_size) = -1;
            ret.at((3*i)+base_size+1) = -1;
            ret.at((3*i)+base_size+2) = -1;
        }else{
            ret.at((3*i)+base_size) = this->closest.at(i).first/1.2;
            ret.at((3*i)+base_size+1) = (Utils::calculate_angle(this->position, this->closest.at(i).second->position))/(2*PI);
            ret.at((3*i)+base_size+2) = (this->closest.at(i).second->position[2]/41000);
        }
    }
    // std::cout<<"out"<<'\n';
    return ret;
}

void Traffic::set_actions(std::vector<float> actions)
{

    // std::cout<<actions[0]<<'\n';
    this->reward=0;

    // if ((actions[0] < target_heading -1 || actions[0] > target_heading+1)
    // ){
    //     // this->target_heading = actions[0];
    //     this->reward -= 1;
    // }
    // if ((actions[1] < target_altitude -1 || actions[1] > target_altitude+1)
    // ){
    //     // this->target_altitude = actions[1];
    //     this->reward -= 1;
    // }
    // if ((actions[2] < target_speed -1 || actions[2] > target_speed+1)
    // ){
    //     // this->target_speed = actions[2];
    //     this->reward -= 1;
    // }

    this->target_heading = actions[0];
    this->target_altitude = actions[1];
    this->target_speed = actions[2];
}
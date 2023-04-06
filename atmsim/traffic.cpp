#include "traffic.h"
#include "macro.h"
#include <math.h>
#include "utils.h"


Traffic::Traffic(float longitude, float lattitude,
                 float speed, float rate_of_climb,
                 float altitude, Airport *destination,
                 std::string callsign, int frame_length, int ID)
{
    this->position = Eigen::Vector3f(longitude, lattitude, altitude);
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
    this->frame_length = frame_length;

    this->heading = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));
    this->target_speed=speed;
    this->target_heading=heading.value;
    this->target_altitude=altitude;
    this->ID = ID;

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
    }
    else if (this->speed>350){
        this->speed = 350;
    }
    
    if (this->position[2]<250){
        this->position[2] = 250;
    }
    else if (this->position[2]>41000){
        this->position[2] = 41000;
    }
}


void Traffic::adjust_params(){
    float det = (this->heading)-(this->target_heading);

    det = (det>180|| det<-180) ? -det : det;
    this->rate_of_turn = (abs(det)<3) ? det : ((det < 0 ) - (det > 0 )) *3;

    det = this->position[2] - this->target_altitude;
    this->rate_of_climb = (abs(det) < 20) ? det : ((det < 0 ) - (det > 0 )) *20;
    
    det = this->speed - this->target_speed;
    this->rate_of_speed = (abs(det) < 1) ? det : this->rate_of_speed = ((det < 0 ) - (det > 0 ));

}

void Traffic::step(Weather *weather)
{
    this->reward= 0;

    this->position[2]+=this->rate_of_climb *this->scale_speed;
    this->heading +=this->rate_of_turn *this->scale_speed;

    this->position[0]+=(sin(this->heading.value*(PI/180))*1/pow(60,3))*this->speed *this->scale_speed;
    this->position[1]+=(cos(this->heading.value*(PI/180))*1/pow(60,3))*this->speed *this->scale_speed;
    this->speed += this->rate_of_speed *this->scale_speed;
    
    adjust_params();
    verify_constraints();
    if(std::isnan(this->position[0])){
        std::cout<<this->speed<<'\n';
        std::cout<<this->position<<'\n';
    }

    this->destination_hdg.value = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));
}

// pos x, pos y, pos z, heading, 
std::vector<float> Traffic::get_observation()
{

    return {this->position[0]/2.5f, (this->position[1]-51.5)/1.5,this->position[2]/41000, 
            (180-this->heading.value)/180, this->speed/350, this->destination->position[0]/2.5f,
            (this->destination->position[1]-50)/3};

}

void Traffic::set_actions(std::vector<float> actions)
{

    // std::cout<<actions[0]<<'\n';
    this->target_heading = actions[0];
    this->target_altitude = actions[1];
    this->target_speed = actions[2];
}
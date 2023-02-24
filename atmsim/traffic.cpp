#include "traffic.h"
#include "macro.h"
#include <math.h>
#include "utils.h"


Traffic::Traffic(float longitude, float lattitude,
                 const float speed, const float rate_of_climb,
                 const float altitude, Airport *destination,
                 std::string callsign, int frame_length)
{
    this->position = Eigen::Vector3d(longitude, lattitude, altitude);
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
    this->frame_length = frame_length;

    this->heading = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));

    this->target_speed=speed;
    this->target_heading=heading.value;
    this->target_altitude=altitude;
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

void Traffic::step(Weather* weather)
{
    this->reward= 0;

    this->position[2]+=this->rate_of_climb;
    this->heading +=this->rate_of_turn;

    this->position[0]+=sin(this->heading.value*(PI/180))*1/pow(60,3)*this->speed;
    this->position[1]+=cos(this->heading.value*(PI/180))*1/pow(60,3)*this->speed;
    this->speed += this->rate_of_speed;
    
    adjust_params();
    verify_constraints();

    this->destination_hdg.value = Utils::rad_to_deg(Utils::calculate_angle(this->position, this->destination->position));
}

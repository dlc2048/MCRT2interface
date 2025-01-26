
#define _USE_MATH_DEFINES

#include <cmath>
#include <random>

#include "generator.hpp"


std::uniform_real_distribution<> uniform(0., 1.);
std::normal_distribution<>       normal(0., 1.);


class SamplerOne : public SamplerBase {
public:


    SamplerOne(int pid, double weight) 
        : SamplerBase(pid, weight) {
    }


    void sample() {

        // position example

        static const double std_x = 4.;
        static const double std_y = 2.;
        static const double std_z = 6.;

        x = std_x * normal(rand_state);
        y = std_y * normal(rand_state);
        z = std_z * normal(rand_state);

        // direction example (isotropic)

        double cost = 1.0 - 2.0  * uniform(rand_state);
        double sint = std::sqrt((1.0 - cost) * (1.0 + cost));
        double phi  = 2.0 * M_PI * uniform(rand_state);
        double cosp = std::cos(phi);
        double sinp = std::sin(phi);

        u = sint * cosp;
        v = sint * sinp;
        w = cost;

        // energy and weight

        eke = 1.0;
        wee = 1.0;

    }


};


class SamplerTwo : public SamplerBase {
public:


    SamplerTwo(int pid, double weight)
        : SamplerBase(pid, weight) {
    }


    void sample() {
        x = 0.;
        y = 0.;
        z = 0.;

        // pencil beam example
        u = 0.;
        v = 0.;
        w = 1.;

        // energy and weight

        eke = 2.0;
        wee = 1.0;

    }


};


Generator::Generator() {

    SamplerOne* beam1 = new SamplerOne(0, 0.25);  // photon beam, weight 0.25
    SamplerTwo* beam2 = new SamplerTwo(0, 0.75);  // photon beam, weight 0.75

    this->registerBeam(beam1);
    this->registerBeam(beam2);

}

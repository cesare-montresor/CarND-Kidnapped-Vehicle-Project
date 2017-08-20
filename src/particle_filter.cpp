/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cmath>

#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine rand_eng;
  
  normal_distribution<> dist_x(x, std[0]);
  normal_distribution<> dist_y(y, std[1]);
  normal_distribution<> dist_theta(theta, std[2]);
  
  num_particles_ = 10;
  
  //cout<<" --------------- INIT ----------------\n";
  for (int i=0;i<num_particles_;i++){
    Particle p;
    p.id = i;
    p.x = dist_x(rand_eng);
    p.y = dist_y(rand_eng);
    p.theta = dist_theta(rand_eng);
    p.weight = 1.0;
    particles_.push_back(p);
    weights_.push_back(1.0);
    //cout<<"x: "<<p.x<<" | y: "<<p.y<<" | t: "<<p.theta<<"\n";
  }
  
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  default_random_engine rand_eng;
  
  normal_distribution<> dist_x(0, std_pos[0]);
  normal_distribution<> dist_y(0, std_pos[1]);
  normal_distribution<> dist_theta(0, std_pos[2]);
  
  double yaw_dt = yaw_rate*delta_t;
  double vel_yaw = velocity/yaw_rate;
  double vel_dt = velocity * delta_t;
  
  //cout<<" --------------- PREDICTION ----------------\n";
  for (int i=0;i<particles_.size();i++){
    Particle &p = particles_[i];
    if (fabs(yaw_rate) > 0.0001){
      p.x += (vel_yaw * ( sin(p.theta+yaw_dt) - sin(p.theta) )) + dist_x(rand_eng);
      p.y += (vel_yaw * ( cos(p.theta) - cos(p.theta+yaw_dt) )) + dist_y(rand_eng);
    }else{
      p.x += (vel_dt * cos(p.theta)) + dist_x(rand_eng);
      p.y += (vel_dt * sin(p.theta)) + dist_y(rand_eng);
    }
    p.theta += yaw_dt + dist_theta(rand_eng);
    //cout<<"x: "<<p.x<<" | y: "<<p.y<<" | t: "<<p.theta<<"\n";
  }
  

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for (int i = 0; i < observations.size(); i++)
  {
    LandmarkObs &o = observations[i];
    o.id = -1;
    double min_dist = MAXFLOAT;
    for (int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs p = predicted[j];
      double distance = dist(o.x, o.y, p.x, p.y);
    
      if ( distance < min_dist ) {
        min_dist = distance;
        o.id = j;
        //cout<<" | id: "<<j<<" min_dist: "<<min_dist;
      }
    }
    //cout<<"\n";
    //cout<<"px: "<<predicted[o.id].x<<" | ox: "<<o.x<<"\n";
    
    //cout<<"py: "<<predicted[o.id].y<<" | oy: "<<o.y<<"\n";
  }
  

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  //cout<<" --------------- UPDATE WEIGHTS ----------------\n";
  
  //Precalculation for gaussian error
  double std_xy = std_landmark[0]*std_landmark[1];
  double std_x2_2 = std_landmark[0]*std_landmark[0]*2;
  double std_y2_2 = std_landmark[1]*std_landmark[1]*2;
  double gauss_norm = (1.0 / (2.0*M_PI*std_xy));
  
  
  for (int i=0;i<particles_.size();i++){
    Particle &p = particles_[i];
    
    vector<LandmarkObs> nearby_landmarks;
    for (int j=0; j < map_landmarks.landmark_list.size(); j++)
    {
      auto &lm = map_landmarks.landmark_list[j];
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
      if (distance <= sensor_range){
        LandmarkObs near_lm;
        near_lm.id = lm.id_i;
        near_lm.x = lm.x_f;
        near_lm.y = lm.y_f;
        nearby_landmarks.push_back(near_lm);
      }
    }
    
    vector<LandmarkObs> obs_particle;
    double theta_cos = cos(p.theta);
    double theta_sin = sin(p.theta);
    for (int j=0; j < observations.size(); j++)
    {
      LandmarkObs o = observations[j];
      double ox = p.x + theta_cos * o.x + (- theta_sin * o.y);
      double oy = p.y + theta_sin * o.x + theta_cos * o.y;
      o.x = ox;
      o.y = oy;
      obs_particle.push_back(o);
    }
    
    dataAssociation(nearby_landmarks, obs_particle);
    
    // Calculate new weight for particle.
    p.weight = 1.0;
    for (int j=0; j < obs_particle.size(); j++){
      LandmarkObs o = obs_particle[j];
      if (o.id > 0)
      {
        auto &nearest = nearby_landmarks[o.id];
        double dx = o.x - nearest.x;
        double dy = o.y - nearest.y;
        
        
        double dx2 = dx*dx;
        double dy2 = dy*dy;
        
        double exponent = (dx2 / std_x2_2) + (dy2 / std_y2_2) ;
        double e = exp(-exponent);
        double multi_gauss = gauss_norm * e;
        p.weight *= multi_gauss;
        
        
        //p.weight += dx2 * dy2;
        //cout<<"exponent: "<<exponent<<" | e: "<<e<<"\n";
        //cout<<"Error: "<<"dx: "<<((dy2)/(2*std_y2))<<"  | dy: "<<((dy2)/(2*std_y2))<<"  | Err: "<<p.weight<<"\n";
        //cout<<"Error: "<<"dx2: "<<dx2<<"  | dy2: "<<dy2<<"  | Err: "<<err<<"\n";
        
      }
    }
    
    //p.weight = 1/p.weight;
    weights_[i] = p.weight;

  }
}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine rand_eng;
  
  std::discrete_distribution<> weights_distr(weights_.begin(), weights_.end());
  
  //cout<<" --------------- RESAMPLE ----------------\n";
  std::vector<Particle> newParticles;
  for (int i = 0; i < particles_.size(); i++){
    unsigned index = weights_distr(rand_eng);
    Particle p = particles_[index];
    //cout<<"index: "<<index<<" | x: "<<p.x<<" | y: "<<p.y<<" | t: "<<p.theta<<"\n";
    newParticles.push_back(p);
  }
  
  particles_ = newParticles;
  
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

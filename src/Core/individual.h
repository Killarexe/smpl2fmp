#pragma once

#include "AudioFile/AudioFile.h"
#include <cstdint>
#include <random>

class Individual {
  public:
    double fitness;

    virtual Individual crossover(Individual* parent, std::mt19937 rng);
    virtual void mutate(double mutationRate, std::mt19937 rng);
    virtual void randomize(std::mt19937 rng);

    virtual void printData();
    virtual void saveData();

    virtual AudioFile<double>::AudioBuffer synthetize(double frequency, double duration, uint32_t sampleRate);

  private:
    virtual double generate(double frequency, uint32_t sampleRate);
};

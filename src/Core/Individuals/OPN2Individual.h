#pragma once

#include "Core/individual.h"
#include <cstdint>
#include <random>

class OPN2Operator {
  public:
    uint8_t multiple;
    uint8_t totalLevel;
    uint8_t attackRate;
    uint8_t decayRate;
    uint8_t sustainLevel;

    OPN2Operator(){}
};

class OPN2Individual : public Individual {
  public:
    OPN2Individual();

    Individual crossover(Individual* parent, std::mt19937 rng) override;
    void mutate(double mutationRate, std::mt19937 rng) override;
    void randomize(std::mt19937 rng) override;

    void printData() override;
    void saveData() override;

    AudioFile<double>::AudioBuffer synthetize(double frequency, double duration, uint32_t sampleRate) override;

    uint8_t algorithm;
    uint8_t feedback;
    OPN2Operator operators[4];
};

#pragma once

#include "Core/individual.h"
#include "FMCores/Nuked-OPN2/ym3438.h"
#include <cstdint>
#include <memory>
#include <random>

constexpr size_t OPN2_OPERATOR_COUNT = 4;

class OPN2Operator {
  public:
    uint8_t multiple = 0;
    uint8_t totalLevel = 0;
    uint8_t attackRate = 0;
    uint8_t decayRate = 0;
    uint8_t sustainLevel = 0;
    uint8_t detune = 0;

    OPN2Operator() = default;
};

class OPN2Individual : public Individual {
  public:
    OPN2Individual() = default;

    std::unique_ptr<Individual> crossover(Individual* parent, std::mt19937& rng) override;
    void mutate(double mutationRate, std::mt19937& rng) override;
    void randomize(std::mt19937& rng) override;

    void printData() override;
    void saveData() override;

    void synthetize(double frequency, double duration, uint32_t sampleRate, AudioFile<double>::AudioBuffer& buffer) override;

    std::unique_ptr<Individual> clone() const override;

    uint8_t algorithm = 0;
    uint8_t feedback = 0;
    OPN2Operator operators[OPN2_OPERATOR_COUNT];
  private:
    void setPatch(ym3438_t* chip);
};

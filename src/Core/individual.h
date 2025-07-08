#pragma once

#include "AudioFile/AudioFile.h"
#include <cstdint>
#include <filesystem>
#include <random>

class Individual {
  public:
    double fitness;

    void saveAudio(std::filesystem::path path, double frequency, double duration, uint32_t sampleRate);

    virtual Individual crossover(Individual* parent, std::mt19937 rng) = 0;
    virtual void mutate(double mutationRate, std::mt19937 rng) = 0;
    virtual void randomize(std::mt19937 rng) = 0;

    virtual void printData() = 0;
    virtual void saveData() = 0;

    virtual AudioFile<double>::AudioBuffer synthetize(double frequency, double duration, uint32_t sampleRate) = 0;
};

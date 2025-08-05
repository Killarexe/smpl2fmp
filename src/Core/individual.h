#pragma once

#include "AudioFile/AudioFile.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>

class Individual {
  public:
    double fitness = 0;

    void saveAudio(std::filesystem::path path, double frequency, double duration, uint32_t sampleRate);

    virtual std::unique_ptr<Individual> crossover(Individual* parent, std::mt19937& rng) = 0;
    virtual void mutate(double mutationRate, std::mt19937& rng) = 0;
    virtual void randomize(std::mt19937& rng) = 0;

    virtual void printData() = 0;
    virtual void saveData() = 0;

    virtual void synthetize(double frequency, double duration, uint32_t sampleRate, AudioFile<double>::AudioBuffer& buffer) = 0;

    virtual std::unique_ptr<Individual> clone() const = 0;
    virtual ~Individual() = default;
};

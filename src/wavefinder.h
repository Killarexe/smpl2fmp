#pragma once

#include "AudioFile/AudioFile.h"
#include "Core/individual.h"
#include <cstdint>
#include <random>
#include <vector>

template <class T = Individual>
class Wavefinder {
  public:
    Wavefinder(AudioFile<double> targetSamples, uint16_t populationSize, uint16_t tournamentSize, double mutationRate);
    T run(uint16_t generations);
  
  private:
    uint16_t populationSize;
    uint16_t tournamentSize;
    double mutationRate;
    AudioFile<double>targetSamples;

    T tournamentSelect(const std::vector<T> population, std::mt19937 rng);
    double calculateFitness(AudioFile<double>::AudioBuffer samples, uint32_t sampleRate, int sampleSize, std::vector<std::pair<double, double>> targetSpectrum);
};

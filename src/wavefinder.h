#pragma once

#include "AudioFile/AudioFile.h"
#include "Core/individual.h"
#include <cstddef>
#include <fftw3.h>
#include <functional>
#include <memory>
#include <random>
#include <vector>

class Wavefinder {
  private:
    std::vector<std::unique_ptr<Individual>> population;
    std::mt19937 rng;
    
    size_t populationSize;
    size_t maxGenerations;
    size_t tournamentSize;
    double mutationRate;

    AudioFile<double>targetSamples;
    
    fftw_complex* targetFFT;
    fftw_complex* synthFFT;
    double* targetMagnitude;
    size_t fftSize;

    std::vector<double> monoBuffer;
    std::vector<double> targetEnergy;

    std::function<std::unique_ptr<Individual>()>individualFactory;

    void calculateFitness();
    double calculateSpectralDistanceFromTarget(const AudioFile<double>::AudioBuffer& buffer);

    void findTargetBaseFrequency();
    void calulcateTargetEnergy();
    
    void initFFTW();
    void freeFFTW();
    void computeFFT(const AudioFile<double>::AudioBuffer& buffer, fftw_complex* output);

    size_t tournamentSelection(size_t size);
    void crossoverPopulation();
    void mutatePopulation();
  
  public:
    double targetFrequency;

    Wavefinder(
      std::function<std::unique_ptr<Individual>()> factory,
      size_t popSize = 100,
      size_t genSize = 100,
      size_t turSize = 3,
      double mutRate = 0.1,
      unsigned int seed = std::random_device{}()
    ) : rng(seed),
      populationSize(popSize),
      maxGenerations(genSize),
      tournamentSize(turSize),
      mutationRate(mutRate),
      targetFFT(nullptr),
      synthFFT(nullptr),
      targetMagnitude(nullptr),
      fftSize(0),
      individualFactory(factory) {
    initializePopulation();
  }

  ~Wavefinder() {
    freeFFTW();
  }

  Individual* find(const AudioFile<double> targetSamples, double samplesFrequency);
  
  Individual* getBestIndividual();
  double getBestFitness();

  void initializePopulation();

  void setTarget(const AudioFile<double> newTarget);
};

template<typename IndividualType>
std::unique_ptr<Wavefinder> createWavefinder(size_t populationSize, size_t maxGenerations, size_t tournamentSize, double mutationRate, unsigned int seed) {
  auto factory = []() -> std::unique_ptr<Individual> {
    return std::make_unique<IndividualType>();
  };
  return std::make_unique<Wavefinder>(factory, populationSize, maxGenerations, tournamentSize, mutationRate, seed);
}

#pragma once

#include <AudioFile.h>
#include "Core/individual.h"
#include "fftprocessor.h"
#include "pgbar/pgbar.hpp"
#include <complex>
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
    
    std::vector<std::complex<double>> targetSpectrum;
    std::vector<double> targetMagnitude;
    std::vector<double> targetEnergy;

    std::vector<double> audioBuffer;

    double diversityThreshold;
    double fitnessImprovementThreshold;

    std::function<std::unique_ptr<Individual>()> individualFactory;

    void calculateFitness(FFTProcessor& fft);
    double calculateSpectralDistanceFromTarget(const double* samples, const std::vector<double>& magnitudes);

    void findTargetBaseFrequency();
    void calculateTargetEnergy();
    
    std::vector<size_t> getSortedIndiciesByFitness();
    double calculatePopulationDiversity();
    size_t rouletteWheelSelection();
    size_t tournamentSelection(size_t size);
    void crossoverPopulation(size_t generation);
    void mutatePopulation(size_t generationsWithoutImprovement);
    void insertRandomIndividuals();
    void simulatedAnnealingStep(size_t generation);

    pgbar::MultiBar<pgbar::ProgressBar<>, pgbar::ProgressBar<>> progressBars;
  
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
      diversityThreshold(0.1),
      fitnessImprovementThreshold(0.001),
      individualFactory(factory) {
    initializePopulation();
  }

  ~Wavefinder() {}

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

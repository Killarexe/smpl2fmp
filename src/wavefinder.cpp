#include "wavefinder.h"
#include "AudioFile/AudioFile.h"
#include "Core/Individuals/OPN2Individual.h"
#include "Core/individual.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <fftw3.h>
#include <vector>

template<>
Wavefinder<OPN2Individual>::Wavefinder(AudioFile<double> targetSamples, uint16_t populationSize, uint16_t tournamentSize, double mutationRate) {
  this->targetSamples = targetSamples;
  this->populationSize = populationSize;
  this->tournamentSize = tournamentSize;
  this->mutationRate = mutationRate;
}

std::pair<std::vector<std::pair<double, double>>, double> getSamplesSpectrumAndFrequency(AudioFile<double>::AudioBuffer samples, uint32_t sampleRate, int samplesSize) {
  const int spectrumSize = samplesSize / 2 + 1;

  double* targetSamplesInput = (double*)fftw_malloc(sizeof(double) * samplesSize);
  fftw_complex* targetSpectrumOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * spectrumSize);


  for (uint16_t i = 0; i < samplesSize; i++) {
  std::cout << "Here!" << std::endl;
    targetSamplesInput[i] = samples[0][i];
  }

  fftw_plan targetPlan = fftw_plan_dft_r2c_1d(samplesSize, targetSamplesInput, targetSpectrumOutput, FFTW_ESTIMATE);
  fftw_execute(targetPlan);

  std::vector<std::pair<double, double>> targetSpectrum;
  for (uint16_t i = 0; i < spectrumSize; i++) {
    double frequency = (double)i * sampleRate / samplesSize;
    double real = targetSpectrumOutput[i][0];
    double imagenery = targetSpectrumOutput[i][1];
    double magnitude = sqrt(real * real + imagenery * imagenery);
    targetSpectrum.push_back({frequency, magnitude});
  }

  int maxBin = 1;
  double maxMagnitude = targetSpectrum[1].second;
  for (uint16_t i = 2; i < spectrumSize; i++) {
    if (targetSpectrum[i].second > maxMagnitude) {
      maxMagnitude = targetSpectrum[i].second;
      maxBin = i;
    }
  }

  fftw_destroy_plan(targetPlan);
  fftw_free(targetSamplesInput);
  fftw_free(targetSpectrumOutput);
  return {targetSpectrum, targetSpectrum[maxBin].first};
}

template <>
double Wavefinder<OPN2Individual>::calculateFitness(AudioFile<double>::AudioBuffer buffer, uint32_t sampleRate, int samplesSize, std::vector<std::pair<double, double>> targetSpectrum) {
  std::pair<std::vector<std::pair<double, double>>, double> bufferSpectrum = getSamplesSpectrumAndFrequency(buffer, sampleRate, samplesSize);
  int numHarmonics = bufferSpectrum.first.size();
  if (numHarmonics > 16) {
    numHarmonics = 16;
  }
  double diffSum = 0.0;
  for (uint8_t i = 0; i < numHarmonics; i++) {
    std::pair<double, double> bufferComplex = bufferSpectrum.first[i];
    std::pair<double, double> targetComplex = targetSpectrum[i];
    double bufferMod = sqrt(bufferComplex.first * bufferComplex.first + bufferComplex.second * bufferComplex.second);
    double targetMod = sqrt(targetComplex.first * targetComplex.first + targetComplex.second * targetComplex.second);
    diffSum += abs(bufferMod - targetMod);
  }
  return (double)numHarmonics / (1.0 + diffSum);
}

template <>
OPN2Individual Wavefinder<OPN2Individual>::tournamentSelect(std::vector<OPN2Individual> population, std::mt19937 rng) {
  std::uniform_int_distribution<uint16_t> dis(0, population.size() - 1);
  OPN2Individual best = population[dis(rng)];
  for (uint16_t i = 0; i < this->tournamentSize; i++) {
    OPN2Individual contestant = population[dis(rng)];
    if (contestant.fitness > best.fitness) {
      best = contestant;
    }
  }
  return best;
}

bool fitnessComp(Individual a, Individual b) {
  return a.fitness > b.fitness;
}

template <>
OPN2Individual Wavefinder<OPN2Individual>::run(uint16_t generations) {
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<OPN2Individual> population(this->populationSize);
  for (Individual &individual : population) {
    individual.randomize(rng);
  }

  const double targetSamplesLength = this->targetSamples.getLengthInSeconds();
  const uint32_t targetSampleRate = this->targetSamples.getSampleRate();
  std::pair<std::vector<std::pair<double, double>>, double> targetSpectrum = getSamplesSpectrumAndFrequency(this->targetSamples.samples, targetSampleRate, targetSamplesLength);

  for (uint16_t generation = 0; generation < generations; generation++) {
    for (Individual &individual : population) {
      AudioFile<double>::AudioBuffer buffer = individual.synthetize(targetSpectrum.second, targetSamplesLength, targetSampleRate);
      individual.fitness = calculateFitness(buffer, targetSampleRate, targetSamplesLength, targetSpectrum.first);
    }
    std::sort(population.begin(), population.end(), fitnessComp);
    std::cout << "Generation n°" << generation << ": Best fitness: " << population[0].fitness << std::endl; 
    
    if (generation == generations - 1) {
      break;
    }

    std::vector<OPN2Individual> newPopulation;
    newPopulation[0] = population[0];
    for (uint16_t i = 0; i <= this->populationSize; i++) {
      OPN2Individual parent1 = tournamentSelect(population, rng);
      OPN2Individual parent2 = tournamentSelect(population, rng);
      Individual child = parent1.crossover(&parent2, rng);
      newPopulation.push_back(*dynamic_cast<OPN2Individual*>(&child));
    }

    population = newPopulation;
  }

  return population[0];
}

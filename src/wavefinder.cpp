#include "wavefinder.h"
#include "AudioFile/AudioFile.h"
#include "Core/individual.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <fftw3.h>
#include <vector>

template<>
Wavefinder<>::Wavefinder(AudioFile<double> targetSamples, uint16_t populationSize, uint16_t tournamentSize, double mutationRate) {
  this->targetSamples = targetSamples;
  this->populationSize = populationSize;
  this->tournamentSize = tournamentSize;
  this->mutationRate = mutationRate;
}

std::pair<std::vector<std::pair<double, double>>, double> getSamplesSpectrumAndFrequency(AudioFile<double> samples) {
  const uint32_t sampleRate = samples.getSampleRate();
  const int samplesSize = samples.getNumSamplesPerChannel();
  const int spectrumSize = samplesSize / 2 + 1;

  double* targetSamplesInput = (double*)fftw_malloc(sizeof(double) * samplesSize);
  fftw_complex* targetSpectrumOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * spectrumSize);

  for (uint16_t i = 0; i < samplesSize; i++) {
    targetSamplesInput[i] = samples.samples[0][i];
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

double Wavefinder<Individual>::calculateFitness(AudioFile<double>::AudioBuffer buffer, std::vector<std::pair<double, double>> targetSpectrum) {
  std::pair<std::vector<std::pair<double, double>>, double> bufferSpectrum = getSamplesSpectrumAndFrequency(buffer);
  int numHarmonics = std::max(bufferSpectrum.first.size(), 16);
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
Individual Wavefinder<Individual>::tournamentSelect(std::vector<Individual> population, std::mt19937 rng) {
  std::uniform_int_distribution<uint16_t> dis(0, population.size() - 1);
  Individual best = population[dis(rng)];
  for (uint16_t i = 0; i < this->tournamentSize; i++) {
    Individual contestant = population[dis(rng)];
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
Individual Wavefinder<Individual>::run(uint16_t generations) {
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<Individual> population(this->populationSize);
  for (Individual &individual : population) {
    individual.randomize(rng);
  }

  const double targetSamplesLength = this->targetSamples.getLengthInSeconds();
  const uint32_t targetSampleRate = this->targetSamples.getSampleRate();
  std::pair<std::vector<std::pair<double, double>>, double> targetSpectrum = getSamplesSpectrumAndFrequency(this->targetSamples);

  for (uint16_t generation = 0; generation < generations; generation++) {
    for (Individual &individual : population) {
      AudioFile<double>::AudioBuffer buffer = individual.synthetize(targetSpectrum.second, targetSamplesLength, targetSampleRate);
      individual.fitness = calculateFitness(buffer, targetSpectrum.first);
    }
    std::sort(population.begin(), population.end(), fitnessComp);
    
    if (generation == generations - 1) {
      break;
    }

    std::vector<Individual> newPopulation;
    newPopulation[0] = population[0];
    for (uint16_t i = 0; i <= this->populationSize; i++) {
      Individual parent1 = tournamentSelect(population, rng);
      Individual parent2 = tournamentSelect(population, rng);
      Individual child = parent1.crossover(&parent2, rng);
      newPopulation.push_back(child);
    }

    population = newPopulation;
  }

  return population[0];
}

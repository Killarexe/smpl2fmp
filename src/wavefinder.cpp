#include "wavefinder.h"
#include "AudioFile/AudioFile.h"
#include "Core/individual.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <iostream>
#include <memory>
#include <random>

void Wavefinder::initializePopulation() {
  population.clear();
  population.reserve(populationSize);

  for (size_t i = 0; i < populationSize; i++) {
    auto individual = individualFactory();
    individual->randomize(rng);
    population.push_back(std::move(individual));
  }
}

void Wavefinder::setTarget(const AudioFile<double> target) {
  targetSamples = target;
  initFFTW();
}

void Wavefinder::calculateFitness() {
  for (auto& individual : population) {
    AudioFile<double>::AudioBuffer synthesized = individual->synthetize(targetFrequency, targetSamples.getLengthInSeconds(), targetSamples.getSampleRate());
    individual->fitness = calculateSpectralDistanceFromTarget(synthesized);
  }
}

double Wavefinder::calculateSpectralDistanceFromTarget(const AudioFile<double>::AudioBuffer& buffer) {
  if (buffer.empty() || targetSamples.samples.empty()) {
    return 1.0;
  }

  computeFFT(buffer, synthFFT);

  double spectralDiff = 0.0;
  for (size_t i = 1; i < fftSize; i++) {
    double synthNorm = std::sqrt(synthFFT[i][0] * synthFFT[i][0] + synthFFT[i][1] * synthFFT[i][1]);
    spectralDiff += std::abs(synthNorm - targetMagnitude[i]);
  }

  /*double timeDiff = 0.0;
  for (size_t i = 0; i < buffer[0].size(); i++) {
    timeDiff += std::abs(buffer[0][i] - targetSamples.samples[0][i]);
  }

  const size_t windowSize = buffer[0].size() / 10;
  std::vector<double> synthEnergy, targetEnergy;
  for (size_t w = 0; w < 10; w++) {
    size_t start = w * windowSize;
    size_t end = std::min(start + windowSize, buffer[0].size());

    double synthRMS = 0.0;
    double targetRMS = 0.0;
    for (size_t i = start; i < end; i++) {
      synthRMS += buffer[0][i] * buffer[0][i];
      targetRMS += targetSamples.samples[0][i] * targetSamples.samples[0][i];
    }
    synthEnergy.push_back(std::sqrt(synthRMS / (end - start)));
    targetEnergy.push_back(std::sqrt(targetRMS / (end - start)));
  }

  double envelopeDiff = 0.0;
  for (size_t i = 0; i < 10; i++) {
    envelopeDiff += std::abs(synthEnergy[i] - targetEnergy[i]);
  }

  double difference = spectralDiff + timeDiff * 0.75 + envelopeDiff * 0.5;
  return (double)fftSize / (1.0 + difference);*/
  return (double)fftSize / (1.0 + spectralDiff);
}

void Wavefinder::computeFFT(const AudioFile<double>::AudioBuffer& buffer, fftw_complex* output) {
  if (buffer.empty() || !output) {
    return;
  }
  size_t bufferSize = buffer[0].size();

  std::vector<double> monoBuffer(bufferSize, 0.0);
  for (size_t channel = 0; channel < buffer.size(); channel++) {
    for (size_t i = 0; i < bufferSize && i < buffer[channel].size(); i++) {
      monoBuffer[i] += buffer[channel][i];
    }
  }

  if (buffer.size() > 1) {
    for (double& sample : monoBuffer) {
      sample /= buffer.size();
    }
  }

  fftw_plan plan = fftw_plan_dft_r2c_1d(bufferSize, monoBuffer.data(), output, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void Wavefinder::initFFTW() {
  if (targetSamples.samples.empty()) {
    return;
  }
  fftSize = targetSamples.samples[0].size() / 2 + 1;

  freeFFTW();

  targetFFT = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);
  synthFFT = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);
  targetMagnitude = (double*)malloc(sizeof(double) * fftSize);

  if (!targetFFT || !synthFFT || !targetMagnitude) {
    freeFFTW();
    return;
  }

  computeFFT(targetSamples.samples, targetFFT);
  findTargetBaseFrequency();
}

void Wavefinder::findTargetBaseFrequency() {
  double maxMagnitude = 0.0;
  size_t baseFrequencyIndex = 0;
  for (size_t i = 0; i < fftSize; i++) {
    double magnitude = std::sqrt(targetFFT[i][0] * targetFFT[i][0] + targetFFT[i][1] * targetFFT[i][1]);
    targetMagnitude[i] = magnitude;
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      baseFrequencyIndex = i;
    }
  }
  targetFrequency = (double)baseFrequencyIndex * ((double)targetSamples.getSampleRate() / (double)targetSamples.samples[0].size());
  std::cout << "Found base frequency: " << targetFrequency << std::endl;
}

void Wavefinder::freeFFTW() {
  if (targetFFT) {
    fftw_free(targetFFT);
    targetFFT = nullptr;
  }
  if (synthFFT) {
    fftw_free(synthFFT);
    synthFFT = nullptr;
  }
  if (targetMagnitude) {
    free(targetMagnitude);
    targetMagnitude = nullptr;
  }
}

size_t Wavefinder::tournamentSelection(size_t tournamentSize) {
  std::uniform_int_distribution<size_t> dis(0, population.size() - 1);
  size_t bestIndex = dis(rng);
  double bestFitness = population[bestIndex]->fitness;
  for (size_t i = 0; i < tournamentSize; i++) {
    size_t contestantIndex = dis(rng);
    double contestantFitness = population[contestantIndex]->fitness;
    if (bestFitness < contestantFitness) {
      bestFitness = contestantFitness;
      bestIndex = contestantIndex;
    }
  }
  return bestIndex;
}

void Wavefinder::crossoverPopulation() {
  std::uniform_int_distribution<size_t> parentDist(0, population.size() - 1);
  std::vector<std::unique_ptr<Individual>> newPopulation;
  newPopulation.reserve(populationSize);
  newPopulation.push_back(getBestIndividual()->clone());

  for (size_t i = 1; i < population.size(); i++) {
    size_t parent1Index = tournamentSelection(tournamentSize);
    size_t parent2Index = tournamentSelection(tournamentSize);

    auto child = population[parent1Index]->crossover(population[parent2Index].get(), rng);

    newPopulation.push_back(std::move(child));
  }

  population = std::move(newPopulation);
}

void Wavefinder::mutatePopulation() {
  for (size_t i = 1; i < population.size(); i++) {
    population[i]->mutate(mutationRate, rng);
  }
}

Individual* Wavefinder::getBestIndividual() {
  if (population.empty()) {
    return nullptr;
  }

  auto best = std::min_element(
    population.begin(), population.end(),
    [](const std::unique_ptr<Individual>& a, const std::unique_ptr<Individual>& b) {
      return a->fitness > b->fitness;
    }
  );

  return best->get();
}

double Wavefinder::getBestFitness() {
  Individual* best = getBestIndividual();
  return best ? best->fitness : 0.0;
}

Individual* Wavefinder::find(const AudioFile<double> targetSamples) {
  setTarget(targetSamples);

  for (size_t generation = 0; generation < maxGenerations; generation++) {
    calculateFitness();

    std::cout << "Generation n°" << generation << ": Best fitness => " << getBestFitness() << std::endl;

    crossoverPopulation();
    mutatePopulation();
  }

  calculateFitness();
  return getBestIndividual();
}

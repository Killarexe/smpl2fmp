#include "wavefinder.h"
#include "AudioFile/AudioFile.h"
#include "Core/individual.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

Individual* Wavefinder::find(const AudioFile<double> targetSamples) {
  setTarget(targetSamples);

  for (size_t generation = 0; generation < maxGenerations; generation++) {
    calculateFitness();

    if (generation % 10 == 0) {
      std::cout << "Generation n°" << generation << ": Best fitness => " << getBestFitness() << std::endl;
      getBestIndividual()->printData();
    }

    std::sort(population.begin(), population.end(),
      [](std::unique_ptr<Individual>& a, std::unique_ptr<Individual>& b) {
        return a->fitness < b->fitness;
      }
    );
    crossoverPopulation();
    mutatePopulation();
  }

  calculateFitness();
  return getBestIndividual();
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

  size_t minSize = std::min(buffer[0].size(), targetSamples.samples[0].size());
  if (minSize == 0) {
    return 1.0;
  }

  computeFFT(buffer, synthFFT);

  double diffSum = 0.0;
  for (size_t i = 0; i < fftSize; i++) {
    double synthNorm = std::sqrt(synthFFT[i][0] * synthFFT[i][0] + synthFFT[i][1] * synthFFT[i][1]);
    double targetNorm = std::sqrt(targetFFT[i][0] * targetFFT[i][0] + targetFFT[i][1] * targetFFT[i][1]);
    diffSum += std::abs(synthNorm - targetNorm);
  }
  return (double)fftSize / (1.0 + diffSum);
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

  for (size_t i = 0; i < bufferSize; i++) {
    double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (bufferSize - 1)));
    monoBuffer[i] *= window;
  }

  fftw_plan plan = fftw_plan_dft_r2c_1d(bufferSize, monoBuffer.data(), output, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void Wavefinder::initFFTW() {
  if (targetSamples.samples.empty()) {
    return;
  }
  fftSize = targetSamples.samples[0].size();

  freeFFTW();

  targetFFT = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);
  synthFFT = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftSize);

  if (!targetFFT || !synthFFT) {
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
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      baseFrequencyIndex = i;
    }
  }
  targetFrequency = (double)baseFrequencyIndex / (double)targetSamples.getSampleRate() / (double)targetSamples.samples[0].size();
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
  newPopulation.push_back(population[0]->clone());

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
      return a->fitness < b->fitness;
    }
  );

  return best->get();
}

double Wavefinder::getBestFitness() {
  Individual* best = getBestIndividual();
  return best ? best->fitness : 0.0;
}

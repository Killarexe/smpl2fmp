#include "wavefinder.h"
#include <AudioFile.h>
#include "Core/individual.h"
#include "fftprocessor.h"
#include "pgbar/pgbar.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <omp.h>

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
  findTargetBaseFrequency();
  calulcateTargetEnergy();
}

void Wavefinder::calculateFitness(FFTProcessor& fft) {
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < populationSize; i++) {
    population[i]->synthetize(targetFrequency, targetSamples.getLengthInSeconds(), targetSamples.getSampleRate(), fft.getInput(i));
    #pragma omp critical
    {
      progressBars.tick<1>();
    }
  }

  fft.execute();

  #pragma omp parallel for
  for (size_t i = 0; i < populationSize; i++) {
    population[i]->fitness = calculateSpectralDistanceFromTarget(fft.getInput(i), fft.getMagnitude(i));
  }
  
  progressBars.reset<1>();
}

double Wavefinder::calculateSpectralDistanceFromTarget(const double* samples, const std::vector<double>& magnitudes) {
  if (targetSamples.samples.empty()) {
    return 1.0;
  }

  const size_t sampleSize = targetSamples.samples[0].size(); 

  double spectralDiff = 0.0;
  size_t i;
  for (i = 1; i < magnitudes.size(); i++) {
    spectralDiff += std::abs(magnitudes[i] - targetMagnitude[i]);
  }
  double spectralScore = (double)magnitudes.size() / (1.0 + spectralDiff);

  /*double timeDiff = 0.0;
  for (size_t i = 0; i < sampleSize; i++) {
    timeDiff += std::abs(samples[i] - targetSamples.samples[0][i]);
  }
  timeDiff /= sampleSize;

  double envelopeDiff = 0.0;
  if (!targetEnergy.empty()) {
    const size_t windowSize = sampleSize / 10;
    for (size_t w = 0; w < 10; w++) {
      size_t start = w * windowSize;
      size_t end = std::min(start + windowSize, sampleSize);

      double synthRMS = 0.0;
      for (size_t i = start; i < end; i++) {
        synthRMS += samples[i] * samples[i];
      }

      double synthEnergy = std::sqrt(synthRMS / (end - start));
      envelopeDiff += std::abs(synthEnergy - targetEnergy[w]);
    }
    envelopeDiff /= targetEnergy.size();
  }*/

  return spectralScore;
}

void Wavefinder::calulcateTargetEnergy() {
  targetEnergy.clear();
  targetEnergy.reserve(10);
  const size_t windowSize = targetSamples.samples[0].size() / 10;
  for (size_t w = 0; w < 10; w++) {
    size_t start = w * windowSize;
    size_t end = std::min(start + windowSize, targetSamples.samples[0].size());

    double targetRMS = 0.0;
    for (size_t i = start; i < end; i++) {
      double sample = targetSamples.samples[0][i];
      targetRMS += sample * sample;
    }
    targetEnergy.push_back(std::sqrt(targetRMS / (end - start)));
  }
}

void Wavefinder::findTargetBaseFrequency() {
  size_t sampleSize = targetSamples.samples[0].size();
  size_t fftSize = sampleSize / 2 + 1;
  fftw_complex* result = fftw_alloc_complex(fftSize);
  fftw_plan plan = fftw_plan_dft_r2c_1d(sampleSize, targetSamples.samples[0].data(), result, FFTW_ESTIMATE);

  fftw_execute(plan);

  targetSpectrum.clear();
  targetSpectrum.reserve(fftSize);
  targetMagnitude.clear();
  targetMagnitude.reserve(fftSize);

  double maxMagnitude = 0.0;
  size_t baseFrequencyIndex = 0;

  for (size_t i = 0; i < fftSize; i++) {
    std::complex<double> complex = std::complex(result[i][0], result[i][1]);
    double magnitude = std::abs(complex);
    
    targetSpectrum.push_back(complex);
    targetMagnitude.push_back(magnitude);

    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      baseFrequencyIndex = i;
    }
  }

  fftw_destroy_plan(plan);
  fftw_free(result);

  targetFrequency = (double)baseFrequencyIndex * ((double)targetSamples.getSampleRate() / (double)targetSamples.samples[0].size());
  std::cout << "Found base frequency: " << targetFrequency << std::endl;
}

size_t Wavefinder::tournamentSelection(size_t tournamentSize) {
  size_t eliteCount = std::min(populationSize / 20, static_cast<size_t>(5));
  std::uniform_int_distribution<size_t> dis(0, population.size() - 1);
  size_t bestIndex = dis(rng);
  double bestFitness = population[bestIndex]->fitness;
  for (size_t i = eliteCount; i < tournamentSize; i++) {
    size_t contestantIndex = dis(rng);
    double contestantFitness = population[contestantIndex]->fitness;
    if (bestFitness < contestantFitness) {
      bestFitness = contestantFitness;
      bestIndex = contestantIndex;
    }
  }
  return bestIndex;
}

size_t Wavefinder::rouletteWheelSelection()  {
  double totalFitness = 0.0;
  double minFitness = std::numeric_limits<double>::max();
  for (const auto& individual : population) {
    minFitness = std::min(minFitness, individual->fitness);
  }
  for (const auto& individual : population) {
    totalFitness += individual->fitness - minFitness + 1.0;
  }
  if (totalFitness <= 0.0) {
    std::uniform_int_distribution<size_t> dis(0, populationSize - 1);
    return dis(rng);
  }
  std::uniform_real_distribution<double> dis(0.0, totalFitness);
  double pick = dis(rng);
  double current = 0.0;
  for (size_t i = 0; i < populationSize; i++) {
    current += population[i]->fitness - minFitness + 1.0;
    if (current >= pick) {
      return i;
    }
  }
  return populationSize - 1;
}

std::vector<size_t> Wavefinder::getSortedIndiciesByFitness() {
  std::vector<size_t> indicies(populationSize);
  std::iota(indicies.begin(), indicies.end(), 0);
  std::sort(indicies.begin(), indicies.end(),
    [this](size_t a, size_t b) {
      return population[a]->fitness > population[b]->fitness;
    });
  return indicies;
}

double Wavefinder::calculatePopulationDiversity() {
  if (populationSize < 2) {
    return 1.0;
  }
  
  double totalDistance = 0.0;
  size_t comparaisons = 0;
  size_t sampleSize = std::min(populationSize, static_cast<size_t>(20));
  std::uniform_int_distribution<size_t> dis(0, populationSize - 1);
  for (size_t i = 0; i < sampleSize; i++) {
    for (size_t j = i + 1; j < sampleSize; j++) {
      size_t index1 = dis(rng);
      size_t index2 = dis(rng);
      if (index1 != index2) {
        double dist = population[index1]->calculateDistance(population[index2].get());
        totalDistance += dist;
        comparaisons++;
      }
    }
  }
  return comparaisons > 0 ? totalDistance / comparaisons : 0.0;
}

void Wavefinder::insertRandomIndividuals() {
  std::vector<size_t> sortedIndicies = getSortedIndiciesByFitness();
  size_t replaceCount = populationSize / 10;
  for (size_t i = 0; i < replaceCount; i++) {
    size_t worstIndex = sortedIndicies[populationSize - 1 - i];
    std::unique_ptr<Individual> newIndividual = individualFactory();
    newIndividual->randomize(rng);
    population[worstIndex] = std::move(newIndividual);
  }
}

void Wavefinder::simulatedAnnealingStep(size_t generation) {
  const size_t sampleSize = targetSamples.samples[0].size();
  const size_t fftSize = sampleSize / 2 + 1;
  double* synthesized = (double*)malloc(sizeof(double) * sampleSize);
  fftw_complex* result = fftw_alloc_complex(fftSize);
  fftw_plan plan = fftw_plan_dft_r2c_1d(sampleSize, synthesized, result, FFTW_ESTIMATE);
  double temperature = 1.0 - (double)generation / maxGenerations;
  for (size_t i = 0; i < populationSize / 10; i++) {
    std::uniform_int_distribution<size_t> dis(0, populationSize);
    size_t index = dis(rng);

    std::unique_ptr<Individual> candidate = population[index]->clone();
    candidate->mutate(mutationRate * 3.0, rng);

    candidate->synthetize(targetFrequency, targetSamples.getLengthInSeconds(), targetSamples.getSampleRate(), synthesized);

    fftw_execute(plan);

    std::vector<double> magnitudes;
    magnitudes.reserve(fftSize);

    for (size_t j = 0; j < fftSize; j++) {
      magnitudes.push_back(std::sqrt(result[i][0] * result[i][0] + result[i][1] * result[i][1]));
    }
      
    candidate->fitness = calculateSpectralDistanceFromTarget(synthesized, magnitudes);

    double fitnessDiff = candidate->fitness - population[index]->fitness;
    if (fitnessDiff > 0 || (temperature > 0 && std::exp(fitnessDiff / temperature) > std::uniform_real_distribution<double>(0.0, 1.0)(rng))) {
      population[index] = std::move(candidate);
    }
  }

  fftw_free(result);
  free(synthesized);
  fftw_destroy_plan(plan);
}

void Wavefinder::crossoverPopulation(size_t generation) {
  std::uniform_int_distribution<size_t> parentDist(0, population.size() - 1);
  std::vector<std::unique_ptr<Individual>> newPopulation;
  newPopulation.reserve(populationSize);

  std::vector<size_t> sortedIndicies = getSortedIndiciesByFitness();
  size_t eliteCount = std::min(populationSize / 20, static_cast<size_t>(5));

  for (size_t i = 0; i < eliteCount; i++) {
    newPopulation.push_back(population[sortedIndicies[i]]->clone());
  }

  while (newPopulation.size() < populationSize) {
    size_t parent1Index, parent2Index;
    if (generation % 10 < 3) {
      parent1Index = rouletteWheelSelection();
      parent2Index = rouletteWheelSelection();
    } else {
      parent1Index = tournamentSelection(tournamentSize);
      parent2Index = tournamentSelection(tournamentSize);
    }
    std::unique_ptr<Individual> child = population[parent1Index]->crossover(population[parent2Index].get(), rng);
    newPopulation.push_back(std::move(child));
  }

  population = std::move(newPopulation);
}

void Wavefinder::mutatePopulation(size_t generationsWithoutImprovement) {
  double currentDiversity = calculatePopulationDiversity();
  double adaptiveMutationRate = mutationRate;
  if (currentDiversity < diversityThreshold) {
    adaptiveMutationRate *= 3.0;
  }

  size_t eliteCount = std::min(populationSize / 20, static_cast<size_t>(5));
  for (size_t i = eliteCount; i < population.size(); i++) {
    double individualMutationRate = adaptiveMutationRate;
    if (generationsWithoutImprovement > 20 && i % 10 == 0) {
      individualMutationRate *= 5.0;
    }
    population[i]->mutate(individualMutationRate, rng);
  }
}

Individual* Wavefinder::getBestIndividual() {
  if (population.empty()) {
    return nullptr;
  }

  auto best = std::max_element(
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

Individual* Wavefinder::find(const AudioFile<double> targetSamples, double samplesFrequency) {
  setTarget(targetSamples);
  if (samplesFrequency != 0.0) {
    targetFrequency = samplesFrequency;
    std::cout << "Target frequency forced to: " << samplesFrequency << "Hz!" << std::endl;
  }

  FFTProcessor fft(targetSamples.samples[0].size(), populationSize);

  size_t generationsWithoutImprovement = 0;
  double lastBestFintness = 0.0;

  progressBars.config<0>().tasks(maxGenerations);
  progressBars.config<0>().style(
    pgbar::config::Line::Sped |
    pgbar::config::Line::Elpsd |
    pgbar::config::Line::Cntdwn |
    pgbar::config::Line::Entire
  );
  progressBars.config<0>().speed_unit({"Gen/s"});
  progressBars.tick_to<0>(0);

  progressBars.config<1>().tasks(populationSize);
  progressBars.config<1>().style(
    pgbar::config::Line::Sped |
    pgbar::config::Line::Elpsd |
    pgbar::config::Line::Cntdwn |
    pgbar::config::Line::Entire
  );
  progressBars.config<1>().speed_unit({"Pop/s"});

  for (size_t generation = 0; generation < maxGenerations; generation++) {
    calculateFitness(fft);

    double currentBestFitness = getBestFitness();
    if (currentBestFitness > lastBestFintness + fitnessImprovementThreshold) {
      generationsWithoutImprovement = 0;
      lastBestFintness = currentBestFitness;
    } else {
      generationsWithoutImprovement++;
    }

    if (generationsWithoutImprovement > 30) {
      insertRandomIndividuals();
      generationsWithoutImprovement = 0;
    }

    crossoverPopulation(generation);
    mutatePopulation(generationsWithoutImprovement);
    //if (generation % 50 == 49) {
    //  simulatedAnnealingStep(generation);
    //}

    std::ostringstream stream;
    stream << "Best fitness: " << currentBestFitness;
    progressBars.config<0>().postfix(stream.str());
    progressBars.tick<0>();
  }

  calculateFitness(fft);
  return getBestIndividual();
}

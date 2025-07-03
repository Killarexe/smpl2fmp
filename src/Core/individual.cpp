#include "individual.h"
#include "AudioFile/AudioFile.h"
#include <cstdint>
#include <random>

Individual Individual::crossover(Individual* _parent, std::mt19937 _rng) {
  return Individual();
}

void Individual::mutate(double _mutationRate, std::mt19937 _rng) {
  return;
}

void Individual::randomize(std::mt19937 _rng) {
  return;
}

void Individual::printData() {
  return;
}

void Individual::saveData() {
  return;
}

AudioFile<double>::AudioBuffer Individual::synthetize(double _frequency, double _duration, uint32_t _sampleRate) {
  AudioFile<double>::AudioBuffer buffer;
  return buffer;
}

double Individual::generate(double _frequency, uint32_t _sampleRate) {
  return 0.0;
}

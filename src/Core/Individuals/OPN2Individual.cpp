#include "OPN2Individual.h"
#include "AudioFile/AudioFile.h"
#include <cstdint>
#include <random>

OPN2Individual::OPN2Individual() {}

Individual OPN2Individual::crossover(Individual* parent, std::mt19937 rng) {
  OPN2Individual* castParent = dynamic_cast<OPN2Individual*>(parent);

  OPN2Individual child = OPN2Individual();
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (uint8_t i = 0; i < 4; i++) {
    if (dis(rng) < 0.5) {
      child.operators[i] = this->operators[i];
    } else {
      child.operators[i] = castParent->operators[i];
    }
  }

  if (dis(rng) < 0.5) {
    child.algorithm = this->algorithm;
  } else {
    child.algorithm = castParent->algorithm;
  }

  if (dis(rng) < 0.5) {
    child.feedback = this->feedback;
  } else {
    child.feedback = castParent->feedback;
  }

  return child;
}

void OPN2Individual::mutate(double mutationRate, std::mt19937 rng) {
  std::uniform_real_distribution<double> mut_dis(0.0, 1.0);
  std::uniform_int_distribution<uint8_t> first_dis(0, 7);
  std::uniform_int_distribution<uint8_t> second_dis(0, 16);
  std::uniform_int_distribution<uint8_t> third_dis(0, 32);
  std::uniform_int_distribution<uint8_t> forth_dis(0, 128);

  for (uint8_t i = 0; i < 4; i++) {
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].multiple = second_dis(rng);
    }
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].totalLevel = forth_dis(rng);
    }
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].attackRate = third_dis(rng);
    }
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].decayRate = third_dis(rng);
    }
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].sustainLevel = second_dis(rng);
    }
  }
  if (mut_dis(rng) < mutationRate) {
    this->algorithm = first_dis(rng);
  }
  if (mut_dis(rng) < mutationRate) {
    this->feedback = first_dis(rng);
  }
}

void OPN2Individual::randomize(std::mt19937 rng) {
  mutate(1.1, rng);
}

void OPN2Individual::printData() {

}

void OPN2Individual::saveData() {
  //TODO: When working in export formats.
  return;
}

double OPN2Individual::generate(double frequency, uint32_t sampleRate) {
  return 0.0;
}

AudioFile<double>::AudioBuffer OPN2Individual::synthetize(double frequency, double duration, uint32_t sampleRate) {

}

#include "OPN2Individual.h"
#include "AudioFile/AudioFile.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include "Core/individual.h"
#include "FMCores/Nuked-OPN2/ym3438.h"

constexpr uint8_t CHANNEL = 0;
constexpr double CHIP_CLOCK = 7670454.0;
constexpr uint8_t OP_OFFSETS[4] = {0x30, 0x34, 0x38, 0x3C};

std::unique_ptr<Individual> OPN2Individual::crossover(Individual* parent, std::mt19937& rng) {
  const OPN2Individual* castParent = dynamic_cast<OPN2Individual*>(parent);

  OPN2Individual child;
  child.fitness = 0;
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
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

  return child.clone();
}

void OPN2Individual::mutate(double mutationRate, std::mt19937& rng) {
  std::uniform_real_distribution<double> mut_dis(0.0, 1.0);
  std::uniform_int_distribution<uint8_t> first_dis(0, 7);
  std::uniform_int_distribution<uint8_t> second_dis(0, 15);
  std::uniform_int_distribution<uint8_t> third_dis(0, 31);
  std::uniform_int_distribution<uint8_t> forth_dis(0, 127);

  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
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
    if (mut_dis(rng) < mutationRate) {
      this->operators[i].detune = first_dis(rng);
    }
  }
  if (mut_dis(rng) < mutationRate) {
    this->algorithm = first_dis(rng);
  }
  if (mut_dis(rng) < mutationRate) {
    this->feedback = first_dis(rng);
  }
}

void OPN2Individual::randomize(std::mt19937& rng) {
  mutate(1.1, rng);
}

void OPN2Individual::printData() {
  std::cout << "Algorithm: " << (int)algorithm << "\nFeedback: " << (int)feedback << std::endl;
  for (size_t index = 0; index < OPN2_OPERATOR_COUNT; index++) {
    OPN2Operator op = operators[index];
    std::cout << "Operator n°" << index + 1 << std::endl;
    std::cout << "Multiple: " << (int)op.multiple << std::endl;
    std::cout << "Total level: " << (int)op.totalLevel << std::endl;
    std::cout << "Attack rate: " << (int)op.attackRate << std::endl;
    std::cout << "Decay rate: " << (int)op.decayRate - 3 << std::endl;
    std::cout << "Detune: " << (int)op.detune << std::endl;
    std::cout << "Sustain level: " << (int)op.sustainLevel << "\n" <<std::endl;
  }
}

void OPN2Individual::saveData() {
  //TODO: When working in export formats.
  return;
}

static uint8_t getOctave(double frequency) {
  if (frequency < 82.41) {
    return 1;
  }
  if (frequency < 164.81) {
    return 2;
  }
  if (frequency < 329.63) {
    return 3;
  }
  if (frequency < 659.25) {
    return 4;
  }
  if (frequency < 1318.51) {
    return 5;
  }
  if (frequency < 2637.02) {
    return 6;
  }
  if (frequency < 5274.04) {
    return 7;
  }
  return 8;
}

static uint16_t getFNum(double frequency) {
  return (uint16_t)((144.0 * frequency * 1048526.0 / CHIP_CLOCK) / (double)(1 << (getOctave(frequency) - 1)));
}

static void writeChipRegister(ym3438_t* chip, uint8_t address, uint8_t data) {
  OPN2_WriteBuffered(chip, 0, address);
  OPN2_WriteBuffered(chip, 1, data);
}

void OPN2Individual::setPatch(ym3438_t* chip) {
  unsigned char algorithmFeedback = (feedback << 3) | algorithm;
  writeChipRegister(chip, 0xB0 + CHANNEL, algorithmFeedback);
  writeChipRegister(chip, 0xB4 + CHANNEL, 0xC0); // Stereo + AMS + FMS

  for (unsigned char op_index = 0; op_index < 4; op_index++) {
    OPN2Operator op = operators[op_index];
    unsigned char base_register = OP_OFFSETS[op_index] + CHANNEL;
    unsigned char detune_multiplier = (op.detune << 4) | op.multiple;
    unsigned char keyscale_attackrate = (0 << 6) | op.attackRate;
    unsigned char am_decay_rate = (0 << 7) | op.decayRate; // No AM
    unsigned char sustainlevel_releaserate = (op.sustainLevel << 4) | 0x0F;

    writeChipRegister(chip, base_register, detune_multiplier);
    writeChipRegister(chip, base_register + 0x10, op.totalLevel);
    writeChipRegister(chip, base_register + 0x20, keyscale_attackrate);
    writeChipRegister(chip, base_register + 0x30, am_decay_rate);
    writeChipRegister(chip, base_register + 0x40, 0x00); // No sustain rate
    writeChipRegister(chip, base_register + 0x50, sustainlevel_releaserate);
    writeChipRegister(chip, base_register + 0x60, 0x00); // No SSG-EG
  }
}

AudioFile<double>::AudioBuffer OPN2Individual::synthetize(double frequency, double duration, uint32_t sampleRate) {
  uint32_t totalSamples = (uint32_t)(duration * sampleRate);

  AudioFile<double>::AudioBuffer result;
  result.resize(2);
  result[0].resize(totalSamples);
  result[1].resize(totalSamples);

  ym3438_t chip;
  OPN2_Reset(&chip, sampleRate, CHIP_CLOCK);
  OPN2_SetOptions(ym3438_type_ym2612);

  // Disable DAC and disable LFO
  writeChipRegister(&chip, 0x2A, 0);
  writeChipRegister(&chip, 0x2B, 0);
  writeChipRegister(&chip, 0x22, 0);

  // Disable Timer A & B and put on normal mode
  writeChipRegister(&chip, 0x24, 0);
  writeChipRegister(&chip, 0x25, 0);
  writeChipRegister(&chip, 0x26, 0);
  writeChipRegister(&chip, 0x27, 0);

  setPatch(&chip);

  uint16_t fNum = getFNum(frequency);
  uint8_t octave = getOctave(frequency);

  writeChipRegister(&chip, 0xA4 + CHANNEL, ((octave << 3) | (fNum >> 8)) & 0x3F);
  writeChipRegister(&chip, 0xA0 + CHANNEL, fNum & 0xFF);

  for (unsigned char channel = 0; channel < 6; channel++) {
    writeChipRegister(&chip, 0x28, 0x00 | channel);
  }

  int32_t* leftBuffer = (int32_t*)malloc(sizeof(int32_t) * totalSamples);
  int32_t* rightBuffer = (int32_t*)malloc(sizeof(int32_t) * totalSamples);
  int32_t* buffer[2] = {leftBuffer, rightBuffer};

  writeChipRegister(&chip, 0x28, 0xF0 | CHANNEL);

  OPN2_GenerateStream(&chip, buffer, totalSamples);

  writeChipRegister(&chip, 0x28, 0x00 | CHANNEL);

  for (size_t i = 0; i < totalSamples; i++) {
    result[0][i] = (double)buffer[0][i] / 16384.0; 
    result[1][i] = (double)buffer[1][i] / 16384.0; 
  }

  free(leftBuffer);
  free(rightBuffer);

  return result;
}

std::unique_ptr<Individual> OPN2Individual::clone() const {
  return std::make_unique<OPN2Individual>(*this);
}

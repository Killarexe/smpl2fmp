#include "OPN2Individual.h"
#include "AudioFile/AudioFile.h"
#include <cmath>
#include <cstdint>
#include <random>
#include "FMCores/Nuked-OPN2/ym3438.h"

constexpr uint8_t CHANNEL = 0;
constexpr double CHIP_CLOCK = 7670454.0;
constexpr double CHIP_RATE = CHIP_CLOCK / 144.0;
const double FNUM_BASE = pow(2.0, 20.0) / CHIP_RATE;
constexpr uint8_t OP_OFFSETS[4] = {0x30, 0x34, 0x38, 0x3C};

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

static uint16_t getFNum(double frequency) {
  return (uint16_t)(frequency * FNUM_BASE);
}

static uint8_t getOctave(double frequency) {
  if (frequency < 82.41) {
    return 0;
  }
  if (frequency < 164.81) {
    return 1;
  }
  if (frequency < 329.63) {
    return 2;
  }
  if (frequency < 659.25) {
    return 3;
  }
  if (frequency < 1318.51) {
    return 4;
  }
  if (frequency < 2637.02) {
    return 5;
  }
  if (frequency < 5274.04) {
    return 6;
  }
  return 7;
}

static void setPatch(ym3438_t* chip, OPN2Individual* patch) {
  uint8_t algorithmFeedback = (patch->feedback << 3) | patch->algorithm;
  OPN2_Write(chip, 0, 0xB0 + CHANNEL);
  OPN2_Write(chip, 1, algorithmFeedback);

  for (uint8_t index = 0; index < 4; index++) {
    uint8_t baseRegister = OP_OFFSETS[index] + CHANNEL;
    uint8_t detuneMultiplier = (0 << 4) | patch->operators[index].multiple; //TODO: Add detune
    uint8_t keyScaleAttackRate = (0 << 6) | patch->operators[index].attackRate; //TODO: Add env scale
    uint8_t AMDecayRate = (0 << 7) | patch->operators[index].decayRate;
    uint8_t sustainLevelReleaseRate = (patch->operators[index].sustainLevel << 4) | 0;
    
    OPN2_Write(chip, 0, baseRegister);
    OPN2_Write(chip, 1, detuneMultiplier);

    OPN2_Write(chip, 0, baseRegister + 0x10);
    OPN2_Write(chip, 1, patch->operators[index].totalLevel);

    OPN2_Write(chip, 0, baseRegister + 0x20);
    OPN2_Write(chip, 1, keyScaleAttackRate);

    OPN2_Write(chip, 0, baseRegister + 0x30);
    OPN2_Write(chip, 1, AMDecayRate);

    OPN2_Write(chip, 0, baseRegister + 0x40);
    OPN2_Write(chip, 1, 0); //TODO: Sustain rate

    OPN2_Write(chip, 0, baseRegister + 0x50);
    OPN2_Write(chip, 1, sustainLevelReleaseRate);

    OPN2_Write(chip, 0, baseRegister + 0x60);
    OPN2_Write(chip, 1, 0); //TODO: SSS-EG
  }
}

AudioFile<double>::AudioBuffer OPN2Individual::synthetize(double frequency, double duration, uint32_t sampleRate) {
  uint32_t totalSamples = (uint32_t)(duration * sampleRate);

  AudioFile<double>::AudioBuffer result;
  result.resize(2);
  result[0].resize(totalSamples);
  result[1].resize(totalSamples);

  ym3438_t chip;
  OPN2_Reset(&chip);
  OPN2_SetChipType(1);

  setPatch(&chip, this);

  uint16_t fNum = getFNum(frequency);
  uint8_t octave = getOctave(frequency);

  OPN2_Write(&chip, 0, 0xA0 + CHANNEL);
  OPN2_Write(&chip, 1, fNum & 0xFF);
  OPN2_Write(&chip, 0, 0xA4 + CHANNEL);
  OPN2_Write(&chip, 1, ((octave << 3) | (fNum >> 8)) & 0xFF);

  OPN2_Write(&chip, 0, 0x28);
  OPN2_Write(&chip, 0, 0xF0 | CHANNEL);


  const double samplesPerClock = sampleRate / CHIP_RATE;
  double clockAccumulator = 0.0;
  int16_t chipBuffer[2];
  uint32_t sampleIndex = 0;
  while (sampleIndex < totalSamples) {
    OPN2_Clock(&chip, chipBuffer);
    clockAccumulator += samplesPerClock;
    while (clockAccumulator >= 1.0 && sampleIndex < totalSamples) {
      result[0][sampleIndex] = chipBuffer[0];
      result[1][sampleIndex] = chipBuffer[1];
      sampleIndex++;
      clockAccumulator = -1.0;
    }
  }

  OPN2_Write(&chip, 0, 0x28);
  OPN2_Write(&chip, 1, 0x00 | CHANNEL);
  return result;
}

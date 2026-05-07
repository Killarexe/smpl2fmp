#include "OPN2_individual.h"
#include "fm_individual.h"
#include "../fm_cores/ym3438.h"
#include "../math/random.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHIP_CLOCK 7670454
#define CHANNEL 0

ym3438_t YM3438;

uint16_t get_fnum(float frequency) {
  uint8_t octave = (uint8_t)(ceil(frequency / 82.41) - 1);
  if (octave > 7) {
    octave = 7;
  }

  double note = 144.0 * frequency * 1048526 / CHIP_CLOCK;
  return (uint16_t)round(note / (double)(1 << octave));
}

void OPN2_write_register(uint8_t address, uint8_t data) {
  OPN2_WriteBuffered(&YM3438, 0, address);
  OPN2_WriteBuffered(&YM3438, 1, data);
}

void OPN2_set_patch(OPN2Patch patch) {
  uint8_t algorithm_feedback = ((patch.feedback & 0x07) << 3) | (patch.algorithm & 0x07);
  OPN2_write_register(0xB0 + CHANNEL, algorithm_feedback);
  OPN2_write_register(0xB4 + CHANNEL, 0xC0);
  uint8_t operator_base_addresses[4] = {0x30, 0x38, 0x34, 0x3C};

  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
    OPN2Operator operator = patch.operators[i];
    uint8_t base_address = operator_base_addresses[i] + CHANNEL;
    uint8_t detune_multiplier = (operator.detune << 4) | (operator.multiple & 0x0F);
    uint8_t keyscale_attackrate = operator.attack_rate & 0x1F;
    uint8_t am_decay_rate = operator.decay_rate & 0x1F;
    uint8_t sustainlevel_release = (operator.sustain_level << 4) | 0x0F;

    OPN2_write_register(base_address, detune_multiplier);
    OPN2_write_register(base_address + 0x10, operator.total_level);
    OPN2_write_register(base_address + 0x20, keyscale_attackrate);
    OPN2_write_register(base_address + 0x30, am_decay_rate);
    OPN2_write_register(base_address + 0x40, 0x00); // No sustain rate
    OPN2_write_register(base_address + 0x50, sustainlevel_release);
    OPN2_write_register(base_address + 0x60, 0x00); // No SSG-EG
  }
}

FMIndividual* OPN2_create_individuals(size_t size) {
  FMIndividual* individuals = malloc(size * sizeof(FMIndividual));
  if (!individuals) {
    return NULL;
  }

  for (size_t i = 0; i < size; i++) {
    individuals[i].type = FM_INIDIVIDUAL_TYPE_OPN;
    individuals[i].randomize = OPN2_randomize;
    individuals[i].crossover = OPN2_crossover;
    individuals[i].mutate = OPN2_mutate;
    individuals[i].synthesize = OPN2_synthesize;
    individuals[i].print_data = OPN2_print_data;
    individuals[i].fitness = 0.0;
    individuals[i].fm_patch = malloc(sizeof(OPN2Patch));
  }

  return individuals;
}

void OPN2_randomize(FMIndividual* individual) {
  if (!individual) {
    return;
  }
  if (!individual->fm_patch || individual->type != FM_INIDIVIDUAL_TYPE_OPN) {
    return;
  }

  OPN2Patch* patch = (OPN2Patch*)individual->fm_patch;

  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
    OPN2Operator* operator = &patch->operators[i];
    operator->multiple = random_range_u8(0, 8);
    operator->total_level = random_range_u8(0, 64);
    operator->attack_rate = random_range_u8(0, 32);
    operator->decay_rate = random_range_u8(0, 32);
    operator->sustain_level = random_range_u8(0, 16);
    operator->detune = random_range_u8(0, 7);
  }

  patch->algorithm = random_range_u8(0, 8);
  patch->feedback = random_range_u8(0, 0);
}

void OPN2_crossover(FMIndividual* parent1, FMIndividual* parent2, FMIndividual* child) {
  if (!parent1 || !parent2 || !child) {
    return;
  }

  if (!parent1->fm_patch || !parent2->fm_patch || !child->fm_patch) {
    return;
  }

  if (parent1->type != FM_INIDIVIDUAL_TYPE_OPN || parent2->type != FM_INIDIVIDUAL_TYPE_OPN || child->type != FM_INIDIVIDUAL_TYPE_OPN) {
    return;
  }

  OPN2Patch* parent1_patch = (OPN2Patch*)parent1->fm_patch;
  OPN2Patch* parent2_patch = (OPN2Patch*)parent2->fm_patch;
  OPN2Patch* child_patch = (OPN2Patch*)child->fm_patch;

  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
    if (random_bool(0.5f)) {
      child_patch->operators[i] = parent1_patch->operators[i];
    } else {
      child_patch->operators[i] = parent2_patch->operators[i];
    }
  }

  child_patch->algorithm = random_bool(0.5f) ? parent1_patch->algorithm : parent2_patch->algorithm;
  child_patch->feedback = random_bool(0.5f) ? parent1_patch->feedback : parent2_patch->feedback;
}

void OPN2_mutate(FMIndividual* individual, float mutate_rate) {
  if (!individual) {
    return;
  }
  if (!individual->fm_patch || individual->type != FM_INIDIVIDUAL_TYPE_OPN) {
    return;
  }

  OPN2Patch* patch = (OPN2Patch*)individual->fm_patch;
  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
    OPN2Operator* operator = &patch->operators[i];
    if (random_bool(mutate_rate)) {
      operator->multiple = random_range_u8(0, 16);
    }
    if (random_bool(mutate_rate)) {
      operator->total_level = random_range_u8(0, 128);
    }
    if (random_bool(mutate_rate)) {
      operator->attack_rate = random_range_u8(0, 32);
    }
    if (random_bool(mutate_rate)) {
      operator->decay_rate = random_range_u8(0, 32);
    }
    if (random_bool(mutate_rate)) {
      operator->sustain_level = random_range_u8(0, 16);
    }
    if (random_bool(mutate_rate)) {
      operator->detune = random_range_u8(0, 7);
    }
  }

  if (random_bool(mutate_rate)) {
    patch->algorithm = random_range_u8(0, 8);
  }
  if (random_bool(mutate_rate)) {
    patch->feedback = random_range_u8(0, 8);
  }
}

void OPN2_synthesize(FMIndividual* individual, float base_frequency, uint32_t sample_rate, double* output, size_t output_size) {
  if (!individual) {
    return;
  }
  if (!individual->fm_patch || individual->type != FM_INIDIVIDUAL_TYPE_OPN) {
    return;
  }

  OPN2Patch* patch = (OPN2Patch*)individual->fm_patch;

  OPN2_Reset(&YM3438, sample_rate, CHIP_CLOCK);
  OPN2_SetChipType(ym3438_mode_ym2612);

  // Disable DAC and LFO
  OPN2_write_register(0x2A, 0x00);
  OPN2_write_register(0x2B, 0x00);
  OPN2_write_register(0x22, 0x08);

  // Disable timers and put it on "normal" mode
  OPN2_write_register(0x24, 0x00);
  OPN2_write_register(0x25, 0x00);
  OPN2_write_register(0x26, 0x00);
  OPN2_write_register(0x27, 0x00);

  OPN2_set_patch(*patch);

  uint8_t octave = (uint8_t)(ceil(base_frequency / 82.41) - 1);
  if (octave > 7) {
    octave = 7;
  }
  uint16_t fnum = get_fnum(base_frequency);

  OPN2_write_register(0xA4 + CHANNEL, ((octave << 3) | (fnum >> 8)) & 0x3F);
  OPN2_write_register(0xA0 + CHANNEL, fnum & 0xFF);

  for (size_t channel = 0; channel < 6; channel++) {
    OPN2_write_register(0x28, 0x00 | channel);
  }

  int16_t* raw_output = malloc(2 * output_size * sizeof(int16_t)); 
  OPN2_write_register(0x28, 0xF0 | CHANNEL);
  OPN2_GenerateStream(&YM3438, raw_output, output_size);
  OPN2_write_register(0x28, 0x00 | CHANNEL);

  for (size_t i = 0; i < output_size; i++) {
    output[i] = (double)(raw_output[i * 2] + raw_output[i * 2 + 1]) / 8192.0;
  }

  free(raw_output);
}

void OPN2_print_data(FMIndividual* individual) {
  if (!individual) {
    return;
  }
  if (!individual->fm_patch || individual->type != FM_INIDIVIDUAL_TYPE_OPN) {
    return;
  }

  OPN2Patch* patch = (OPN2Patch*)individual->fm_patch;

  printf("Algorithm: %d\n", patch->algorithm);
  printf("Feedback: %d\n", patch->feedback);
  for (size_t i = 0; i < OPN2_OPERATOR_COUNT; i++) {
    OPN2Operator* operator = &patch->operators[i];
    printf("Operator n°%li\n", (i + 1));
    printf("\tMultiple: %d\n", operator->multiple);
    printf("\tTotal level: %d\n", operator->total_level);
    printf("\tAttack rate: %d\n", operator->attack_rate);
    printf("\tDecay rate: %d\n", operator->decay_rate);
    printf("\tSustain level: %d\n", operator->sustain_level);
    printf("\tDetune: %d\n", operator->detune);
  }
} 

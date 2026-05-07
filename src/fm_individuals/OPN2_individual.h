#ifndef OPN2_INDIVIDUAL_H
#define OPN2_INDIVIDUAL_H

#define OPN2_OPERATOR_COUNT 4

#include "fm_individual.h"

#include <stdint.h>
#include <stddef.h>

typedef struct OPN2Operator {
  uint8_t multiple;
  uint8_t total_level;
  uint8_t attack_rate;
  uint8_t decay_rate;
  uint8_t sustain_level;
  uint8_t detune;
} OPN2Operator;

typedef struct OPN2Patch {
  uint8_t algorithm;
  uint8_t feedback;
  OPN2Operator operators[OPN2_OPERATOR_COUNT];
} OPN2Patch;

FMIndividual* OPN2_create_individuals(size_t size);

void OPN2_randomize(FMIndividual* individual);

void OPN2_crossover(FMIndividual* parent1, FMIndividual* parent2, FMIndividual* child);

void OPN2_mutate(FMIndividual* individual, float mutate_rate);

void OPN2_synthesize(FMIndividual* individual, float base_frequency, uint32_t sample_rate, double* output, size_t output_size);

void OPN2_print_data(FMIndividual* individual);

#endif

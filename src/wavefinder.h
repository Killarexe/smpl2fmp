#ifndef WAVEFINDER_H
#define WAVEFINDER_H

#include "fm_individuals/fm_individual.h"
#include <stddef.h>

typedef struct Wavefinder {
  size_t population_size;
  size_t generation_count;
  FMIndividual* individuals;

  size_t tournament_size;

  double* target_spectrum;
  double target_frequency;
  size_t target_size;
  uint32_t target_sample_rate;
} Wavefinder;


void wavefinder_init(
  Wavefinder* wavefinder,
  FMIndividualType individual_type,
  size_t population_size, size_t generation_count, size_t tournament_size,
  double* target_spectrum, double target_frequency, size_t target_size, uint32_t target_sample_rate
);
FMIndividual* wavefinder_find_individual(Wavefinder* wavefinder);
void wavefinder_end(Wavefinder* wavefinder);

#endif

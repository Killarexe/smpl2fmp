#include "fm_individual.h"

#include <stdlib.h>
#include <math.h>
#include "../math/sfft.h"
#include "../math/diffs.h"
#include "OPN2_individual.h"

#define MAX_FRAMES 8192

FMIndividual* fm_inidividual_clone(FMIndividual* individual) {
  FMIndividual* clone = malloc(sizeof(FMIndividual));
  *clone = *individual;
  switch(individual->type) {
    case FM_INIDIVIDUAL_TYPE_OPN:
    default:
      clone->fm_patch = malloc(sizeof(OPN2Patch));
      *(OPN2Patch*)clone->fm_patch = *(OPN2Patch*)individual->fm_patch;
      break;
  }

  return clone;
}

void fm_inidividual_fitness(FMIndividual* individual, double* target_spec, double target_frequency, uint32_t sample_rate, size_t target_size) {
  double* synthetized = malloc(target_size * sizeof(double));
  individual->synthesize(individual, target_frequency, sample_rate, synthetized, target_size);

  size_t output_size = 0;
  size_t fft_size = pow(2.0, ceil(log2((2 * sample_rate) / target_frequency)));
  size_t hop_size = fft_size / 2;
  double* synthetized_spec = averaged_spectrum(synthetized, target_size, &output_size, fft_size, hop_size, MAX_FRAMES);
  free(synthetized);

  size_t spec_size = fft_size / 2 + 1;
  double cosine_score = cosine_similarity(target_spec, synthetized_spec, spec_size);
  double distance_score = spectral_distance(target_spec, synthetized_spec, spec_size);
  //double flux_score = spectral_flux(target_spec, synthetized_spec, spec_size);
  //double fitness = (cosine_score + distance_score + flux_score) / 3.0;
  double harmonic_score = harmonic_peak_score(target_spec, synthetized_spec, spec_size, target_frequency, sample_rate, fft_size);
  double fitness = 0.25 * cosine_score + 0.25 * distance_score + 0.5 * harmonic_score;

  free(synthetized_spec);

  individual->fitness = fitness;
}

void fm_inidividual_free(FMIndividual* individual) {
  if (!individual) {
    return;
  }
  if (individual->fm_patch) {
    free(individual->fm_patch);
  }
  free(individual);
}

void fm_inidividual_array_free(FMIndividual* individuals, size_t size) {
  if (!individuals) {
    return;
  }

  for (size_t i = 0; i < size; i++) {
    if (individuals[i].fm_patch) {
      free(individuals[i].fm_patch);
    }
  }

  free(individuals);
}

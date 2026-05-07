#include "wavefinder.h"
#include "fm_individuals/OPN2_individual.h"
#include "fm_individuals/fm_individual.h"
#include "math/random.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int compare_individual(const void* a, const void* b) {
  FMIndividual* first = (FMIndividual*)a;
  FMIndividual* second = (FMIndividual*)b;
  if (first->fitness > second->fitness) {
    return -1;
  } else if (first->fitness < second->fitness) {
    return 1;
  }
  return 0;
}

void sort_individuals_by_fitness(Wavefinder* wavefinder) {
  qsort(wavefinder->individuals, wavefinder->population_size, sizeof(FMIndividual), compare_individual);
}

void wavefinder_init(
  Wavefinder* wavefinder,
  FMIndividualType individual_type,
  size_t population_size, size_t generation_count, size_t tournament_size,
  double* target_spectrum, double target_frequency, size_t target_size, uint32_t target_sample_rate
) {
  if (!wavefinder) {
    return;
  }

  printf("Population size: %li individuals\n", population_size);
  printf("Generation count: %li\n", generation_count);
  printf("Tournament size: %li individuals\n", tournament_size);
  printf("Target frequency: %fHz\n", target_frequency);
  printf("Target size: %li samples\n", target_size);
  printf("Target sample rate: %uHz\n", target_sample_rate);

  wavefinder->population_size = population_size;
  wavefinder->generation_count = generation_count;
  wavefinder->target_spectrum = target_spectrum;
  wavefinder->target_frequency = target_frequency;
  wavefinder->target_size = target_size;
  wavefinder->target_sample_rate = target_sample_rate;
  wavefinder->tournament_size = tournament_size;

  switch (individual_type) {
    case FM_INIDIVIDUAL_TYPE_OPN:
    default:
      wavefinder->individuals = OPN2_create_individuals(population_size);
      break;
  }

  srand(time(NULL));

  for (size_t i = 0; i < population_size; i++) {
    wavefinder->individuals[i].randomize(&wavefinder->individuals[i]);
  } 
}

FMIndividual* wavefinder_tournament_select(Wavefinder* wavefinder) {
  FMIndividual* best = NULL;
  size_t seen_indicies[wavefinder->tournament_size];
  size_t seen_count = 0;

  for (size_t i = 0; i < wavefinder->tournament_size; i++) {
    size_t contestant_index;
    bool already_seen;

    do {
      already_seen = false;
      contestant_index = random_range_s(0, wavefinder->population_size - 1);
      for (size_t j = 0; j < seen_count; j++) {
        if (seen_indicies[j] == contestant_index) {
          already_seen = true;
          break;
        }
      }
    } while (already_seen);

    seen_indicies[seen_count++] = contestant_index;
    FMIndividual* contestant = &wavefinder->individuals[contestant_index];
    if (best == NULL || contestant->fitness > best->fitness) {
      best = contestant;
    }
  }

  return best;
}

FMIndividual* wavefinder_find_individual(Wavefinder* wavefinder) {
  for (size_t generation = 0; generation < wavefinder->generation_count; generation++) {
    for (size_t i = 0; i < wavefinder->population_size; i++) {
      fm_inidividual_fitness(&wavefinder->individuals[i], wavefinder->target_spectrum, wavefinder->target_frequency, wavefinder->target_sample_rate, wavefinder->target_size);
    }
    sort_individuals_by_fitness(wavefinder);

    /*for (size_t i = 0; i < wavefinder->population_size; i++) {
      printf("I%li: %f\n", i, wavefinder->individuals[i].fitness);
    }*/

    printf("Best fitness generation %li: %f\n", generation, wavefinder->individuals[0].fitness);
    if (generation == (wavefinder->generation_count - 1)) {
      break;
    }

    FMIndividual* new_individuals = NULL;
    switch (wavefinder->individuals[0].type) {
      case FM_INIDIVIDUAL_TYPE_OPN:
      default:
        new_individuals = OPN2_create_individuals(wavefinder->population_size);
        break;
    }
    if (!new_individuals) {
      FMIndividual* result = fm_inidividual_clone(&wavefinder->individuals[0]);
      return result;
    }

    //printf("New: %p\n", new_individuals);

    switch (wavefinder->individuals[0].type) {
      case FM_INIDIVIDUAL_TYPE_OPN:
      default:
        OPN2Patch* fm_patch = (OPN2Patch*)new_individuals[0].fm_patch;
        *fm_patch = *(OPN2Patch*)wavefinder->individuals[0].fm_patch;
        break;
    }


    for (size_t i = 1; i < wavefinder->population_size; i++) {
      FMIndividual* parent1 = wavefinder_tournament_select(wavefinder);
      FMIndividual* parent2 = wavefinder_tournament_select(wavefinder);
      parent1->crossover(parent1, parent2, &new_individuals[i]);
      new_individuals[i].mutate(&new_individuals[i], 0.3);
    }

    fm_inidividual_array_free(wavefinder->individuals, wavefinder->population_size);

    wavefinder->individuals = new_individuals;
  }

  return fm_inidividual_clone(&wavefinder->individuals[0]);
}

void wavefinder_end(Wavefinder* wavefinder) {
  if (!wavefinder) {
    return;
  }
  if (!wavefinder->individuals) {
    return;
  }
  fm_inidividual_array_free(wavefinder->individuals, wavefinder->population_size);
}

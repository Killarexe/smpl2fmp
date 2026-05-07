#ifndef FM_INIDIVIDUAL_H
#define FM_INIDIVIDUAL_H

#include <stdint.h>
#include <stddef.h>

typedef enum FMIndividualType {
  FM_INIDIVIDUAL_TYPE_OPN,
} FMIndividualType;

typedef struct FMIndividual {
  FMIndividualType type;
  void (*randomize)(struct FMIndividual* individual);
  void (*crossover)(struct FMIndividual* parent1, struct FMIndividual* parent2, struct FMIndividual* child);
  void (*mutate)(struct FMIndividual* individual, float mutation_rate);
  void (*synthesize)(struct FMIndividual* individual, float base_frequency, uint32_t sample_rate, double* output, size_t output_size);
  void (*print_data)(struct FMIndividual* individual);
  void* fm_patch;
  double fitness;
} FMIndividual;

FMIndividual* fm_inidividual_clone(FMIndividual* individual);

void fm_inidividual_fitness(FMIndividual* individual, double* target_spec, double target_frequency, uint32_t sample_rate, size_t target_size);

void fm_inidividual_free(FMIndividual* individual);
void fm_inidividual_array_free(FMIndividual* individuals, size_t size);

#endif

#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

double random_double();
double random_range_double(double min, double max);

uint8_t random_u8();
uint8_t random_range_u8(uint8_t min, uint8_t max);

size_t random_range_s(size_t min, size_t max);

bool random_bool(float probability);

#endif

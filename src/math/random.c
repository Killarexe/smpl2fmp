#include "random.h"
#include <stdlib.h>
#include <math.h>

double random_double() {
  return (double)rand() / (double)RAND_MAX;
}

double random_range_double(double min, double max) {
  return min + random_double() * (max - min);
}

uint8_t random_u8() {
  return (uint8_t)(rand() & 0xFF);
}

uint8_t random_range_u8(uint8_t min, uint8_t max) {
  return (uint8_t)round(random_range_double(min, max));
}

size_t random_range_s(size_t min, size_t max) {
  return (size_t)round(random_range_double(min, max));
}

bool random_bool(float probability) {
  return random_double() > probability;
}

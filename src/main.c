#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "fm_individuals/OPN2_individual.h"
#include "fm_individuals/fm_individual.h"
#include "math/sfft.h"
#include "tinywav/tinywav.h"
#include "wavefinder.h"

void test_patch() {
  OPN2Patch patch = {
    .algorithm = 3,
    .feedback = 7,
    .operators = {
      (OPN2Operator){
        .attack_rate = 31,
        .decay_rate = 31,
        .sustain_level = 0,
        .total_level = 31,
        .multiple = 1
      },
      (OPN2Operator){
        .attack_rate = 31,
        .decay_rate = 31,
        .sustain_level = 0,
        .total_level = 34,
        .multiple = 1
      },
      (OPN2Operator){
        .attack_rate = 31,
        .decay_rate = 31,
        .sustain_level = 0,
        .total_level = 45,
        .multiple = 1
      },
      (OPN2Operator){
        .attack_rate = 31,
        .decay_rate = 31,
        .sustain_level = 0,
        .total_level = 0,
        .multiple = 0
      }
    }
  };

  FMIndividual fm_individual = {
    .type = FM_INIDIVIDUAL_TYPE_OPN,
    .crossover = OPN2_crossover,
    .mutate = OPN2_mutate,
    .randomize = OPN2_randomize,
    .synthesize = OPN2_synthesize,
    .print_data = OPN2_print_data,
    .fitness = 0.0,
    .fm_patch = &patch
  };

  fm_individual.print_data(&fm_individual);

  const size_t sample_rate = 44100;
  const size_t sample_size = sample_rate * 2;
  double* raw_output_samples = malloc(sample_size * sizeof(double));
  OPN2_synthesize(&fm_individual, 440.0, sample_rate, raw_output_samples, sample_size);

  float* output_samples = malloc(sample_size * sizeof(float));
  for (size_t i = 0; i < sample_size; i++) {
    output_samples[i] = (float)raw_output_samples[i];
  }
  free(raw_output_samples);

  TinyWav write_wav;
  tinywav_open_write(&write_wav, 1, sample_rate, TW_FLOAT32, TW_INLINE, "test.wav");
  tinywav_write_f(&write_wav, output_samples, sample_size);
  tinywav_close_write(&write_wav);
  free(output_samples);
}

int main(int argc, char *argv[]) {
  //test_patch();
  if (argc < 3) {
    printf("smpl2fmp [input_file] [output_file]\n");
    return 0;
  }

  char* input_file_path = argv[1];
  char* output_file_path = argv[2];

  TinyWav read_input;
  if (tinywav_open_read(&read_input, input_file_path, TW_INLINE)) {
    printf("Failed to read WAV file...\n");
    return 0;
  };

  uint32_t input_sample_rate = read_input.h.SampleRate;
  int32_t input_sample_size = read_input.numFramesInHeader;
  int16_t input_num_channels = read_input.numChannels;

  float* input_samples_raw = malloc(input_num_channels * input_sample_size * sizeof(float));
  tinywav_read_f(&read_input, input_samples_raw, input_sample_size);
  tinywav_close_read(&read_input);

  double* input_samples = malloc(input_sample_size * sizeof(double));
  for (int32_t i = 0; i < input_num_channels; i++) {
    for (int16_t j = 0; j < input_sample_size; j++) { 
      input_samples[j] += input_samples_raw[j + (i * input_sample_size)] / input_num_channels;
    }
  }
  free(input_samples_raw); 

  double frequency = base_frequency(input_samples, input_sample_size, input_sample_rate);

  size_t spec_size = 0;
  size_t fft_size = pow(2.0, ceil(log2((2 * input_sample_rate) / frequency)));
  size_t hop_size = fft_size / 2;
  double* spec = averaged_spectrum(input_samples, input_sample_size, &spec_size, fft_size, hop_size, 8192);
  free(input_samples);

  Wavefinder wavefinder;
  wavefinder_init(&wavefinder, FM_INIDIVIDUAL_TYPE_OPN, 100, 100, 3, spec, frequency, input_sample_size, input_sample_rate);
  FMIndividual* result = wavefinder_find_individual(&wavefinder);
  wavefinder_end(&wavefinder);
  free(spec);

  double* raw_output_samples = malloc(input_sample_size * sizeof(double));
  result->synthesize(result, frequency, input_sample_rate, raw_output_samples, input_sample_size);

  float* output_samples = malloc(input_sample_size * sizeof(float));
  for (size_t i = 0; i < (size_t)input_sample_size; i++) {
    output_samples[i] = (float)raw_output_samples[i];
  }
  free(raw_output_samples);

  TinyWav write_output;
  if (tinywav_open_write(&write_output, 1, input_sample_rate, TW_FLOAT32, TW_INLINE, output_file_path)) {
    printf("Faield to write file\n");
    return 0;
  }
  tinywav_write_f(&write_output, output_samples, input_sample_size);
  tinywav_close_write(&write_output);
  free(output_samples);

  result->print_data(result);
  fm_inidividual_free(result);

  return 0;
} 

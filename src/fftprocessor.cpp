#include "fftprocessor.h"
#include <cmath>
#include <cstddef>
#include <fftw3.h>
#include <stdexcept>
#include <vector>

FFTProcessor::FFTProcessor(size_t sampleSize, size_t numSamples) {
  this->sampleSize = sampleSize;
  this->numSamples = numSamples;
  outputSize = sampleSize / 2 + 1;

  inputBuffer = fftw_alloc_real(sampleSize * numSamples);
  outputBuffer = fftw_alloc_complex(outputSize * numSamples);

  int n[] = {(int)sampleSize};

  plan = fftw_plan_many_dft_r2c(
    1, n, numSamples, inputBuffer, NULL, 1, sampleSize, outputBuffer, NULL, 1, outputSize, FFTW_ESTIMATE
  );
}

FFTProcessor::~FFTProcessor() {
  fftw_destroy_plan(plan);
  fftw_free(inputBuffer);
  fftw_free(outputBuffer);
  fftw_cleanup();
}

double* FFTProcessor::getInput(size_t index) {
  if (index >= numSamples) {
    throw std::invalid_argument("simple index out of bounds!");
  }
  return inputBuffer + index * sampleSize;
}

void FFTProcessor::setSample(size_t index, const std::vector<double>& sample) {
  if (index >= numSamples || sample.size() != sampleSize) {
    throw std::invalid_argument("Invalid sample index or sample size");
  }

  double* sampleStart = inputBuffer + index * sampleSize;
  std::copy(sample.begin(), sample.end(), sampleStart);
}

void FFTProcessor::execute() {
  fftw_execute(plan);
}

std::vector<double> FFTProcessor::getMagnitude(size_t index) {
  if (index >= numSamples) {
    throw std::invalid_argument("simple index out of bounds!");
  }

  std::vector<double> result(outputSize);
  fftw_complex* outputStart = outputBuffer + index * outputSize;

  for (size_t i = 0; i < outputSize; ++i) {
    double real = outputStart[i][0];
    double imag = outputStart[i][1];
    result[i] = std::sqrt(real * real + imag * imag);
  }

  return result;
}

#include "smpl2fmp.h"

#include <CLI/CLI.hpp>
#include <AudioFile.h>
#include "Core/Individuals/OPN2Individual.h"
#include "Core/individual.h"
#include "wavefinder.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <random>

namespace fs = std::filesystem;

int init(int argc, char **argv) {
  CLI::App app{"A program which convert samples into FM patches."};
  argv = app.ensure_utf8(argv);

  fs::path input_filename = "";
  app.add_option("-i,--input", input_filename, "Input sample file");

  fs::path output_sample_filename = "";
  app.add_option("-o,--output", output_sample_filename, "Output FM sample file");

  size_t generations = 1000;
  app.add_option("-g,--generations", generations, "Number of generations");

  size_t populations = 100;
  app.add_option("-p,--populations", populations, "Number of populations per generations");

  size_t tournaments = 3;
  app.add_option("-t,--tournaments", tournaments, "Number of tournaments on the selection phase");

  double mutate_rate = 0.2;
  app.add_option("-m,--mutate", mutate_rate, "Mutation rate on the mixing phase");

  double sample_frequency = 0.0;
  app.add_option("-f,--frequency", sample_frequency, "Set sample frequency (if the programs detects wrong)");


  CLI11_PARSE(app, argc, argv);

  if (argc < 2) {
    //TODO: Launch ImGui software.
    std::cout << "Launching graphical interface..." << std::endl;
    return 0;
  }

  if (!fs::exists(input_filename)) {
    std::cout << "No input file found." << std::endl;
    return 0;
  }

  std::cout << "Starting converting sample fo FM patch..." << std::endl;
  std::cout << "Input sample file: " << input_filename << std::endl;
  if (!output_sample_filename.empty()) {
    std::cout << "Output sample file: " << output_sample_filename << std::endl;
  }
  std::cout << "Number of generations: " << generations << std::endl;
  std::cout << "Number of populations: " << populations << std::endl;
  std::cout << "Number of tournaments: " << tournaments << std::endl;
  std::cout << "Mutation rate: " << mutate_rate << std::endl;

  AudioFile<double> input_samples;
  input_samples.load(input_filename);

  auto wavefinder = createWavefinder<OPN2Individual>(populations, generations, tournaments, mutate_rate, std::random_device{}());
  Individual* result = wavefinder->find(input_samples, sample_frequency);

  std::cout << "Finished! Best fitness: " << result->fitness << "\nFM patch:" << std::endl;
  result->printData();

  if (!output_sample_filename.empty()) {
    result->saveAudio(output_sample_filename, wavefinder->targetFrequency, input_samples.getLengthInSeconds(), input_samples.getSampleRate());
  }
  return 0;
}

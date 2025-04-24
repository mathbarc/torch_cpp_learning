#ifndef UTILS_DATASET_HPP
#define UTILS_DATASET_HPP
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <torch/data.h>
#include <torch/data/dataloader.h>
#include <torch/torch.h>

class CSVDataset : public torch::data::Dataset<CSVDataset>
{
  private:
    torch::Tensor samples;
    torch::Tensor labels;
    std::unordered_map<std::string, size_t> labelMap;

  public:
    CSVDataset(std::string filePath, std::string delimiter = ";", bool skipFirstLine = false);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};
#endif

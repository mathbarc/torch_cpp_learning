#include "datasets.hpp"
#include "string.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/copy.h>
#include <cctype>
#include <cstddef>
#include <iostream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/data/example.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

CSVDataset::CSVDataset(std::string filePath, std::string delimiter, bool skipFirstLine)
{
    std::ifstream file(filePath);
    if(!file.is_open())
        throw std::runtime_error(filePath + " could not be openned");
    std::string line;

    bool isFirstLine = true;
    while(std::getline(file, line))
    {
        std::vector<std::string> lineContent = splitString(line, delimiter);
        if(isFirstLine)
        {
            isFirstLine = false;
            if(skipFirstLine)
                continue;
        }
        std::vector<float> sampleValues;
        for(size_t i = 0; i < lineContent.size() - 1; i++)
        {
            sampleValues.push_back(std::atof(lineContent[i].c_str()));
        }

        if(this->labelMap.find(lineContent.back()) == this->labelMap.end())
        {
            this->labelMap.insert({lineContent.back(), this->labelMap.size()});
        }

        long labelValue = this->labelMap[lineContent.back()];

        torch::Tensor sample = torch::from_blob(sampleValues.data(), {1, sampleValues.size()}, torch::kFloat32);
        torch::Tensor label = torch::from_blob(&labelValue, {1, 1}, torch::kInt64);

        if(this->samples.numel())
        {
            this->samples = torch::concat({this->samples, sample}, 0);
            this->labels = torch::concat({this->labels, label}, 0);
        }
        else
        {
            this->samples = sample.clone();
            this->labels = label.clone();
        }
    }
}

torch::data::Example<> CSVDataset::get(size_t index)
{
    return {this->samples[index], this->labels[index]};
}

torch::optional<size_t> CSVDataset::size() const
{
    return this->samples.size(0);
}

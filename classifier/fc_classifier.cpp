#include "datasets.hpp"
#include <torch/data/dataloader_options.h>
#include <torch/serialize.h>

int main(int argc, char **argv)
{
    CSVDataset dataset("/data/ssd1/datasets/crop_recommendation/Crop_Recommendation.csv", ",", true);
    dataset.map(torch::data::transforms::Stack<>());

    auto dataloader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions(32));

    for(auto &batch : *dataloader)
    {
        std::cout << batch.size() << std::endl;
    }

    return 0;
}

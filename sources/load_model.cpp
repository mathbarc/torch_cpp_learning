#include <fstream>
#include <memory>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include <iostream>

int main(int argc, char **argv)
{
    torch::nn::Linear model(5, 1);
    std::ifstream file("linear.pt");
    torch::load(model, file);
    model->pretty_print(std::cout);
    return 0;
}

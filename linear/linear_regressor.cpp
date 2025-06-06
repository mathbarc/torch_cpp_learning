
#include <ATen/core/TensorBody.h>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nn/modules/container/parameterdict.h>
#include <torch/nn/modules/linear.h>
#include <torch/script.h>
#include <torch/serialize/output-archive.h>
#include <torch/torch.h>
#include <vector>

#define POLY_DEGREE 4

class LinearRegressor : public torch::nn::Module
{

  public:
    torch::nn::Linear fc;
    LinearRegressor(long poly_degree) : fc(torch::nn::LinearOptions(poly_degree, 1))
    {
        register_module("linear", this->fc);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return this->fc(x);
    }
};

// Builds features i.e. a matrix with columns [x, x^2, x^3, x^4].
torch::Tensor make_features(torch::Tensor x)
{
    x = x.unsqueeze(1);
    std::vector<torch::Tensor> xs;
    for(int64_t i = 0; i < POLY_DEGREE; ++i)
        xs.push_back(x.pow(i + 1));
    return torch::cat(xs, 1);
}

// Approximated function.
torch::Tensor f(torch::Tensor x, torch::Tensor W_target, torch::Tensor b_target)
{
    return x.mm(W_target) + b_target.item();
}

// Creates a string description of a polynomial.
std::string poly_desc(torch::Tensor W, torch::Tensor b)
{
    auto size = W.size(0);
    std::ostringstream stream;

    stream << "y = ";
    for(int64_t i = 0; i < size; ++i)
        stream << W[i].item<float>() << " x^" << size - i << " ";
    stream << "+ " << b[0].item<float>();
    return stream.str();
}

// Builds a batch i.e. (x, f(x)) pair.
std::pair<torch::Tensor, torch::Tensor> get_batch(torch::Tensor W_target, torch::Tensor b_target,
                                                  int64_t batch_size = 32)
{
    auto random = torch::randn({batch_size});
    auto x = make_features(random);
    auto y = f(x, W_target, b_target);
    return std::make_pair(x, y);
}

int main()
{
    auto W_target = torch::randn({POLY_DEGREE, 1}) * 5;
    auto b_target = torch::randn({1}) * 5;

    // Define the model and optimizer
    auto fc = LinearRegressor(POLY_DEGREE);
    torch::optim::SGD optim(fc.parameters(), .1);

    float loss = 0;
    int64_t batch_idx = 0;

    while(++batch_idx)
    {
        // Get data
        torch::Tensor batch_x, batch_y;
        std::tie(batch_x, batch_y) = get_batch(W_target, b_target);

        // Reset gradients
        optim.zero_grad();

        // Forward pass
        auto output = torch::smooth_l1_loss(fc.forward(batch_x), batch_y);
        loss = output.item<float>();
        std::cout << loss << std::endl;

        // Backward pass
        output.backward();

        // Apply gradients
        optim.step();

        // Stop criterion
        if(loss < 1e-3f)
            break;
    }

    std::cout << "Loss: " << loss << " after " << batch_idx << " batches" << std::endl;
    std::cout << "==> Learned function:\t" << poly_desc(fc.fc->weight.view({-1}), fc.fc->bias) << std::endl;
    std::cout << "==> Actual function:\t" << poly_desc(W_target.view({-1}), b_target) << std::endl;

    torch::serialize::OutputArchive outFile;
    fc.save(outFile);
    outFile.save_to("linear.pt");

    fc.pretty_print(std::cout);

    return 0;
}

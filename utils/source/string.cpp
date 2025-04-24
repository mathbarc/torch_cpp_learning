#include "string.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <torch/data/example.h>
#include <vector>

std::vector<std::string> splitString(std::string line, std::string delimiter)
{
    size_t start = 0;
    line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); }), line.end());

    size_t end = line.find(delimiter, start);

    std::vector<std::string> itens;
    while(end != std::string::npos)
    {
        itens.push_back(line.substr(start, end - start));
        start = end + 1;
        end = line.find(delimiter, start);
    }
    if(start < line.size())
        itens.push_back(line.substr(start, line.size() - start));
    return itens;
}

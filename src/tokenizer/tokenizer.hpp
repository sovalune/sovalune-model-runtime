#pragma once

#include <string>
#include <vector>

namespace sovalune {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;
    
    // Load tokenizer from file
    bool load(const std::string& path);
    
    // Encode text to tokens
    std::vector<int> encode(const std::string& text);
    
    // Decode tokens to text
    std::string decode(const std::vector<int>& tokens);
    
    // Count tokens
    int count_tokens(const std::string& text);
    
    // Get vocab size
    int vocab_size() const;
    
private:
    int vocab_size_ = 0;
};

}  // namespace sovalune

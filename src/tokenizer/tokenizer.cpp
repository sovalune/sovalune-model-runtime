#include "tokenizer.hpp"
#include <iostream>

namespace sovalune {

bool Tokenizer::load(const std::string& path) {
    // TODO: Implement tokenizer loading
    std::cout << "[Tokenizer] Loading from: " << path << std::endl;
    vocab_size_ = 32000;  // Default for LLaMA-style models
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    // TODO: Implement actual tokenization
    // For now, return simple character-based tokens
    std::vector<int> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<int>(c));
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    // TODO: Implement actual detokenization
    std::string result;
    for (int token : tokens) {
        result += static_cast<char>(token);
    }
    return result;
}

int Tokenizer::count_tokens(const std::string& text) {
    return encode(text).size();
}

int Tokenizer::vocab_size() const {
    return vocab_size_;
}

}  // namespace sovalune

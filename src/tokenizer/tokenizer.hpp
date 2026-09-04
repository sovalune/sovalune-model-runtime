#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace sovalune {

/// BPE Tokenizer - implements byte-pair encoding for LLM inference.
///
/// Supports loading vocabulary from a JSON-like simple format and
/// performs tokenization via merge operations.
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    /// Load tokenizer from vocab file (one token per line).
    bool load(const std::string& path);

    /// Encode text to token IDs.
    std::vector<int> encode(const std::string& text);

    /// Decode token IDs to text.
    std::string decode(const std::vector<int>& tokens);

    /// Count tokens in text.
    int count_tokens(const std::string& text);

    /// Get vocab size.
    int vocab_size() const;

    /// Add a token to the vocabulary.
    void add_token(const std::string& token, int id);

    /// Get special token IDs.
    int bos_token() const { return bos_token_; }
    int eos_token() const { return eos_token_; }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    int vocab_size_ = 0;
    int bos_token_ = 1;
    int eos_token_ = 2;

    /// Apply BPE merges to a word.
    std::vector<int> bpe_encode(const std::string& text);

    /// Find the highest priority merge in the word.
    std::pair<int, int> find_best_merge(const std::vector<std::string>& chars) const;
};

}  // namespace sovalune

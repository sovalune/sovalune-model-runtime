#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>

namespace sovalune {

bool Tokenizer::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Failed to open: " << path << std::endl;
        return false;
    }

    std::string line;
    int id = 0;

    // Skip header line if present
    if (std::getline(file, line)) {
        if (line.find("vocab_size") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                vocab_size_ = std::stoi(line.substr(pos + 1));
            }
            id = 0;
        } else {
            // First line is a token
            vocab_[line] = id;
            reverse_vocab_[id] = line;
            id = 1;
        }
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Check if it's a merge line (contains a space separating two tokens)
        if (line.find(' ') != std::string::npos && line[0] != ' ') {
            auto pos = line.find(' ');
            std::string left = line.substr(0, pos);
            std::string right = line.substr(pos + 1);

            // If right part is a number, it's a vocab line
            bool is_number = true;
            for (char c : right) {
                if (!std::isdigit(c)) { is_number = false; break; }
            }

            if (is_number && right.size() < 6) {
                vocab_[left] = std::stoi(right);
                reverse_vocab_[std::stoi(right)] = left;
            } else {
                merges_.emplace_back(left, right);
            }
        } else {
            // Vocab line: token id
            auto pos = line.rfind(' ');
            if (pos != std::string::npos) {
                std::string token = line.substr(0, pos);
                int token_id = std::stoi(line.substr(pos + 1));
                vocab_[token] = token_id;
                reverse_vocab_[token_id] = token;
            } else {
                vocab_[line] = id;
                reverse_vocab_[id] = line;
                id++;
            }
        }
    }

    if (vocab_size_ == 0) {
        vocab_size_ = static_cast<int>(vocab_.size());
    }

    std::cout << "[Tokenizer] Loaded " << vocab_.size() << " tokens, "
              << merges_.size() << " merges" << std::endl;
    return true;
}

void Tokenizer::add_token(const std::string& token, int id) {
    vocab_[token] = id;
    reverse_vocab_[id] = token;
    if (id >= vocab_size_) {
        vocab_size_ = id + 1;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    if (text.empty()) return {};

    std::vector<int> tokens;

    // Split into words (simple whitespace + punctuation splitting)
    std::vector<std::string> words;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];

        if (std::isspace(c) || std::ispunct(c)) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Treat whitespace and punctuation as separate tokens
            std::string special(1, c);
            auto it = vocab_.find(special);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // Unknown character - use byte encoding
                tokens.push_back(static_cast<int>(c) + 256);
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        words.push_back(current);
    }

    // Apply BPE to each word
    for (const auto& word : words) {
        auto word_tokens = bpe_encode(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;
    for (int token : tokens) {
        auto it = reverse_vocab_.find(token);
        if (it != reverse_vocab_.end()) {
            result += it->second;
        } else if (token >= 256 && token < 512) {
            // Byte token
            result += static_cast<char>(token - 256);
        } else {
            result += "<UNK>";
        }
    }
    return result;
}

int Tokenizer::count_tokens(const std::string& text) {
    return static_cast<int>(encode(text).size());
}

int Tokenizer::vocab_size() const {
    return vocab_size_;
}

std::vector<int> Tokenizer::bpe_encode(const std::string& text) {
    if (text.empty()) return {};

    // Start with individual characters
    std::vector<std::string> chars;
    for (char c : text) {
        chars.push_back(std::string(1, c));
    }

    // Apply merges iteratively
    bool changed = true;
    while (changed && chars.size() > 1) {
        changed = false;
        auto [pos, merge_idx] = find_best_merge(chars);

        if (pos >= 0 && merge_idx >= 0) {
            // Apply merge
            std::string merged = chars[pos] + chars[pos + 1];
            chars.erase(chars.begin() + pos + 1);
            chars[pos] = merged;
            changed = true;
        }
    }

    // Convert to token IDs
    std::vector<int> tokens;
    for (const auto& ch : chars) {
        auto it = vocab_.find(ch);
        if (it != vocab_.end()) {
            tokens.push_back(it->second);
        } else {
            // Unknown token - try to find longest matching prefix
            bool found = false;
            for (size_t len = ch.size(); len > 0; --len) {
                auto prefix = ch.substr(0, len);
                auto pit = vocab_.find(prefix);
                if (pit != vocab_.end()) {
                    tokens.push_back(pit->second);
                    found = true;
                    break;
                }
            }
            if (!found) {
                tokens.push_back(0); // UNK token
            }
        }
    }

    return tokens;
}

std::pair<int, int> Tokenizer::find_best_merge(const std::vector<std::string>& chars) const {
    int best_pos = -1;
    int best_priority = -1;
    int best_merge_idx = -1;

    for (size_t i = 0; i + 1 < chars.size(); ++i) {
        for (size_t m = 0; m < merges_.size(); ++m) {
            if (merges_[m].first == chars[i] && merges_[m].second == chars[i + 1]) {
                // Earlier merge = higher priority
                int priority = static_cast<int>(merges_.size() - m);
                if (priority > best_priority) {
                    best_priority = priority;
                    best_pos = static_cast<int>(i);
                    best_merge_idx = static_cast<int>(m);
                }
                break;
            }
        }
    }

    return {best_pos, best_merge_idx};
}

}  // namespace sovalune

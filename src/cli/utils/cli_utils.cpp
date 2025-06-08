#include "cli_utils.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

void CliUtils::clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void CliUtils::printSectionHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void CliUtils::printProgress(int current, int total, const std::string& prefix) {
    const int barWidth = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << prefix;
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ("
              << current << "/" << total << ")\r";
    std::cout.flush();
}

void CliUtils::printTable(const std::vector<std::vector<std::string>>& data,
                          const std::vector<std::string>& headers) {
    if (data.empty())
        return;

    size_t numCols = data[0].size();
    std::vector<size_t> colWidths(numCols, 0);

    // Calculate column widths
    if (!headers.empty()) {
        for (size_t i = 0; i < std::min(headers.size(), numCols); ++i) {
            colWidths[i] = std::max(colWidths[i], headers[i].length());
        }
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < std::min(row.size(), numCols); ++i) {
            colWidths[i] = std::max(colWidths[i], row[i].length());
        }
    }

    // Print headers
    if (!headers.empty()) {
        std::cout << "| ";
        for (size_t i = 0; i < std::min(headers.size(), numCols); ++i) {
            std::cout << std::left << std::setw(colWidths[i]) << headers[i] << " | ";
        }
        std::cout << "\n";

        // Print separator
        std::cout << "|";
        for (size_t i = 0; i < numCols; ++i) {
            std::cout << std::string(colWidths[i] + 2, '-') << "|";
        }
        std::cout << "\n";
    }

    // Print data
    for (const auto& row : data) {
        std::cout << "| ";
        for (size_t i = 0; i < std::min(row.size(), numCols); ++i) {
            std::cout << std::left << std::setw(colWidths[i]) << row[i] << " | ";
        }
        std::cout << "\n";
    }
}

std::string CliUtils::formatNumber(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string CliUtils::formatPercent(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << (value * 100.0) << "%";
    return oss.str();
}

void CliUtils::waitForEnter(const std::string& message) {
    std::cout << message;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

bool CliUtils::isValidNumber(const std::string& str) {
    if (str.empty())
        return false;

    std::istringstream iss(str);
    double value;
    iss >> value;
    return iss.eof() && !iss.fail();
}

bool CliUtils::isValidPositiveInt(const std::string& str) {
    if (str.empty())
        return false;

    for (char c : str) {
        if (!std::isdigit(c))
            return false;
    }

    int value = std::stoi(str);
    return value > 0;
}

bool CliUtils::isValidFilename(const std::string& filename) {
    if (filename.empty())
        return false;

    // Basic filename validation (can be extended)
    const std::string invalid_chars = "<>:\"/\\|?*";
    for (char c : filename) {
        if (invalid_chars.find(c) != std::string::npos) {
            return false;
        }
    }
    return true;
}

std::string CliUtils::toLowerCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string CliUtils::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";

    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> CliUtils::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }

    return tokens;
}
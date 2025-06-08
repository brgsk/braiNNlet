#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class CliUtils {
  public:
    static void clearScreen();
    static void printSectionHeader(const std::string& title);
    static void printProgress(int current, int total, const std::string& prefix = "");
    static void printTable(const std::vector<std::vector<std::string>>& data,
                           const std::vector<std::string>& headers = {});
    static std::string formatNumber(double value, int precision = 4);
    static std::string formatPercent(double value, int precision = 1);
    static void waitForEnter(const std::string& message = "Press Enter to continue...");

    // Input validation helpers
    static bool isValidNumber(const std::string& str);
    static bool isValidPositiveInt(const std::string& str);
    static bool isValidFilename(const std::string& filename);

    // String utilities
    static std::string toLowerCase(const std::string& str);
    static std::string trim(const std::string& str);
    static std::vector<std::string> split(const std::string& str, char delimiter);
};
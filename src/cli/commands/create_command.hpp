#pragma once

#include <string>
#include <vector>

class CreateCommand {
  public:
    int execute(const std::vector<std::string>& args);

  private:
    void showHelp();
    int createFromCommandLine(const std::vector<std::string>& args);
};
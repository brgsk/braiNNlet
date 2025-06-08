#pragma once

#include <string>
#include <vector>

class EvalCommand {
  public:
    int execute(const std::vector<std::string>& args);

  private:
    void showHelp();
};
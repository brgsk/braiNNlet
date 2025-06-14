#include <iostream>
#include <string>

void showBanner()
{
    std::cout << "\033[2J\033[H"; // Clear screen
    std::cout << "\n";
    std::cout << "██████╗ ██████╗  █████╗ ██╗███╗   ██╗███╗   ██╗██╗     ███████╗████████╗\n";
    std::cout << "██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║████╗  ██║██║     ██╔════╝╚══██╔══╝\n";
    std::cout << "██████╔╝██████╔╝███████║██║██╔██╗ ██║██╔██╗ ██║██║     █████╗     ██║   \n";
    std::cout << "██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║██║╚██╗██║██║     ██╔══╝     ██║   \n";
    std::cout << "██████╔╝██║  ██║██║  ██║██║██║ ╚████║██║ ╚████║███████╗███████╗   ██║   \n";
    std::cout << "╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝   \n";
    std::cout << "\n🧠 Interactive Neural Network Explorer\n";
    std::cout << "======================================\n\n";
}

void showMenu()
{
    std::cout << "Choose an option:\n\n";
    std::cout << "1️⃣  About braiNNlet\n";
    std::cout << "2️⃣  Technical Features\n";
    std::cout << "3️⃣  Getting Started Guide\n";
    std::cout << "4️⃣  Training Tips\n";
    std::cout << "5️⃣  Launch GUI Application\n";
    std::cout << "0️⃣  Exit\n\n";
    std::cout << "Enter your choice: ";
}

void showAbout()
{
    std::cout << "\n📋 About braiNNlet:\n";
    std::cout << "════════════════════\n";
    std::cout << "• Build neural networks layer-by-layer with intuitive controls\n";
    std::cout << "• Train on real datasets (MNIST handwritten digits)\n";
    std::cout << "• Visualize training progress with real-time plots\n";
    std::cout << "• Modern C++20 implementation with Qt6 GUI\n";
    std::cout << "• Educational tool for understanding deep learning\n";
    std::cout << "• Academic project for Programming II coursework\n\n";
}

void showTechnical()
{
    std::cout << "\n🔧 Technical Features:\n";
    std::cout << "═══════════════════════\n";
    std::cout << "• Core: C++20, Eigen3 for matrix operations\n";
    std::cout << "• GUI: Qt6 with interactive charts and controls\n";
    std::cout << "• Datasets: MNIST (60,000 samples) with automatic loading\n";
    std::cout << "• Activations: ReLU, Sigmoid, Tanh, Linear\n";
    std::cout << "• Loss Functions: CrossEntropy, MSE, BinaryCrossEntropy\n";
    std::cout << "• Real-time training visualization and metrics\n\n";
}

void showGettingStarted()
{
    std::cout << "\n🚀 Getting Started:\n";
    std::cout << "═══════════════════\n";
    std::cout << "1. Run './build/brainnlet.exe' for the full GUI application\n";
    std::cout << "2. Click 'Load Dataset' and select MNIST\n";
    std::cout << "3. Configure network layers (Add/Edit/Remove)\n";
    std::cout << "4. Set training parameters (epochs, batch size, learning rate)\n";
    std::cout << "5. Click 'Train Network' and watch real-time progress\n";
    std::cout << "6. Experiment with different architectures and parameters\n\n";
}

void showTrainingTips()
{
    std::cout << "\n💡 Training Tips:\n";
    std::cout << "═════════════════\n";
    std::cout << "• Start with 2-3 hidden layers (128, 64 neurons)\n";
    std::cout << "• Use ReLU activation for hidden layers, Linear for output\n";
    std::cout << "• Learning rate: 0.01 is a good starting point\n";
    std::cout << "• Batch size: 32-128 works well for MNIST\n";
    std::cout << "• Watch the loss curve - it should decrease over time\n";
    std::cout << "• If loss plateaus, try adjusting learning rate\n";
    std::cout << "• Validation accuracy shows real performance\n\n";
}

void launchGUI()
{
    std::cout << "\n🚀 Launching GUI Application...\n";
    std::cout << "═══════════════════════════════\n";
    std::cout << "Run this command in your terminal:\n";
    std::cout << "./build/brainnlet.exe\n\n";
    std::cout << "If the GUI doesn't start, make sure Qt6 is properly installed.\n\n";
}

int main()
{
    showBanner();

    int choice;
    std::string input;

    while (true)
    {
        showMenu();
        std::getline(std::cin, input);

        try
        {
            choice = std::stoi(input);
        }
        catch (...)
        {
            choice = -1; // Invalid input
        }

        std::cout << "\n";

        switch (choice)
        {
        case 1:
            showAbout();
            break;
        case 2:
            showTechnical();
            break;
        case 3:
            showGettingStarted();
            break;
        case 4:
            showTrainingTips();
            break;
        case 5:
            launchGUI();
            break;
        case 0:
            std::cout << "🎓 Thank you for exploring braiNNlet!\n";
            std::cout << "For the full experience, launch the GUI application.\n\n";
            return 0;
        default:
            std::cout << "❌ Invalid choice. Please enter 0-5.\n\n";
            continue;
        }

        std::cout << "Press Enter to continue...";
        std::cin.get();
        showBanner(); // Clear and show banner again
    }

    return 0;
}
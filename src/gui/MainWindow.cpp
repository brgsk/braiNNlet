#include "MainWindow.hpp"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QMessageBox>
#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QTimer>
#include <QGroupBox>
#include <QListWidget>
#include <QInputDialog>

namespace gui
{

    MainWindow::MainWindow(QWidget *parent)
        : QMainWindow(parent), m_centralWidget(nullptr), m_plotWidget(nullptr), m_datasetCombo(nullptr), m_layersList(nullptr), m_addLayerButton(nullptr), m_removeLayerButton(nullptr), m_editLayerButton(nullptr), m_epochsSpinBox(nullptr), m_batchSizeSpinBox(nullptr), m_learningRateSpinBox(nullptr), m_trainButton(nullptr), m_stopButton(nullptr), m_progressBar(nullptr), m_logOutput(nullptr), m_statusLabel(nullptr), m_accuracyLabel(nullptr), m_network(std::make_unique<nn::Network>()), m_dataset(nullptr), m_trainer(nullptr), m_isTraining(false), m_updateTimer(new QTimer(this)), m_currentTrainAccuracy(0.0), m_currentValAccuracy(0.0), m_totalSteps(0), m_currentStep(0)
    {
        setWindowTitle("iNNsight - Neural Network Training");
        setFixedSize(1200, 800);

        // Modern dark theme styling
        setStyleSheet(R"(
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: #3c3c3c;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #ffffff;
        }
        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #666666;
            color: #999999;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #404040;
            border: 1px solid #666666;
            border-radius: 4px;
            padding: 4px;
            color: #ffffff;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid #ffffff;
        }
        QListWidget {
            background-color: #404040;
            border: 1px solid #666666;
            border-radius: 4px;
            alternate-background-color: #4a4a4a;
        }
        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #555555;
        }
        QListWidget::item:selected {
            background-color: #0078d4;
        }
        QTextEdit {
            background-color: #1e1e1e;
            border: 1px solid #666666;
            border-radius: 4px;
            color: #ffffff;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        QProgressBar {
            border: 1px solid #666666;
            border-radius: 4px;
            text-align: center;
            background-color: #404040;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
        QLabel {
            color: #ffffff;
        }
    )");

        // Initialize with a default layer configuration
        m_layerConfigs.push_back(LayerConfig(128, "ReLU"));
        m_layerConfigs.push_back(LayerConfig(64, "ReLU"));

        setupUI();
        updateLayersList();

        connect(m_updateTimer, &QTimer::timeout, this, &MainWindow::updateProgress);

        m_logOutput->append("Ready. Load dataset and configure network layers to start training.");
    }

    MainWindow::~MainWindow() = default;

    void MainWindow::setupUI()
    {
        m_centralWidget = new QWidget;
        setCentralWidget(m_centralWidget);

        auto *mainLayout = new QHBoxLayout(m_centralWidget);
        mainLayout->setSpacing(20);

        // Left panel for controls
        auto *leftPanel = new QWidget;
        leftPanel->setFixedWidth(450);
        auto *leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setSpacing(15);

        // Controls section with organized groups
        auto *controlsLayout = new QVBoxLayout;

        // Dataset group
        auto *datasetGroup = new QGroupBox("Dataset");
        auto *datasetLayout = new QGridLayout(datasetGroup);

        datasetLayout->addWidget(new QLabel("Dataset:"), 0, 0);
        m_datasetCombo = new QComboBox;
        m_datasetCombo->addItem("MNIST");
        datasetLayout->addWidget(m_datasetCombo, 0, 1);

        auto *loadButton = new QPushButton("Load Dataset");
        connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadData);
        datasetLayout->addWidget(loadButton, 1, 0, 1, 2);

        controlsLayout->addWidget(datasetGroup);

        // Network architecture group
        auto *networkGroup = new QGroupBox("Network Architecture");
        auto *networkLayout = new QVBoxLayout(networkGroup);

        networkLayout->addWidget(new QLabel("Hidden Layers:"));

        m_layersList = new QListWidget;
        m_layersList->setMaximumHeight(120);
        m_layersList->setSelectionMode(QAbstractItemView::SingleSelection);
        networkLayout->addWidget(m_layersList);

        auto *layerButtonsLayout = new QHBoxLayout;

        m_addLayerButton = new QPushButton("Add Layer");
        connect(m_addLayerButton, &QPushButton::clicked, this, &MainWindow::onAddLayer);
        layerButtonsLayout->addWidget(m_addLayerButton);

        m_editLayerButton = new QPushButton("Edit Layer");
        connect(m_editLayerButton, &QPushButton::clicked, this, &MainWindow::onEditLayer);
        layerButtonsLayout->addWidget(m_editLayerButton);

        m_removeLayerButton = new QPushButton("Remove Layer");
        connect(m_removeLayerButton, &QPushButton::clicked, this, &MainWindow::onRemoveLayer);
        layerButtonsLayout->addWidget(m_removeLayerButton);

        networkLayout->addLayout(layerButtonsLayout);

        controlsLayout->addWidget(networkGroup);

        // Training parameters group
        auto *trainingGroup = new QGroupBox("Training Parameters");
        auto *trainingLayout = new QGridLayout(trainingGroup);

        trainingLayout->addWidget(new QLabel("Epochs:"), 0, 0);
        m_epochsSpinBox = new QSpinBox;
        m_epochsSpinBox->setRange(1, 100);
        m_epochsSpinBox->setValue(10);
        trainingLayout->addWidget(m_epochsSpinBox, 0, 1);

        trainingLayout->addWidget(new QLabel("Batch Size:"), 1, 0);
        m_batchSizeSpinBox = new QSpinBox;
        m_batchSizeSpinBox->setRange(16, 512);
        m_batchSizeSpinBox->setValue(64);
        trainingLayout->addWidget(m_batchSizeSpinBox, 1, 1);

        trainingLayout->addWidget(new QLabel("Learning Rate:"), 2, 0);
        m_learningRateSpinBox = new QDoubleSpinBox;
        m_learningRateSpinBox->setRange(0.0001, 1.0);
        m_learningRateSpinBox->setDecimals(4);
        m_learningRateSpinBox->setSingleStep(0.001);
        m_learningRateSpinBox->setValue(0.01);
        trainingLayout->addWidget(m_learningRateSpinBox, 2, 1);

        // Training action buttons
        m_trainButton = new QPushButton("Train Network");
        connect(m_trainButton, &QPushButton::clicked, this, &MainWindow::onTrain);
        trainingLayout->addWidget(m_trainButton, 3, 0);

        m_stopButton = new QPushButton("Stop Training");
        m_stopButton->setEnabled(false);
        connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::onStop);
        trainingLayout->addWidget(m_stopButton, 3, 1);

        controlsLayout->addWidget(trainingGroup);

        // Progress bar
        m_progressBar = new QProgressBar;
        m_progressBar->setVisible(false);
        controlsLayout->addWidget(m_progressBar);

        // Status label
        m_statusLabel = new QLabel("Status: Ready");
        controlsLayout->addWidget(m_statusLabel);

        controlsLayout->addStretch(); // Push everything up

        leftLayout->addLayout(controlsLayout);
        mainLayout->addWidget(leftPanel);

        // Right panel for plot and log
        auto *rightPanel = new QWidget;
        auto *rightLayout = new QVBoxLayout(rightPanel);

        // Plot widget for training progress
        m_plotWidget = new PlotWidget;
        m_plotWidget->setTitle("Loss Progress");
        m_plotWidget->setMinimumHeight(400);
        m_plotWidget->setStyleSheet("background-color: #2b2b2b; border: 1px solid #666666; border-radius: 8px;");
        m_plotWidget->addDataSeries("Training Loss", QColor("#00a8cd"));
        m_plotWidget->addDataSeries("Validation Loss", QColor("#ee5fbb"));
        rightLayout->addWidget(m_plotWidget);

        // Accuracy display
        m_accuracyLabel = new QLabel("Training accuracy: --%, Validation accuracy: --%");
        m_accuracyLabel->setStyleSheet("font-weight: bold; font-size: 14px; color: #4ecdc4; padding: 10px; background-color: #3c3c3c; border-radius: 6px; margin: 5px 0px;");
        m_accuracyLabel->setAlignment(Qt::AlignCenter);
        rightLayout->addWidget(m_accuracyLabel);

        // Log output
        auto *logLabel = new QLabel("Training Log:");
        logLabel->setStyleSheet("font-weight: bold; margin-top: 10px;");
        rightLayout->addWidget(logLabel);

        m_logOutput = new QTextEdit;
        m_logOutput->setFixedHeight(200);
        m_logOutput->setReadOnly(true);
        rightLayout->addWidget(m_logOutput);

        mainLayout->addWidget(rightPanel);
    }

    void MainWindow::updateLayersList()
    {
        m_layersList->clear();

        for (size_t i = 0; i < m_layerConfigs.size(); ++i)
        {
            const auto &layer = m_layerConfigs[i];
            QString text = QString("Layer %1: %2 neurons, %3 activation")
                               .arg(i + 1)
                               .arg(layer.neurons)
                               .arg(layer.activation);
            m_layersList->addItem(text);
        }

        // Update button states
        m_removeLayerButton->setEnabled(!m_layerConfigs.empty());
        m_editLayerButton->setEnabled(!m_layerConfigs.empty());
    }

    void MainWindow::onAddLayer()
    {
        bool neuronsOk, activationOk;

        int neurons = QInputDialog::getInt(this, "Add Layer", "Number of neurons:", 64, 1, 1024, 1, &neuronsOk);
        if (!neuronsOk)
            return;

        QStringList activations = {"ReLU", "Sigmoid", "Tanh"};
        QString activation = QInputDialog::getItem(this, "Add Layer", "Activation function:",
                                                   activations, 0, false, &activationOk);
        if (!activationOk)
            return;

        m_layerConfigs.push_back(LayerConfig(neurons, activation));
        updateLayersList();

        m_logOutput->append(QString("➕ Added layer: %1 neurons, %2 activation").arg(neurons).arg(activation));
    }

    void MainWindow::onRemoveLayer()
    {
        if (m_layerConfigs.empty())
            return;

        int currentRow = m_layersList->currentRow();
        if (currentRow < 0)
            currentRow = m_layerConfigs.size() - 1; // Remove last if none selected

        if (currentRow >= 0 && currentRow < static_cast<int>(m_layerConfigs.size()))
        {
            const auto &removedLayer = m_layerConfigs[currentRow];
            m_logOutput->append(QString("➖ Removed layer %1: %2 neurons, %3 activation")
                                    .arg(currentRow + 1)
                                    .arg(removedLayer.neurons)
                                    .arg(removedLayer.activation));

            m_layerConfigs.erase(m_layerConfigs.begin() + currentRow);
            updateLayersList();
        }
    }

    void MainWindow::onEditLayer()
    {
        int currentRow = m_layersList->currentRow();
        if (currentRow < 0 || currentRow >= static_cast<int>(m_layerConfigs.size()))
        {
            QMessageBox::information(this, "Edit Layer", "Please select a layer to edit.");
            return;
        }

        auto &layer = m_layerConfigs[currentRow];

        bool neuronsOk, activationOk;

        int neurons = QInputDialog::getInt(this, "Edit Layer", "Number of neurons:",
                                           layer.neurons, 1, 1024, 1, &neuronsOk);
        if (!neuronsOk)
            return;

        QStringList activations = {"ReLU", "Sigmoid", "Tanh"};
        int currentIndex = activations.indexOf(layer.activation);
        if (currentIndex < 0)
            currentIndex = 0;

        QString activation = QInputDialog::getItem(this, "Edit Layer", "Activation function:",
                                                   activations, currentIndex, false, &activationOk);
        if (!activationOk)
            return;

        m_logOutput->append(QString("✏️ Edited layer %1: %2→%3 neurons, %4→%5 activation")
                                .arg(currentRow + 1)
                                .arg(layer.neurons)
                                .arg(neurons)
                                .arg(layer.activation)
                                .arg(activation));

        layer.neurons = neurons;
        layer.activation = activation;
        updateLayersList();
    }

    void MainWindow::onLoadData()
    {
        QString dataset = m_datasetCombo->currentText();

        if (dataset == "MNIST")
        {
            m_dataset = std::make_unique<data::MnistDataset>();
            if (m_dataset->load(""))
            {
                m_logOutput->append("✓ MNIST dataset loaded successfully");
                m_logOutput->append(QString("  Samples: %1, Input size: %2, Classes: %3")
                                        .arg(m_dataset->size())
                                        .arg(m_dataset->input_size())
                                        .arg(m_dataset->output_size()));
                updateStatus();
            }
            else
            {
                m_logOutput->append("✗ Failed to load MNIST dataset");
            }
        }
    }

    void MainWindow::onTrain()
    {
        if (!m_dataset)
        {
            QMessageBox::warning(this, "Error", "Please load a dataset first");
            return;
        }

        if (m_layerConfigs.empty())
        {
            QMessageBox::warning(this, "Error", "Please add at least one hidden layer");
            return;
        }

        m_isTraining = true;
        m_trainButton->setEnabled(false);
        m_stopButton->setEnabled(true);
        m_progressBar->setVisible(true);
        m_progressBar->setValue(0);

        // Clear previous plot data
        m_plotWidget->clearData();

        // Reset accuracy display and step counter
        m_currentTrainAccuracy = 0.0;
        m_currentValAccuracy = 0.0;
        m_currentStep = 0;
        m_accuracyLabel->setText("Accuracies: Training: --%, Validation: --% (Step 0/0)");

        trainNetwork();
    }

    void MainWindow::onStop()
    {
        if (m_trainer)
        {
            m_trainer->stop();
        }
        m_isTraining = false;
        m_trainButton->setEnabled(true);
        m_stopButton->setEnabled(false);
        m_progressBar->setVisible(false);
        m_updateTimer->stop();
        m_logOutput->append("⏹ Training stopped");
    }

    void MainWindow::trainNetwork()
    {
        // Build network with user-configured layers
        m_network = std::make_unique<nn::Network>();
        m_network->set_loss_function(nn::LossType::CrossEntropy);

        // Add user-configured hidden layers
        for (const auto &layerConfig : m_layerConfigs)
        {
            nn::ActivationType activation = nn::ActivationType::ReLU;
            if (layerConfig.activation == "Sigmoid")
                activation = nn::ActivationType::Sigmoid;
            else if (layerConfig.activation == "Tanh")
                activation = nn::ActivationType::Tanh;

            m_network->add_layer(layerConfig.neurons, activation);
        }

        // Add output layer (always linear for classification)
        m_network->add_layer(m_dataset->output_size(), nn::ActivationType::Linear);

        // Log network architecture
        m_logOutput->append("🧠 Created network architecture:");
        for (size_t i = 0; i < m_layerConfigs.size(); ++i)
        {
            const auto &layer = m_layerConfigs[i];
            m_logOutput->append(QString("  Hidden Layer %1: %2 neurons, %3 activation")
                                    .arg(i + 1)
                                    .arg(layer.neurons)
                                    .arg(layer.activation));
        }
        m_logOutput->append(QString("  Output Layer: %1 neurons, Linear activation").arg(m_dataset->output_size()));

        // Create trainer with user-specified parameters
        m_trainer = std::make_unique<training::Trainer>(*m_network, *m_dataset);

        training::TrainingConfig config;
        config.epochs = m_epochsSpinBox->value();
        config.batch_size = m_batchSizeSpinBox->value();
        config.learning_rate = m_learningRateSpinBox->value();
        config.validation_split = 0.2;

        // Calculate total training steps
        int trainingSamples = static_cast<int>(m_dataset->size() * (1.0 - config.validation_split));
        int batchesPerEpoch = (trainingSamples + config.batch_size - 1) / config.batch_size;
        m_totalSteps = config.epochs * batchesPerEpoch;
        m_currentStep = 0;

        // Set total steps on plot widget for proper X-axis scaling
        m_plotWidget->setTotalSteps(m_totalSteps);

        m_progressBar->setMaximum(config.epochs);

        m_logOutput->append(QString("⚙️ Training config: %1 epochs, batch size %2, learning rate %3")
                                .arg(config.epochs)
                                .arg(config.batch_size)
                                .arg(config.learning_rate, 0, 'f', 4));
        m_logOutput->append(QString("📊 Total training steps: %1 (%2 batches/epoch × %3 epochs)")
                                .arg(m_totalSteps)
                                .arg(batchesPerEpoch)
                                .arg(config.epochs));

        // Set up callbacks

        // Batch-level callback for real-time plot updates
        m_trainer->set_on_batch_end([this](int batch, const training::TrainingMetrics &metrics)
                                    {
            // Increment current step counter
            m_currentStep++;
            
            // Update plot with batch-level training loss only
            m_plotWidget->addDataPoint("Training Loss", metrics.loss);
            
            // Update current training accuracy
            m_currentTrainAccuracy = metrics.accuracy;
            m_accuracyLabel->setText(QString("Accuracies: Training: %1%, Validation: %2% (Step %3/%4)")
                                   .arg(m_currentTrainAccuracy * 100, 0, 'f', 1)
                                   .arg(m_currentValAccuracy * 100, 0, 'f', 1)
                                   .arg(m_currentStep)
                                   .arg(m_totalSteps));
            
            // Force immediate plot update
            m_plotWidget->update();
            QApplication::processEvents(); });

        // Epoch-level callback for validation metrics and logging
        m_trainer->set_on_epoch_end([this](int epoch, const training::TrainingMetrics &train_metrics,
                                           const training::TrainingMetrics &val_metrics)
                                    {
        m_progressBar->setValue(epoch + 1);
        
        double train_loss = train_metrics.loss;
        double train_acc = train_metrics.accuracy;
        double val_loss = val_metrics.loss;
        double val_acc = val_metrics.accuracy;
        
        // Add validation loss to plot (only available at epoch level)
        m_plotWidget->addDataPoint("Validation Loss", val_loss);
        
        // Update validation accuracy
        m_currentValAccuracy = val_acc;
        m_accuracyLabel->setText(QString("Accuracies: Training: %1%, Validation: %2% (Step %3/%4)")
                               .arg(m_currentTrainAccuracy * 100, 0, 'f', 1)
                               .arg(m_currentValAccuracy * 100, 0, 'f', 1)
                               .arg(m_currentStep)
                               .arg(m_totalSteps));
        
        // Force immediate plot update
        m_plotWidget->update();
        QApplication::processEvents();
        
        m_logOutput->append(QString("📊 Epoch %1: Loss=%2, Acc=%3%, Val_Loss=%4, Val_Acc=%5%")
                           .arg(epoch + 1)
                           .arg(train_loss, 0, 'f', 3)
                           .arg(train_acc * 100, 0, 'f', 1)
                           .arg(val_loss, 0, 'f', 3)
                           .arg(val_acc * 100, 0, 'f', 1)); });

        m_trainer->set_on_training_end([this](const std::vector<training::TrainingMetrics> &history)
                                       {
        m_isTraining = false;
        m_trainButton->setEnabled(true);
        m_stopButton->setEnabled(false);
        m_progressBar->setVisible(false);
        m_updateTimer->stop();
        
        if (!history.empty()) {
            const auto& final_metrics = history.back();
            m_logOutput->append(QString("🎉 Training completed! Final accuracy: %1%")
                               .arg(final_metrics.accuracy * 100, 0, 'f', 1));
        } });

        m_updateTimer->start(100);
        m_trainer->train(config);
    }

    void MainWindow::updateProgress()
    {
        m_plotWidget->update();
        updateStatus();
    }

    void MainWindow::updateStatus()
    {
        if (m_isTraining)
        {
            m_statusLabel->setText("Status: Training in progress...");
        }
        else if (m_dataset)
        {
            m_statusLabel->setText(QString("Status: Dataset loaded (%1 samples), %2 hidden layers configured")
                                       .arg(m_dataset->size())
                                       .arg(m_layerConfigs.size()));
        }
        else
        {
            m_statusLabel->setText("Status: Ready");
        }
    }

} // namespace gui
#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QTimer>
#include <QListWidget>
#include <QListWidgetItem>
#include <memory>

#include "../core/nn/Network.hpp"
#include "../core/data/MnistLoader.hpp"
#include "../core/training/Trainer.hpp"

#include "PlotWidget.hpp"

namespace gui
{

    struct LayerConfig
    {
        int neurons;
        QString activation;

        LayerConfig(int n = 64, const QString &act = "ReLU") : neurons(n), activation(act) {}
    };

    class MainWindow : public QMainWindow
    {
        Q_OBJECT

    public:
        explicit MainWindow(QWidget *parent = nullptr);
        ~MainWindow() override;

    private slots:
        void onLoadData();
        void onTrain();
        void onStop();
        void updateProgress();
        void onAddLayer();
        void onRemoveLayer();
        void onEditLayer();

    private:
        void setupUI();
        void trainNetwork();
        void updateStatus();
        void updateLayersList();

        // Essential UI components
        QWidget *m_centralWidget;
        PlotWidget *m_plotWidget;

        // Dataset controls
        QComboBox *m_datasetCombo;

        // Network architecture controls
        QListWidget *m_layersList;
        QPushButton *m_addLayerButton;
        QPushButton *m_removeLayerButton;
        QPushButton *m_editLayerButton;

        // Training parameter controls
        QSpinBox *m_epochsSpinBox;
        QSpinBox *m_batchSizeSpinBox;
        QDoubleSpinBox *m_learningRateSpinBox;

        // Action controls
        QPushButton *m_trainButton;
        QPushButton *m_stopButton;
        QProgressBar *m_progressBar;
        QTextEdit *m_logOutput;
        QLabel *m_statusLabel;
        QLabel *m_accuracyLabel;

        // Core components
        std::unique_ptr<nn::Network> m_network;
        std::unique_ptr<data::Dataset> m_dataset;
        std::unique_ptr<training::Trainer> m_trainer;

        // Layer configuration
        std::vector<LayerConfig> m_layerConfigs;

        bool m_isTraining;
        QTimer *m_updateTimer;

        // Current accuracy tracking
        double m_currentTrainAccuracy;
        double m_currentValAccuracy;

        // Training progress tracking
        int m_totalSteps;
        int m_currentStep;
        int m_batchesPerEpoch;
    };

} // namespace gui
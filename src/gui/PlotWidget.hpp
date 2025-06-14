#pragma once

#include <QWidget>
#include <QPainter>
#include <vector>
#include <deque>

namespace gui
{

    struct DataPoint
    {
        double value;
        int step;

        DataPoint(double v, int s) : value(v), step(s) {}
    };

    struct PlotData
    {
        std::vector<DataPoint> points;
        QColor color;
        QString name;

        PlotData(const QString &n, const QColor &c)
            : color(c), name(n) {}
    };

    class PlotWidget : public QWidget
    {
        Q_OBJECT

    public:
        explicit PlotWidget(QWidget *parent = nullptr);
        ~PlotWidget() override = default;

        void addDataSeries(const QString &name, const QColor &color);
        void addDataPoint(const QString &series_name, double value);
        void addDataPointAtStep(const QString &series_name, double value, int step);
        void setTitle(const QString &title);
        void clearData();
        void setTotalSteps(int totalSteps);

    protected:
        void paintEvent(QPaintEvent *event) override;

    private:
        void drawData(QPainter &painter);
        void updateDataRange();

        std::vector<PlotData> m_data_series;
        QString m_title;
        double m_min_y, m_max_y;
        int m_totalSteps;

        static constexpr int MARGIN = 40;
    };

} // namespace gui
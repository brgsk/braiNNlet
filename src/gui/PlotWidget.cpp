#include "PlotWidget.hpp"
#include <QPaintEvent>
#include <QPainter>
#include <QFontMetrics>
#include <algorithm>
#include <limits>

namespace gui
{

    PlotWidget::PlotWidget(QWidget *parent)
        : QWidget(parent), m_min_y(0.0), m_max_y(1.0), m_totalSteps(0)
    {
        setMinimumSize(300, 200);
        setAutoFillBackground(true);

        // Set dark background
        QPalette pal = palette();
        pal.setColor(QPalette::Window, QColor("#2b2b2b"));
        setPalette(pal);
    }

    void PlotWidget::addDataSeries(const QString &name, const QColor &color)
    {
        m_data_series.emplace_back(name, color);
    }

    void PlotWidget::addDataPoint(const QString &series_name, double value)
    {
        for (auto &series : m_data_series)
        {
            if (series.name == series_name)
            {
                // Add point with auto-incrementing step (for sequential data like training loss)
                int step = series.points.size();
                series.points.emplace_back(value, step);
                updateDataRange();
                update();
                break;
            }
        }
    }

    void PlotWidget::addDataPointAtStep(const QString &series_name, double value, int step)
    {
        for (auto &series : m_data_series)
        {
            if (series.name == series_name)
            {
                // Add point at specific step (for validation data at epoch boundaries)
                series.points.emplace_back(value, step);
                updateDataRange();
                update();
                break;
            }
        }
    }

    void PlotWidget::setTitle(const QString &title)
    {
        m_title = title;
        update();
    }

    void PlotWidget::clearData()
    {
        for (auto &series : m_data_series)
        {
            series.points.clear();
        }
        update();
    }

    void PlotWidget::setTotalSteps(int totalSteps)
    {
        m_totalSteps = totalSteps;
        update();
    }

    void PlotWidget::paintEvent(QPaintEvent *event)
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        // Fill background
        painter.fillRect(rect(), QColor("#2b2b2b"));

        // Draw title
        if (!m_title.isEmpty())
        {
            painter.setPen(Qt::white);
            QFont titleFont = painter.font();
            titleFont.setBold(true);
            titleFont.setPointSize(12);
            painter.setFont(titleFont);
            painter.drawText(rect().adjusted(0, 5, 0, 0), Qt::AlignTop | Qt::AlignHCenter, m_title);
        }

        drawData(painter);
    }

    void PlotWidget::drawData(QPainter &painter)
    {
        if (m_data_series.empty())
            return;

        // Increase margins to make room for axes labels
        QRect plotRect = rect().adjusted(60, 40, -20, -40);

        // Draw plot border
        painter.setPen(QPen(QColor("#666666"), 1));
        painter.drawRect(plotRect);

        // Draw grid lines and Y-axis labels
        painter.setPen(QPen(QColor("#444444"), 1, Qt::DashLine));
        painter.setFont(QFont("Arial", 9));

        for (int i = 0; i <= 5; ++i)
        {
            int y = plotRect.top() + (plotRect.height() * i) / 5;
            painter.drawLine(plotRect.left(), y, plotRect.right(), y);

            // Y-axis labels (inverted because Qt coordinates are top-down)
            double value = m_max_y - (m_max_y - m_min_y) * i / 5.0;
            painter.setPen(Qt::white);
            painter.drawText(QRect(5, y - 10, 50, 20), Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(value, 'f', 2));
            painter.setPen(QPen(QColor("#444444"), 1, Qt::DashLine));
        }

        // Draw vertical grid lines and X-axis labels
        if (m_totalSteps > 0)
        {
            for (int i = 0; i <= 5; ++i)
            {
                int x = plotRect.left() + (plotRect.width() * i) / 5;
                painter.drawLine(x, plotRect.top(), x, plotRect.bottom());

                // X-axis labels (showing step numbers based on total steps)
                int stepNumber = (m_totalSteps - 1) * i / 5;
                painter.setPen(Qt::white);
                painter.drawText(QRect(x - 20, plotRect.bottom() + 5, 40, 20),
                                 Qt::AlignCenter, QString::number(stepNumber));
                painter.setPen(QPen(QColor("#444444"), 1, Qt::DashLine));
            }
        }

        // Draw axis labels
        painter.setPen(Qt::white);
        QFont axisFont("Arial", 10, QFont::Bold);
        painter.setFont(axisFont);

        // Y-axis label
        painter.save();
        painter.translate(15, plotRect.center().y());
        painter.rotate(-90);
        painter.drawText(QRect(-50, -10, 100, 20), Qt::AlignCenter, "Value");
        painter.restore();

        // X-axis label
        painter.drawText(QRect(plotRect.left(), plotRect.bottom() + 25, plotRect.width(), 20),
                         Qt::AlignCenter, "Batch/Epoch");

        for (const auto &series : m_data_series)
        {
            if (series.points.empty())
                continue;

            painter.setPen(QPen(series.color, 3));

            QVector<QPointF> plotPoints;

            // Use total steps for X-axis scaling
            double totalRange = m_totalSteps > 0 ? m_totalSteps - 1 : 1;

            for (const auto &dataPoint : series.points)
            {
                // Position points based on their actual step number
                double x_ratio = dataPoint.step / totalRange;
                double x = plotRect.left() + x_ratio * plotRect.width();
                double y_norm = (dataPoint.value - m_min_y) / (m_max_y - m_min_y);
                double y = plotRect.bottom() - y_norm * plotRect.height();
                plotPoints.append(QPointF(x, y));
            }

            if (plotPoints.size() > 1)
            {
                painter.drawPolyline(plotPoints);
            }
            else if (plotPoints.size() == 1)
            {
                // Draw single point as a small circle
                painter.setBrush(series.color);
                painter.drawEllipse(plotPoints[0], 3, 3);
            }
        }

        // Draw legend in top-right corner outside plot area
        int legendWidth = 180;
        int legendHeight = m_data_series.size() * 22 + 10;
        int legendX = rect().width() - legendWidth - 10;
        int legendY = 10;

        // Legend background
        painter.setBrush(QColor(0, 0, 0, 150));
        painter.setPen(QPen(QColor("#666666"), 1));
        painter.drawRoundedRect(legendX, legendY, legendWidth, legendHeight, 8, 8);

        // Legend items
        int itemY = legendY + 15;
        for (const auto &series : m_data_series)
        {
            // Draw color indicator (line sample)
            painter.setPen(QPen(series.color, 3));
            painter.drawLine(legendX + 10, itemY, legendX + 25, itemY);

            // Draw text
            painter.setPen(Qt::white);
            QFont legendFont("Arial", 9);
            painter.setFont(legendFont);
            painter.drawText(legendX + 35, itemY - 6, legendWidth - 45, 20,
                             Qt::AlignLeft | Qt::AlignVCenter, series.name);
            itemY += 22;
        }
    }

    void PlotWidget::updateDataRange()
    {
        if (m_data_series.empty())
            return;

        m_min_y = std::numeric_limits<double>::max();
        m_max_y = std::numeric_limits<double>::lowest();

        for (const auto &series : m_data_series)
        {
            for (const auto &dataPoint : series.points)
            {
                m_min_y = std::min(m_min_y, dataPoint.value);
                m_max_y = std::max(m_max_y, dataPoint.value);
            }
        }

        // Add some padding
        double range = m_max_y - m_min_y;
        if (range < 1e-6)
            range = 1.0;
        m_min_y -= range * 0.1;
        m_max_y += range * 0.1;
    }

} // namespace gui
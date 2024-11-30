class ChartManager {
    constructor() {
        // Initialize charts with fixed aspect ratio and data window size
        this.maxDataPoints = 100; // Limit the number of visible points
        this.lossChart1 = this.createLineChart('lossChart1', 'Loss - Model 1', ['Training Loss', 'Test Loss']);
        this.lossChart2 = this.createLineChart('lossChart2', 'Loss - Model 2', ['Training Loss', 'Test Loss']);
        this.accuracyChart1 = this.createLineChart('accuracyChart1', 'Accuracy - Model 1', ['Training Accuracy', 'Validation Accuracy']);
        this.accuracyChart2 = this.createLineChart('accuracyChart2', 'Accuracy - Model 2', ['Training Accuracy', 'Validation Accuracy']);
    }

    createLineChart(canvasId, title, labels) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: labels.map((label, index) => ({
                    label: label,
                    data: [],
                    borderColor: index === 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)',
                    backgroundColor: index === 0 ? 'rgba(75, 192, 192, 0.1)' : 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
                    pointRadius: 2,
                    pointHoverRadius: 5
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2, // Fixed aspect ratio
                plugins: {
                    title: {
                        display: true,
                        text: title,
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 10
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: title.includes('Loss') ? 'Batch' : 'Epoch'
                        },
                        ticks: {
                            maxTicksLimit: 10 // Limit number of x-axis ticks
                        }
                    },
                    y: {
                        display: true,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: title.includes('Loss') ? 'Loss' : 'Accuracy (%)'
                        },
                        suggestedMax: title.includes('Loss') ? undefined : 100
                    }
                },
                animation: {
                    duration: 0 // Disable animations for better performance
                }
            }
        });
    }

    updateCharts(data) {
        // Update loss charts
        if (data.batch_metrics) {
            // Update Model 1 loss chart with windowing
            this.updateChartData(this.lossChart1, 
                data.batch_metrics.model1.batch_numbers,
                data.batch_metrics.model1.train_loss,
                data.batch_metrics.model1.test_loss
            );

            // Update Model 2 loss chart with windowing
            this.updateChartData(this.lossChart2,
                data.batch_metrics.model2.batch_numbers,
                data.batch_metrics.model2.train_loss,
                data.batch_metrics.model2.test_loss
            );
        }

        // Update accuracy charts
        if (data.epoch_metrics) {
            // Update Model 1 accuracy chart
            this.updateChartData(this.accuracyChart1,
                data.epoch_metrics.model1.epochs,
                data.epoch_metrics.model1.train_acc,
                data.epoch_metrics.model1.val_acc
            );

            // Update Model 2 accuracy chart
            this.updateChartData(this.accuracyChart2,
                data.epoch_metrics.model2.epochs,
                data.epoch_metrics.model2.train_acc,
                data.epoch_metrics.model2.val_acc
            );
        }
    }

    updateChartData(chart, labels, dataset1, dataset2) {
        // Apply windowing if data exceeds maxDataPoints
        if (labels.length > this.maxDataPoints) {
            const startIdx = labels.length - this.maxDataPoints;
            chart.data.labels = labels.slice(startIdx);
            chart.data.datasets[0].data = dataset1.slice(startIdx);
            chart.data.datasets[1].data = dataset2.slice(startIdx);
        } else {
            chart.data.labels = labels;
            chart.data.datasets[0].data = dataset1;
            chart.data.datasets[1].data = dataset2;
        }
        chart.update('none');
    }

    resetCharts() {
        [this.lossChart1, this.lossChart2].forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => dataset.data = []);
            chart.update();
        });

        [this.accuracyChart1, this.accuracyChart2].forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => dataset.data = []);
            chart.update();
        });
    }
} 
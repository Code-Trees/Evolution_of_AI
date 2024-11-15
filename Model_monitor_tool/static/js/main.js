class Dashboard {
    constructor() {
        this.chartManager = new ChartManager();
        this.trainingInterval = null;
        this.logsInterval = null;
        this.isTrainingComplete = false;
        this.autoUpdateInterval = null;
        this.isAutoUpdateEnabled = true;
        this.initializeEventListeners();
        this.initializeSamplePredictions();
    }

    initializeEventListeners() {
        document.getElementById('trainButton').addEventListener('click', () => this.startTraining());
        document.getElementById('stopButton').addEventListener('click', () => this.stopTraining('Stopped by user'));
        document.getElementById('resetButton').addEventListener('click', () => this.resetDashboard());
        document.querySelector('.clear-logs').addEventListener('click', () => {
            document.getElementById('terminal').innerHTML = '';
        });
        document.getElementById('updateSamples').addEventListener('click', () => this.updateSamples());
        document.getElementById('addAugmentation').addEventListener('click', function() {
            const augmentationSelect = document.getElementById('augmentation');
            const selectedAugmentation = augmentationSelect.value;
            const paramsDiv = document.getElementById('augmentation-params');

            // Clear previous parameters
            paramsDiv.innerHTML = '';

            if (selectedAugmentation === 'rotation') {
                paramsDiv.innerHTML += `
                    <label for="rotationAngle">Rotation Angle (degrees):</label>
                    <input type="number" id="rotationAngle" value="10" class="augmentation-param">
                `;
            } else if (selectedAugmentation === 'translation') {
                paramsDiv.innerHTML += `
                    <label for="translationX">Translation X (%):</label>
                    <input type="number" id="translationX" value="0.1" class="augmentation-param">
                    <label for="translationY">Translation Y (%):</label>
                    <input type="number" id="translationY" value="0.1" class="augmentation-param">
                `;
            } else if (selectedAugmentation === 'scaling') {
                paramsDiv.innerHTML += `
                    <label for="scalingFactor">Scaling Factor (0.8 to 1.0):</label>
                    <input type="number" id="scalingFactor" step="0.01" value="0.8" class="augmentation-param">
                `;
            } else if (selectedAugmentation === 'flipping') {
                // No parameters needed for flipping
                paramsDiv.innerHTML += `<p>No parameters needed for Flipping.</p>`;
            }
        });
    }

    initializeSamplePredictions() {
        // Set up auto-update toggle
        document.getElementById('toggleAutoUpdate').addEventListener('click', () => {
            this.isAutoUpdateEnabled = !this.isAutoUpdateEnabled;
            document.getElementById('autoUpdateStatus').textContent = 
                `Auto-update: ${this.isAutoUpdateEnabled ? 'ON' : 'OFF'}`;
            
            if (this.isAutoUpdateEnabled) {
                this.startAutoUpdate();
            } else {
                this.stopAutoUpdate();
            }
        });

        // Set up manual update button
        document.getElementById('updateSamples').addEventListener('click', () => {
            this.updateSamples();
        });

        // Start auto-update
        this.startAutoUpdate();
    }

    startAutoUpdate() {
        if (this.autoUpdateInterval) {
            clearInterval(this.autoUpdateInterval);
        }
        this.autoUpdateInterval = setInterval(() => {
            if (this.isAutoUpdateEnabled) {
                this.updateSamples();
            }
        }, 5000); // Update every 5 seconds
    }

    stopAutoUpdate() {
        if (this.autoUpdateInterval) {
            clearInterval(this.autoUpdateInterval);
            this.autoUpdateInterval = null;
        }
    }

    addLog(message) {
        const terminal = document.getElementById('terminal');
        const timestamp = new Date().toLocaleTimeString();
        terminal.innerHTML += `[${timestamp}] ${message}<br>`;
        terminal.scrollTop = terminal.scrollHeight;
    }

    async startTraining() {
        try {
            const trainButton = document.getElementById('trainButton');
            const stopButton = document.getElementById('stopButton');
            const epochInput = document.getElementById('epochNumber');
            const learningRateInput = document.getElementById('learningRate');
            const batchSizeInput = document.getElementById('batchSize');
            const optimizerSelect = document.getElementById('optimizer');

            // Get model configurations
            const model_configs = {
                model1: {
                    conv1: parseInt(document.getElementById('conv1_channels_1').value) || 16,
                    conv2: parseInt(document.getElementById('conv2_channels_1').value) || 32,
                    conv3: parseInt(document.getElementById('conv3_channels_1').value) || 64
                },
                model2: {
                    conv1: parseInt(document.getElementById('conv1_channels_2').value) || 16,
                    conv2: parseInt(document.getElementById('conv2_channels_2').value) || 32,
                    conv3: parseInt(document.getElementById('conv3_channels_2').value) || 64
                }
            };

            // Get training parameters
            const numEpochs = parseInt(epochInput.value) || 20;
            const learningRate = parseFloat(learningRateInput.value) || 0.01;
            const batchSize = parseInt(batchSizeInput.value) || 128;
            const selectedOptimizer = optimizerSelect.value || 'SGD';

            // Disable UI elements
            trainButton.disabled = true;
            stopButton.disabled = false;
            epochInput.disabled = true;
            trainButton.textContent = 'Training in Progress...';

            // Reset charts before starting new training
            this.chartManager.resetCharts();

            console.log('Sending training request...'); // Debug log

            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    num_epochs: numEpochs,
                    learning_rate: learningRate,
                    batch_size: batchSize,
                    optimizer: selectedOptimizer,
                    model_configs: model_configs
                })
            });

            const data = await response.json();
            console.log('Training response:', data); // Debug log

            if (data.success) {
                this.isTrainingComplete = false;
                this.addLog('Training started successfully');
                
                // Start polling for updates
                if (this.logsInterval) clearInterval(this.logsInterval);
                if (this.trainingInterval) clearInterval(this.trainingInterval);
                
                this.logsInterval = setInterval(() => this.fetchLogs(), 1000);
                this.trainingInterval = setInterval(() => {
                    this.updatePlots();
                    this.checkTrainingStatus();
                }, 1000);
            } else {
                this.handleTrainingStop(data.message || 'Training failed to start');
            }
        } catch (error) {
            console.error('Error:', error);
            this.handleTrainingStop('Server error: Could not start training');
        }
    }

    async stopTraining(reason = 'Stopped by user') {
        if (!this.isTrainingComplete) {
            try {
                const response = await fetch('/stop_training', {
                    method: 'POST'
                });
                const data = await response.json();
                if (data.success) {
                    this.handleTrainingStop(reason);
                }
            } catch (error) {
                console.error('Error stopping training:', error);
            }
        }
    }

    handleTrainingStop(reason) {
        // Clear intervals
        if (this.logsInterval) clearInterval(this.logsInterval);
        if (this.trainingInterval) clearInterval(this.trainingInterval);
        
        // Reset UI
        const trainButton = document.getElementById('trainButton');
        const stopButton = document.getElementById('stopButton');
        const epochInput = document.getElementById('epochNumber');
        
        trainButton.disabled = false;
        stopButton.disabled = true;
        epochInput.disabled = false;
        trainButton.textContent = 'Start Training';
        
        this.addLog(`Training stopped: ${reason}`);
        this.isTrainingComplete = true;
        
        // Update samples after training completes
        if (reason === 'Training completed successfully') {
            this.updateSamples();
        }
    }

    async checkTrainingStatus() {
        try {
            const response = await fetch('/training_status');
            const data = await response.json();
            
            if (!data.is_training && !this.isTrainingComplete) {
                if (data.current_epoch >= data.total_epochs) {
                    this.handleTrainingStop('Training completed successfully');
                } else if (data.current_epoch < data.total_epochs) {
                    this.handleTrainingStop('Training stopped by user');
                }
            }
        } catch (error) {
            console.error('Error checking training status:', error);
        }
    }

    async resetDashboard() {
        try {
            // Reset UI elements
            const trainButton = document.getElementById('trainButton');
            const stopButton = document.getElementById('stopButton');
            const epochInput = document.getElementById('epochNumber');
            
            // Clear intervals if they exist
            if (this.trainingInterval) clearInterval(this.trainingInterval);
            if (this.logsInterval) clearInterval(this.logsInterval);
            
            // Reset button states
            trainButton.disabled = false;
            trainButton.style.backgroundColor = '#4CAF50';
            trainButton.textContent = 'Start Training';
            stopButton.disabled = true;
            epochInput.disabled = false;
            epochInput.value = '20'; // Reset to default epochs
            
            // Clear terminal
            document.getElementById('terminal').innerHTML = '';
            this.addLog('Resetting dashboard...');
            
            // Reset charts
            this.chartManager.resetCharts();
            
            // Clear samples
            document.getElementById('samples').innerHTML = '';
            
            // Call server reset
            const response = await fetch('/reset', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.addLog('Dashboard reset complete');
                this.addLog('Ready for fresh training');
                this.isTrainingComplete = false;
            } else {
                this.addLog('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            this.addLog('Error resetting dashboard');
        }
    }

    async updatePlots() {
        try {
            const response = await fetch('/get_metrics');
            const data = await response.json();
            this.chartManager.updateCharts(data);
        } catch (error) {
            console.error('Error updating plots:', error);
        }
    }

    async fetchLogs() {
        try {
            const response = await fetch('/get_logs');
            const data = await response.json();
            data.logs.forEach(log => {
                // Remove any existing newline characters and add HTML line break
                const cleanLog = log.replace(/\n/g, '');
                this.addLog(cleanLog);
            });
        } catch (error) {
            console.error('Error fetching logs:', error);
        }
    }

    async updateSamples() {
        try {
            const response = await fetch('/get_samples');
            const data = await response.json();
            
            if (!data) return;

            ['model1', 'model2'].forEach(modelName => {
                const wrongContainer = document.getElementById(`${modelName}-wrong`);
                const correctContainer = document.getElementById(`${modelName}-correct`);

                // Update wrong predictions
                wrongContainer.innerHTML = data[modelName].wrong_samples.map(sample => `
                    <div class="sample-item">
                        <img src="data:image/png;base64,${sample.image}" 
                             alt="Prediction">
                        <div class="prediction-label wrong-prediction">
                            Pred: ${sample.predicted}<br>
                            Actual: ${sample.actual}
                        </div>
                    </div>
                `).join('');

                // Update correct predictions
                correctContainer.innerHTML = data[modelName].correct_samples.map(sample => `
                    <div class="sample-item">
                        <img src="data:image/png;base64,${sample.image}" 
                             alt="Prediction">
                        <div class="prediction-label correct-prediction">
                            Pred: ${sample.predicted}<br>
                            Actual: ${sample.actual}
                        </div>
                    </div>
                `).join('');
            });
        } catch (error) {
            console.error('Error updating samples:', error);
        }
    }

    async fetchHistograms() {
        try {
            const response = await fetch('/get_histograms');
            const data = await response.json();
            // Process and display histograms
            console.log(data);
        } catch (error) {
            console.error('Error fetching histograms:', error);
        }
    }

    async fetchEmbeddings() {
        try {
            const response = await fetch('/get_embeddings');
            const data = await response.json();
            // Process and display embeddings
            console.log(data);
        } catch (error) {
            console.error('Error fetching embeddings:', error);
        }
    }

    async fetchPerformanceStats() {
        try {
            const response = await fetch('/get_performance_stats');
            const data = await response.json();
            // Process and display performance stats
            console.log(data);
        } catch (error) {
            console.error('Error fetching performance stats:', error);
        }
    }

    async fetchHParams() {
        try {
            const response = await fetch('/get_hparams');
            const data = await response.json();
            // Process and display hyperparameters
            console.log(data);
        } catch (error) {
            console.error('Error fetching hyperparameters:', error);
        }
    }

    async updateVisualizations() {
        if (window.modelVisualizer) {
            await window.modelVisualizer.updateVisualizations();
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
}); 
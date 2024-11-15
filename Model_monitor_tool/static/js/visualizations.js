class ModelVisualizer {
    constructor() {
        this.architectureContainer = document.getElementById('model-architecture');
        this.activationsContainer = document.getElementById('layer-activations');
        this.initializeVisualizations();
    }

    async initializeVisualizations() {
        // Initialize the graphviz renderer
        this.graphviz = d3.select("#model-architecture").graphviz()
            .zoom(true)
            .fit(true);

        // Create activation charts container
        this.activationsChart = d3.select("#layer-activations")
            .append('div')
            .attr('class', 'activations-wrapper');

        // Initial load
        await this.updateVisualizations();
    }

    async updateVisualizations() {
        await Promise.all([
            this.updateModelArchitecture(),
            this.updateLayerActivations()
        ]);
    }

    async updateModelArchitecture() {
        try {
            const response = await fetch('/get_model_architecture');
            const data = await response.json();
            
            if (data.success) {
                // Render the DOT graph
                this.graphviz.renderDot(data.dot_data);
            } else {
                console.error('Failed to get model architecture:', data.error);
            }
        } catch (error) {
            console.error('Error updating model architecture:', error);
        }
    }

    async updateLayerActivations() {
        try {
            const response = await fetch('/get_layer_activations');
            const data = await response.json();
            
            if (data.success) {
                // Clear previous content
                this.activationsChart.html('');
                
                // Create activation visualizations
                Object.entries(data.activations).forEach(([layerName, stats]) => {
                    const layerDiv = this.activationsChart
                        .append('div')
                        .attr('class', 'layer-stats')
                        .style('margin-bottom', '20px');
                    
                    // Add layer name
                    layerDiv.append('h6')
                        .text(layerName);
                    
                    // Create stats table
                    const table = layerDiv.append('table')
                        .attr('class', 'table table-sm');
                    
                    const tbody = table.append('tbody');
                    
                    // Add stats rows
                    tbody.append('tr')
                        .html(`<td>Shape</td><td>${stats.shape.join(' × ')}</td>`);
                    tbody.append('tr')
                        .html(`<td>Mean</td><td>${stats.mean.toFixed(4)}</td>`);
                    tbody.append('tr')
                        .html(`<td>Std</td><td>${stats.std.toFixed(4)}</td>`);
                    tbody.append('tr')
                        .html(`<td>Min</td><td>${stats.min.toFixed(4)}</td>`);
                    tbody.append('tr')
                        .html(`<td>Max</td><td>${stats.max.toFixed(4)}</td>`);
                    
                    // Add distribution visualization
                    const width = 200;
                    const height = 50;
                    
                    const svg = layerDiv.append('svg')
                        .attr('width', width)
                        .attr('height', height)
                        .style('margin-top', '10px');
                    
                    // Create a simple distribution visualization
                    const scale = d3.scaleLinear()
                        .domain([stats.min, stats.max])
                        .range([0, width]);
                    
                    // Draw distribution line
                    svg.append('line')
                        .attr('x1', 0)
                        .attr('y1', height/2)
                        .attr('x2', width)
                        .attr('y2', height/2)
                        .style('stroke', '#ccc');
                    
                    // Draw mean marker
                    svg.append('circle')
                        .attr('cx', scale(stats.mean))
                        .attr('cy', height/2)
                        .attr('r', 5)
                        .style('fill', '#4CAF50');
                });
            } else {
                console.error('Failed to get layer activations:', data.error);
            }
        } catch (error) {
            console.error('Error updating layer activations:', error);
        }
    }
}

// Initialize visualizer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.modelVisualizer = new ModelVisualizer();
}); 
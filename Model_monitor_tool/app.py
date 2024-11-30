from flask import Flask, render_template, jsonify, request, url_for
import threading
from train import Trainer
from data_transform import data_transform
import torchviz
import io
import base64
import torch
from torch.autograd import Variable

app = Flask(__name__, static_folder='static')
trainer = Trainer(data_transform)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        print("Received start_training request")
        data = request.get_json()
        num_epochs = data.get('num_epochs', 20)
        
        learning_rate = data.get('learning_rate', 0.01)
        batch_size = data.get('batch_size', 128)
        optimizer_name = data.get('optimizer', 'SGD')
        model_configs = data.get('model_configs', None)

        print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, "
              f"batch_size={batch_size}, optimizer={optimizer_name}")
        print(f"Model configs: {model_configs}")

        if trainer.start_training(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            model_configs=model_configs
        ):
            print("Starting training thread")
            threading.Thread(target=trainer.train_model).start()
            return jsonify({"success": True})
        
        print("Could not start training - already in progress")
        return jsonify({"success": False, "message": "Training already in progress"})
    except Exception as e:
        print(f"Error in start_training route: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    trainer.stop_training()
    return jsonify({"success": True, "message": "Training stopped"})

@app.route('/get_logs')
def get_logs():
    """Get and clear the current logs"""
    try:
        logs = trainer.get_and_clear_logs()
        return jsonify({"logs": logs})
    except Exception as e:
        print(f"Error getting logs: {str(e)}")
        return jsonify({"logs": []})

@app.route('/get_metrics')
def get_metrics():
    return jsonify(trainer.get_metrics())

@app.route('/reset', methods=['POST'])
def reset():
    try:
        # Stop any existing training
        trainer.stop_training()
        
        # Reset the trainer
        trainer.reset_model()
        trainer.reset_metrics()
        
        # Clear any stored logs
        trainer.logs = []
        
        # Reset training state
        trainer.training_state.current_epoch = 0
        trainer.training_state.is_training = False
        
        # Reinitialize data loaders
        trainer.setup_data_loaders()
        
        return jsonify({
            "success": True, 
            "message": "Dashboard reset complete"
        })
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        })

@app.route('/get_samples')
def get_samples():
    try:
        samples = trainer.get_sample_predictions()
        if samples is None:
            return jsonify({
                'model1': {
                    'wrong_samples': [],
                    'correct_samples': []
                },
                'model2': {
                    'wrong_samples': [],
                    'correct_samples': []
                }
            })
        return jsonify(samples)
    except Exception as e:
        print(f"Error in get_samples route: {str(e)}")
        return jsonify({
            'model1': {
                'wrong_samples': [],
                'correct_samples': []
            },
            'model2': {
                'wrong_samples': [],
                'correct_samples': []
            }
        })

@app.route('/training_status')
def training_status():
    return jsonify({
        'is_training': trainer.training_state.is_training,
        'current_epoch': trainer.training_state.current_epoch,
        'total_epochs': trainer.training_state.total_epochs
    })

@app.route('/get_histograms')
def get_histograms():
    # Return histograms data (weights, biases, gradients)
    return jsonify({
        'weights': trainer.get_weights_histogram(),
        'biases': trainer.get_biases_histogram(),
        'gradients': trainer.get_gradients_histogram()
    })

@app.route('/get_embeddings')
def get_embeddings():
    # Return embeddings data
    return jsonify(trainer.get_embeddings())

@app.route('/get_performance_stats')
def get_performance_stats():
    # Return performance stats (GPU/TPU usage)
    return jsonify(trainer.get_performance_stats())

@app.route('/get_hparams')
def get_hparams():
    # Return hyperparameters and their metrics
    return jsonify(trainer.get_hparams())

@app.route('/get_model_architecture')
def get_model_architecture():
    try:
        # Create input of the same shape as your training data
        dummy_input = Variable(torch.randn(1, 1, 28, 28)).to(trainer.device)
        
        # Generate DOT graph
        dot = torchviz.make_dot(trainer.model(dummy_input), 
                               params=dict(trainer.model.named_parameters()),
                               show_attrs=True, show_saved=True)
        
        # Save to a buffer
        buffer = io.StringIO()
        dot.save(buffer, format='dot')
        dot_data = buffer.getvalue()
        
        return jsonify({
            'success': True,
            'dot_data': dot_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_layer_activations')
def get_layer_activations():
    try:
        # Get sample input
        dataiter = iter(trainer.test_loader)
        images, _ = next(dataiter)
        sample_input = images[0:1].to(trainer.device)
        
        # Dictionary to store activations
        activations = {}
        
        # Hook function to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks for each layer
        hooks = []
        for name, layer in trainer.model.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(layer.register_forward_hook(get_activation(name)))
        
        # Forward pass
        trainer.model.eval()
        with torch.no_grad():
            _ = trainer.model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process activations for visualization
        processed_activations = {}
        for name, activation in activations.items():
            # Convert to list for JSON serialization
            processed_activations[name] = {
                'mean': float(activation.mean()),
                'std': float(activation.std()),
                'min': float(activation.min()),
                'max': float(activation.max()),
                'shape': list(activation.shape)
            }
        
        return jsonify({
            'success': True,
            'activations': processed_activations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5050)
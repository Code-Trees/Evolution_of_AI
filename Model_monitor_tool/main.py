from app import app, initialize_trainer
from trainer import Trainer

def main():
    # Create trainer instance
    trainer_instance = Trainer()
    
    # Initialize the trainer in the Flask app
    initialize_trainer(trainer_instance)
    
    # Start the Flask app
    app.run(debug=True)

if __name__ == '__main__':
    main() 
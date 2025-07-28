import json
import os

print("--- Verifying Installed Models ---")

MODEL_PATHS = {
    'bi_encoder': './models/bi_encoder',
    'cross_encoder': './models/cross_encoder',
    'summarizer': './models/summarizer'
}

for name, path in MODEL_PATHS.items():
    print(f"\nChecking model: '{name}' at path: '{path}'")
    config_path = os.path.join(path, 'config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # The 'architectures' key tells us the model type.
            architecture = config_data.get('architectures', ['Not specified'])
            print(f"  > SUCCESS: Found config.json.")
            print(f"  > Model Architecture: {architecture[0]}")
            
            # This is the check that will reveal the problem
            if name == 'summarizer' and architecture[0] != 'BartForConditionalGeneration':
                 print("  > !!! WARNING !!! This is NOT a summarization model. This is the source of the error.")

        except Exception as e:
            print(f"  > ERROR: Could not read config.json. Error: {e}")
    else:
        print(f"  > ERROR: Directory or config.json does not exist.")

print("\n--- Verification Complete ---")
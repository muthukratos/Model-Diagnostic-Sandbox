import pandas as pd
import json
from model_selector import filter_models
from trainer import train_and_evaluate_models_adaptive
import profiling

# Load sample dataset
path = r"c:\Users\Abhinand\Downloads\Modal-Diagnostic-Sandbox-main\Modal-Diagnostic-Sandbox-main\sample_titanic.csv"
df = pd.read_csv(path)
profile = profiling.profile_data(df, target_column='Survived')
selected = filter_models(profile, task_type='classification')
print('Filtered models with justifications:')
print(json.dumps(selected, indent=2))
model_names = [m['model'] for m in selected]
training_results = train_and_evaluate_models_adaptive(df, 'Survived', model_names, task_type='classification', test_size=0.2, enable_two_stage='false')
# Merge justifications as main.py does
just_map = {m['model']: m.get('justification') for m in selected}
for item in training_results.get('models_evaluated', []):
    if 'justification' not in item or not item.get('justification'):
        item['justification'] = just_map.get(item.get('model'))
training_results['best_model']['justification'] = training_results['best_model'].get('justification') or just_map.get(training_results['best_model']['model'])
print('\nTraining results with justifications:')
print(json.dumps(training_results, indent=2))

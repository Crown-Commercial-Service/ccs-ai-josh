import os
import pandas as pd
import seaborn as sns

results = pd.read_csv(os.path.join('data', 'evaluation_results.csv'))
results['Document Type'] = results['Document Type'].str.strip()
results['Question Type'] = results['Question Type'].str.strip()
print(results[['Document Type', 'Question Type', 'Correctness', 'Retrieval', 'Groundedness']].head())

results_by_doc_type = results.groupby('Document Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print(results_by_doc_type)
results_by_doc_type.to_csv(os.path.join('data', 'results_by_doc_type.csv'), index=True)

results_by_question_type = results.groupby('Question Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print(results_by_question_type)
results_by_question_type.to_csv(os.path.join('data', 'results_by_question_type.csv'), index=True)

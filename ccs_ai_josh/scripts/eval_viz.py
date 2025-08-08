import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

plt.subplot(1, 2, 1)
sns.histplot(data=results[results['Document Type']=='Fact Sheet'], x='Correctness')
plt.title('Fact Sheet')
plt.subplot(1, 2, 2)
sns.histplot(data=results[results['Document Type']=='Market Report'], x='Correctness')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'correctness_hist.svg'), format='svg')
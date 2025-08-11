import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv(os.path.join('data', 'evaluation_results.csv'))
results['Document Type'] = results['Document Type'].str.strip()
results['Question Type'] = results['Question Type'].str.strip()
results['Document Match'] = pd.Categorical(
    results['Document Match'].astype(str),
    categories=['True', 'False'],
    ordered=True
)
# only considering unambiguous questions at the moment
results = results[results['Unambiguous']=='Yes'].copy().reset_index()

results_by_doc_type = results.groupby('Document Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print("Results by doc type:")
print(results_by_doc_type)
results_by_doc_type.to_csv(os.path.join('data', 'results_by_doc_type.csv'), index=True)

results_by_question_type = results.groupby('Question Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print("Results by question type:")
print(results_by_question_type)
results_by_question_type.to_csv(os.path.join('data', 'results_by_question_type.csv'), index=True)

# correctness histograms
plt.subplot(1, 3, 1)
sns.histplot(data=results[results['Document Type']=='Doc Search'], x='Correctness', stat='percent')
plt.title('Doc Search')
plt.subplot(1, 3, 2)
sns.histplot(data=results[results['Document Type']=='Fact Sheet'], x='Correctness', stat='percent')
plt.title('Fact Sheet')
plt.subplot(1, 3, 3)
sns.histplot(data=results[results['Document Type']=='Market Report'], x='Correctness', stat='percent')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'correctness_hist.svg'), format='svg')
plt.close()

# file match histograms
# restrict to pos controls, as neg controls will be non-matching by design
results_pos = results[results['Question Type']=='Positive'].copy()
plt.subplot(1, 3, 1)
sns.histplot(data=results_pos[results_pos['Document Type']=='Doc Search'], x='Document Match', stat='percent')
plt.title('Doc Search')
plt.subplot(1, 3, 2)
sns.histplot(data=results_pos[results_pos['Document Type']=='Fact Sheet'], x='Document Match', stat='percent')
plt.title('Fact Sheet')
plt.subplot(1, 3, 3)
sns.histplot(data=results_pos[results_pos['Document Type']=='Market Report'], x='Document Match', stat='percent')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'docmatch_hist.svg'), format='svg')
plt.close()
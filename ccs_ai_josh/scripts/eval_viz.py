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

#######################
## POSITIVE CONTROLS ##
#######################

# separate out positive controls
results_pos = results[results['Question Type']=='Positive'].copy().reset_index(drop=True)

pos_results_by_doc_type = results_pos.groupby('Document Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print("Positive control results by doc type:")
print(pos_results_by_doc_type)
pos_results_by_doc_type.to_csv(os.path.join('data', 'pos_results_by_doc_type.csv'), index=True)

# correctness histograms
plt.subplot(1, 3, 1)
sns.histplot(data=results_pos[results_pos['Document Type']=='Doc Search'], x='Correctness', stat='percent')
plt.title('Doc Search')
plt.subplot(1, 3, 2)
sns.histplot(data=results_pos[results_pos['Document Type']=='Fact Sheet'], x='Correctness', stat='percent')
plt.title('Fact Sheet')
plt.subplot(1, 3, 3)
sns.histplot(data=results_pos[results_pos['Document Type']=='Market Report'], x='Correctness', stat='percent')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'pos_correctness_hist_by_doctype.svg'), format='svg')
plt.close()

# file match histograms
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
plt.savefig(os.path.join('data', 'pos_docmatch_hist_by_doctype.svg'), format='svg')
plt.close()

#######################
## NEGATIVE CONTROLS ##
#######################

# separate out negative controls
results_neg = results[results['Question Type']=='Negative'].copy().reset_index(drop=True)

neg_results_by_doc_type = results_neg.groupby('Document Type')[['Correctness', 'Retrieval', 'Groundedness']].mean().round(2)
print("Negative control results by doc type:")
print(neg_results_by_doc_type)
neg_results_by_doc_type.to_csv(os.path.join('data', 'neg_results_by_doc_type.csv'), index=True)

# correctness histograms
plt.subplot(1, 2, 1)
sns.histplot(data=results_neg[results_neg['Document Type']=='Fact Sheet'], x='Correctness', stat='percent')
plt.title('Fact Sheet')
plt.subplot(1, 2, 2)
sns.histplot(data=results_neg[results_neg['Document Type']=='Market Report'], x='Correctness', stat='percent')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'neg_correctness_hist_by_doctype.svg'), format='svg')
plt.close()

# file match histograms
plt.subplot(1, 2, 1)
sns.histplot(data=results_neg[results_neg['Document Type']=='Fact Sheet'], x='Document Match', stat='percent')
plt.title('Fact Sheet')
plt.subplot(1, 2, 2)
sns.histplot(data=results_neg[results_neg['Document Type']=='Market Report'], x='Document Match', stat='percent')
plt.title('Market Report')
plt.tight_layout()
plt.savefig(os.path.join('data', 'neg_docmatch_hist_by_doctype.svg'), format='svg')
plt.close()

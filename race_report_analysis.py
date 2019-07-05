# analysis of r/running race reports to discover: what is the role of 
# (subjective) struggle/pain in a runner's evaluation of their own success/the 
# success of other runners?
# furthermore, does the running community reward others for (subjective) pain?

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

from ols import ols_reg

figpath = 'figures/'

data = pd.read_csv('data/race_report_df.csv')
data[['what','where','when','time',
                'distance_measure','body','body_clean','author','title']] = data[['what','where','when','time','distance_measure','body','body_clean','author','title']].fillna('')

dictionary = pd.read_csv('data/inquirercleaned.csv')

# generate the (# articles) x (# topics) article-topic matrix: 
# i.e. entry (a,k) is number of words from topic k in article a
for i in list(range(len(data))):
    content = data.loc[i,'body_clean']
    word_ct_vec = np.asmatrix(dictionary.Entry.apply(lambda w: len(re.findall(w,content))))
    try:
        article_word_matrix = np.concatenate((article_word_matrix,word_ct_vec),axis=0)
    except NameError:
        article_word_matrix = word_ct_vec

word_topic_matrix = np.asmatrix(dictionary.drop(columns=['Entry']))
article_topic_matrix = np.matmul(article_word_matrix,word_topic_matrix)

# part a - exploratory analysis

# i - which topics appear most frequently, on avg., in race reports?
avg_topic_frequencies = pd.DataFrame({'avg_frequency':np.mean(article_topic_matrix,axis=0).tolist()[0]},index=list(dictionary.columns[1:])).sort_values(by='avg_frequency',ascending=False)

fig, ax = plt.subplots(figsize=(8,4))  
avg_topic_frequencies.plot.bar(y='avg_frequency',ax=ax,color='#008793')
plt.savefig(figpath + 'avg_topic_frequencies.pdf') 

# interpretation: unsurprisingly, words in the "active" category appear most 
# frequently on average, followed by "strong."
# words in the "pain" category appear relatively infrequently on average.
# (however, 98% of race reports have "pain" words in them.)
# interestingly, "negative" words appear with the third highest avg frequency.

# note that the above could simply be a product of the fact that there are more
# dictionary words in the active, strong, etc. categories:
dictionary_topic_frequencies = pd.DataFrame(dictionary[['Pstv', 'Ngtv', 'Strong', 'Weak', 'Active', 'Passive','Pleasur', 'Pain', 'Feel', 'Arousal', 'Virtue', 'Vice']].sum(),columns=['dictionary_frequency']).sort_values(by='dictionary_frequency',ascending=False)

fig, ax = plt.subplots(2,1,figsize=(8,8))
avg_topic_frequencies.plot.bar(y='avg_frequency',ax=ax[0],color='#008793')
dictionary_topic_frequencies.plot.bar(y='dictionary_frequency',ax=ax[1],color='#008793')                           
                                      
# interpretation: in fact, the ordering of average frequencies and the ordering
# of dictionary counts are highly rank-correlated. 

# this suggests that we want to look at average topic frequencies relative to
# the expected count based on dictionary counts.         

# note that because ~50% of words in the full dictionary belong to 2+ topics,
# we use an abbreviated dicitionary below which includes only those words
# belonging to one topic.
topic_count_by_word = pd.DataFrame(dictionary.sum(axis=1),columns=['n_topic'])
idx_single_topic = topic_count_by_word[topic_count_by_word['n_topic']==1].index
dictionary_single_topic = dictionary.loc[idx_single_topic]

# under a null that all words in the abbreviated dictionary are equally likely 
# to be used, the probability that the ith word in a given race report is from
# topic k is Pr(word i in topic k) = (no. dict. words in k/no. dict. words);
# i.e. word i's topic has a multinomial distribution.
multinom_topic_probabilities = (dictionary_single_topic[['Pstv', 'Ngtv', 'Strong', 'Weak', 'Active', 'Passive','Pleasur', 'Pain', 'Feel', 'Arousal', 'Virtue', 'Vice']].sum(axis=0))/len(dictionary_single_topic)
                                      
# if the null is true, then we should see the sample probability of word i 
# being from topic k ~= Pr(word i in topic k) from above. alternatively, it 
# could be the case that all words are not equally likely, i.e. words from
# certain topics are more or less likely to appear in race reports. in this 
# case we want to know what those topics are.
for i in list(range(len(data))):
    content = data.loc[i,'body_clean']
    word_ct_vec = np.asmatrix(dictionary_single_topic.Entry.apply(lambda w: len(re.findall(w,content))))
    try:
        article_word_single_topic_matrix = np.concatenate((article_word_single_topic_matrix,word_ct_vec),axis=0)
    except NameError:
        article_word_single_topic_matrix = word_ct_vec
word_single_topic_matrix = np.asmatrix(dictionary_single_topic.drop(columns=['Entry']))
article_single_topic_matrix = np.matmul(article_word_single_topic_matrix,word_single_topic_matrix)

sample_proportions = pd.Series(np.mean(np.divide(article_single_topic_matrix,np.sum(article_single_topic_matrix+1e-10,1)),axis=0).tolist()[0],index=dictionary_single_topic.columns[1:])
z_statistics = np.divide((sample_proportions - multinom_topic_probabilities),np.sqrt(np.multiply(multinom_topic_probabilities,1-multinom_topic_probabilities)/len(data)))

# interpretation: we can't reject the null hypothesis for any topic, although
# the test statistic for the "active" category is marginally significant and
# positive (1.82). this is not surprising given that the texts being analyzed
# are race reports.

# ii - is there a correlation between certain topic categories in race reports?
# e.g. does a higher number of "pain" words correlate with a higher number of
# "virtue" (or "strong," "pleasure," "positive"?) words? 
# note that using the full set of dictionary words here will introduce
# mechanical correlations because some words are shared by topic categories.
# thus, we'll again use only the set of single-topic words for this analysis.
topics = dictionary.columns[1:]
correlation_matrix = np.corrcoef((article_single_topic_matrix).T)
fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix)
ax.set_xticks(np.arange(np.size(article_single_topic_matrix,1)))
ax.set_yticks(np.arange(np.size(article_single_topic_matrix,1)))
ax.set_xticklabels(topics)
ax.set_yticklabels(topics)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
fig.tight_layout()
plt.savefig(figpath + 'correlations.pdf')

# interpretation:
# counts of "positive" words are highly positively correlated with "strong," 
# "active" words, but interestingly also with "weak," "passive" words.
# similarly, "strong" words are highly positively correlated with "active" 
# words, but also with "passive" words.

# iii - 10 of the 12 word topic categories are opposites. how do these
# correlations change if we take a "net" count with respect to each pair?
# e.g. for each article, take the count of positive words - negative words,
# strong words - weak words, ... etc. and then compute correlations.
net_positive = article_topic_matrix[:,0] - article_topic_matrix[:,1]
net_strong = article_topic_matrix[:,2] - article_topic_matrix[:,3]
net_active = article_topic_matrix[:,4] - article_topic_matrix[:,5]
net_pleasure = article_topic_matrix[:,6] - article_topic_matrix[:,7]
net_virtue = article_topic_matrix[:,10] - article_topic_matrix[:,11]
net_article_topic_matrix = np.concatenate((net_positive, net_strong, net_active, net_pleasure, net_virtue),axis=1)

net_topics = ['Pstv','Strong','Active','Pleasur','Virtue']

correlation_matrix = np.corrcoef((net_article_topic_matrix).T)
fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix)
ax.set_xticks(np.arange(np.size(net_article_topic_matrix,1)))
ax.set_yticks(np.arange(np.size(net_article_topic_matrix,1)))
ax.set_xticklabels(net_topics)
ax.set_yticklabels(net_topics)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
fig.tight_layout()
plt.savefig(figpath + 'net_correlations.pdf')

# interpretation: 
# net "pleasure" is negatively correlated with net "virtue"-that is, net "pain"
# is positively correlated with net "virtue." this supports the idea that 
# runners associate (subjective) suffering with feelings of virtue and pride.
# net "strength" is positively correlated with net "virtue" and net "active."
# net "positive" is also weakly positively correlated with net "pleasure."
# these results are consistent with what one would expect from runners.
# however, there are some surprising results here. e.g. net "positive" and net 
# "strong" and net "positive" and net "active" are negatively correlated.
# not clear what would motivate this. possible that reports which describe more
# difficult races express less positive sentiment overall, but include more 
# description of "pushing through" and/or the activity of running in general.

# part b - further examine the positive relationship between (net) "pain" and 
# (net) "virtue"

# is this relationship statistically significant?
#   --> if yes: further suggests that runners hold themselves in higher esteem
#       when they perceive that they have experienced more physical suffering.

# multiply the net "pleasure" vector by -1 to get a net "pain" score
X = np.concatenate((np.ones((len(net_article_topic_matrix),1)),-1*net_article_topic_matrix[:,3]),axis=1)
y = net_article_topic_matrix[:,4]
reg = ols_reg(X,y) 

# interpretation:
# the coefficient on net "pain" score is 1.09, with a t-stat of 3.9.

# does the relationship persist if we control for race distance? objective race
# performance (speed)?
#   --> e.g. perhaps runners experience more subjective pain over longer races,
#       but are also more proud of those efforts
#   --> similarly, perhaps runners experience more subjective pain when 
#       pushing for a personal best, but are most proud of those efforts.
idx_missing_distance = np.isnan(data['distance_std'])
idx_missing_time = np.isnan(data['time_mins'])

distance = np.reshape(np.asmatrix(data.loc[~idx_missing_distance,'distance_std']),(sum(~idx_missing_distance),1))
X = np.concatenate((np.ones((sum(~idx_missing_distance),1)),-1*net_article_topic_matrix[~idx_missing_distance,3],distance),axis=1)
y = net_article_topic_matrix[~idx_missing_distance,4]
reg = ols_reg(X,y) 

# interpretation:
# the coefficient on net "pain" score drops slightly to 0.96, and retains a t-
# stat of 3.1. 
# notably, the coefficient on race distance is positive but not significant.
# this indicates that runners do not express significantly greater feelings of 
# pride or virtue because they raced longer distances.

distance = np.reshape(np.asmatrix(data.loc[~(idx_missing_distance) & ~(idx_missing_time),'distance_std']),(sum(~(idx_missing_distance) & ~(idx_missing_time)),1))
time = np.reshape(np.asmatrix(data.loc[~(idx_missing_distance) & ~(idx_missing_time),'time_mins']),(sum(~(idx_missing_distance) & ~(idx_missing_time)),1))
X = np.concatenate((np.ones((sum(~(idx_missing_distance) & ~(idx_missing_time)),1)),-1*net_article_topic_matrix[~(idx_missing_distance) & ~(idx_missing_time),3],distance,time),axis=1)
y = net_article_topic_matrix[~(idx_missing_distance) & ~(idx_missing_time),4]
reg = ols_reg(X,y) 

# interpretation:
# the coefficient on net "pain" score again drops slightly to 0.89, but retains
# a t-stat of 3.0.
# the coefficient on time is negative. this makes sense: faster times mean more 
# improvement, PRs to be proud of, etc. however, this relationship is not 
# statistically significant.

# iv - on what (net) topics does the first principal component of the article-
# topic matrix weight most heavily?

# standardize article-topic matrix to perform pca
colmeans = net_article_topic_matrix.mean(0)
colstds = net_article_topic_matrix.std(0)
article_topic_matrix_std = np.divide(net_article_topic_matrix-np.repeat(colmeans,np.size(net_article_topic_matrix,0),axis=0),np.repeat(colstds,np.size(net_article_topic_matrix,0),axis=0))

U, s, W = np.linalg.svd(article_topic_matrix_std,full_matrices=False)

pc1_wts = pd.Series(W.T[:,0].flatten().tolist()[0],index=net_topics)
# first pc weights most heavily on net "strong" and net "active" topics.
# we can think of this as the "fitness" factor based on these weightings.
# that said, these weightings do not shed much light on the relationship 
# between (subjective) pain and a runner's self-evaluation.

pc2_wts = pd.Series(W.T[:,1].flatten().tolist()[0],index=net_topics)
# second pc weights most heavily on net "positive" and net "virtue" topics.
# we can think of this as the "pride" factor based on these weightings.

# does a post's position in pc1/pc2 space have any relationship to its score?
pc_space_data = np.array(np.matmul(net_article_topic_matrix,W))

fig, ax = plt.subplots()
plt.scatter(pc_space_data[:,0],pc_space_data[:,1],c=data['score'].values,cmap='YlOrRd')
plt.xlabel('PC1 ("fitness" factor)')
plt.ylabel('PC2 ("pride" factor)')
plt.savefig(figpath + 'pc1_pc2_space.pdf')

# interpretation:
# nothing particularly noteworthy here. no clear relationship between position 
# in pc space and post score. 

# v - examine the relationship between post score and net "pain" level 
# (controlling for other post elements).
#   --> are posts expressing a higher level of suffering "rewarded" by other
#       runners with a higher score? is suffering "put on a pedestal"?

# first, look for a simple correlation between net pain and post score
net_pain = np.array(-1*net_pleasure)
post_score = np.array(data['score'])

fig, ax = plt.subplots()
plt.scatter(net_pain,post_score)
plt.xlabel('Net pain score')
plt.ylabel('Post upvotes')
plt.savefig(figpath + 'scatter_pain_score.pdf')

# interpretation:
# it looks like there might be a positive relationship between the two 
# variables, but if so it's nonlinear. try transforming post score.

fig, ax = plt.subplots()
plt.scatter(net_pain,np.log(post_score))
plt.xlabel('Net pain score')
plt.ylabel('log(Post upvotes)')
plt.savefig(figpath + 'scatter_pain_logscore.pdf')

# interpretation:
# there is a clear positive, linear relationship between net pain score and the 
# log of post upvotes: i.e. expressing more suffering in a post is correlated 
# with receiving more "likes" on that post.

# is this relationship statistically significant?
X = np.concatenate((np.ones((len(net_article_topic_matrix),1)),np.asmatrix(net_pain)),axis=1)
y = np.asmatrix(np.log(post_score)).T
reg = ols_reg(X,y) 

# interpretation:
# the coefficient on net pain is positive (0.02), but not statistically 
# significant (t ~= 1.5).




        
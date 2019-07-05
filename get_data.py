import pandas as pd
import numpy as np
import praw
import re
import string
import os

# function to return subreddit posts associated with a given keyword
def get_keyword_data(reddit,sub,kw,pt):
    
    keyword_df = pd.DataFrame()
    
    # get posts associated with the keyword 
    key_posts = sub.search(kw)
    
    # get post-level data for each post
    for p in key_posts:
        post_df = get_post_data(reddit,p,pt)
        keyword_df = keyword_df.append(post_df)
    
    keyword_df = keyword_df.reset_index(drop=True)
    
    # return df with each row being a single post
    return keyword_df

# function to get post-level data 
def get_post_data(reddit,p,pt):
    
    # get post id, then send it off to get comment data
    post_vars = vars(p)
    post_id = post_vars['id']
        
    print('now parsing post ', post_vars['id'])
          
    # get the post submission based on the post id
    submission = reddit.submission(id=post_id)
    submission_text = submission.selftext
    
    # get race attributes from post as possible
    what = ''
    try_what = re.findall(r'(?<=What).*?(?=\n)',submission_text)
    try:
        if len(try_what) > 0 and len(try_what[0]) < 100:
            what = try_what[0].translate(pt).strip()
    except:
        what = ''
        
    where = ''
    try_where = re.findall(r'(?<=Where).*?(?=\n)',submission_text)
    try:
        if len(try_where) > 0 and len(try_where[0]) < 100:
            where = try_where[0].translate(pt).strip() 
    except:
        where = ''
        
    when = ''
    try_when = re.findall(r'(?<=When).*?(?=\n)',submission_text)
    try:
        if len(try_when) > 0 and len(try_when[0]) < 100:
            when = try_when[0].translate(pt).strip()
    except:
        when = ''
        
    distance = np.nan
    distance_measure = ''
    try_distance = re.findall(r'(?<=How far).*?(?=\n)',submission_text)
    try:
        if len(try_distance) > 0:
            distance = float(re.search(r'([0-9]{1,3})([\.][0-9]*)?',try_distance[0])[0])
            distance_measure = re.split(r'([0-9]{1,3})([\.][0-9]*)?',try_distance[0])[-1].translate(pt).strip().lower()
    except:
        distance = np.nan
        distance_measure = ''
    
    finish_time = ''
    try_time = re.findall(r'(?<=Finish time).*?(?=\n)',submission_text)
    try:
        if len(try_time) > 0:
            finish_time = re.search(r'([0-9]{1,3})([:][0-9]{2})?([:][0-9]{2})?',try_time[0]).group(0)
    except:
        finish_time = ''
    
    # get the cleaned body of the post
    clean_text = submission_text.lower().translate(pt)
    
    # get the post score and upvote ratio
    score = submission.score
    upvote_ratio = submission.upvote_ratio
    
    # get the post author
    author = submission.author
    
    # get the post title 
    title = submission.title
            
    # return a dataframe with relevant post information
    post_df = pd.DataFrame({'what':what,
                            'where':where,
                            'when':when,
                            'distance':distance,
                            'distance_measure':distance_measure,
                            'time':finish_time,
                            'body':clean_text,
                            'score':score,
                            'upvote_ratio':upvote_ratio,
                            'author':author,
                            'title':title},index=[0])
    
    return post_df

# functions for processing post-data collection:
    
# fill in missing distance measure if race name includes a standard distance
def get_distance(row,std_map):
    matched_names = [len(re.findall(s,row['what'].lower()))>0 for s in std_map['name']]
    if sum(matched_names)==1:
        race = [x for i,x in enumerate(std_map['name']) if matched_names[i]][0]
        row['distance'] = standard_race_map.loc[standard_race_map['name']==race,'distance'].values[0]
        row['distance_measure'] = standard_race_map.loc[standard_race_map['name']==race,'distance_measure'].values[0]
    if sum(matched_names)==2 and 'half marathon' in row['what'].lower():
        row['distance'] = 13.1
        row['distance_measure'] = 'm'
    return row

# standardize the naming convention for distance measures
def get_measure(dist,std_m,std_k):
    if dist in std_m:
        return 'm'
    if dist in std_k:
        return 'k'
    return ''

# convert finishing times from string to numeric, in minutes
def get_time_in_minutes(time,dist):
    
    components = time.split(':')
    
    if len(components) == 3:
        
        return (int(components[0])*60 + int(components[1]) + int(components[2])/60)
    
    elif len(components) == 2:
    
        lower_ci_mins = (1/357)*dist # 357 m/min ~= 4.5 min/mi pace
        upper_ci_mins = (1/100)*dist # 100 m/min ~= 16 min/mi pace
        
        hm_time = int(components[0])*60 + int(components[1])
        ms_time = int(components[0]) + int(components[1])/60
        
        hm_valid = (hm_time >= lower_ci_mins) and (hm_time <= upper_ci_mins) 
        ms_valid = (ms_time >= lower_ci_mins) and (ms_time <= upper_ci_mins) 
        
        if hm_valid and not ms_valid:
            return hm_time
            
        elif not hm_valid and ms_valid:
            return ms_time
            
        else:
            print('Problem parsing time ' + time)
        
    return np.nan

# user data for  reddit auth
client_id = 'VDyvdULRVZdqRw'
client_secret = '44CK3PBrFyJIaIpNVnnKIb7zAQQ'
username = 'febreeze_hotbox'
password = 'A center for ants?'
user_agent = 'fh_research'

# search keyword
keyword = 'race report'

# initialize praw utility to help get data
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent,
                     username=username,
                     password=password)

# specify subreddit of interest
subreddit = reddit.subreddit('running')

punctuation_dict = str.maketrans(dict.fromkeys(string.punctuation))

# get race report data
race_report_df = get_keyword_data(reddit,subreddit,keyword,punctuation_dict)

# get race distance from race name if missing and possible
standard_race_map = pd.DataFrame({'name':['5k','10k','half marathon','marathon'],
                                  'distance':[5,10,13.1,26.2],
                                  'distance_measure':['k','k','m','m']})
idx_nodist = (race_report_df['what']!='') & (np.isnan(race_report_df['distance']))
race_report_df.loc[idx_nodist] = race_report_df.loc[idx_nodist].apply(lambda row: get_distance(row,standard_race_map),axis=1)


# standardize distance measures to either 'm' or 'k' (or unknown)
mile_abbrs = ['m','mi','mile','ms','mis','miles']
kilom_abbrs = ['k','km','kilometer','kilometre','ks','kms','kilometers','kilometres']
race_report_df['distance_measure'] = race_report_df['distance_measure'].apply(lambda m: get_measure(m.lower(),mile_abbrs,kilom_abbrs))

# make an (informed) guess at distance measures where missing and possible
standard_mile_distances = [3.1,6.2,13.1,26.2]
standard_kilom_distances = [5,10,21,42]
idx_nomeasure = (~np.isnan(race_report_df['distance'])) & (race_report_df['distance_measure']=='')
race_report_df.loc[idx_nomeasure,'distance_measure'] = race_report_df.loc[idx_nomeasure,'distance'].apply(lambda dist: get_measure(dist,standard_mile_distances,standard_kilom_distances))  

# get standardized race distance in meters
race_report_df.loc[race_report_df['distance_measure']=='k','distance_std'] = race_report_df.loc[race_report_df['distance_measure']=='k','distance']*1000
race_report_df.loc[race_report_df['distance_measure']=='m','distance_std'] = race_report_df.loc[race_report_df['distance_measure']=='m','distance']*1609   

# get standardized, numeric finish time in minutes
race_report_df['time_mins'] = race_report_df.apply(lambda row: get_time_in_minutes(row['time'],row['distance_std']),axis=1)

# clean body text
race_report_df['body_clean'] = race_report_df['body'].str.lower().str.translate(punctuation_dict).str.replace('\n','')

race_report_df.to_csv('data/race_report_df.csv',index=False)
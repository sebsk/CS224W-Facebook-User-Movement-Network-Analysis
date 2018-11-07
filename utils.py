
# coding: utf-8

# The main network analysis package adopted in this module is igraph.


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy, deepcopy
import geocoder
import geopy.distance
from pprint import pprint
import folium
import igraph
from collections import defaultdict
from collections import OrderedDict
from matplotlib import colors as mcolors
sns.set()


# Add coordinates ('Starting location lon', 'Starting location lat', 'Ending location lon', 'Ending location lat') to the edgelist-like dataframe.
def geo_location(df):
    extracted_lonlat=df['Geometry'][df['Geometry'].str.contains('POINT')].str.replace('(POINT )|\(|\)|,', '').str.split()
    df['Geometry'][df['Geometry'].str.contains('POINT')] = list(map(lambda x: x[0] + ', ' + x[1] + ', '+x[0] + ', ' + x[1], extracted_lonlat))
    split_lonlat = df['Geometry'].str.replace('(MULTILINESTRING )|\(|\)|,', '').str.split()
    location_features = ['Starting location lon', 'Starting location lat', 'Ending location lon', 'Ending location lat']
    for i in range(4):
        df[location_features[i]] = pd.Series(map(lambda x: x[i], split_lonlat)).astype('float')

# In the edgelist-like dataframe, fill unknown region names and location code with the nearest region name and location code. 
# The nearest location has to be within threshold (km) from the unknown location.
def fill_unknown_regions(df, which, threshold=5): # which could fill in "Starting" or "Ending"
    which = which.capitalize()

    df.sort_values(by=[which + ' location lon', which + ' location lat'], inplace=True)
    for i in range(df.shape[0]):
        if '__' in df.iloc[i, ][which + ' Region Name']:
            lon = df.iloc[i, ][which + ' location lon']
            lat = df.iloc[i, ][which + ' location lat']
            if i+1 < df.shape[0]:
                diff_plus_1 = geopy.distance.vincenty((lat, lon), (df.iloc[i+1, ][which + ' location lat'], df.iloc[i+1, ][which + ' location lon'])).km
            else:
                diff_plus_1 = np.inf
            if i-1 >= 0:
                diff_minus_1 = geopy.distance.vincenty((lat, lon), (df.iloc[i-1, ][which + ' location lat'], df.iloc[i-1, ][which + ' location lon'])).km
            else:
                diff_minus_1 = np.inf
            if diff_plus_1 > diff_minus_1 and diff_minus_1 < threshold:
                if df[which + ' Region Name'].iloc[i-1]!='__':
                    df[which + ' Region Name'].iloc[i] = copy(df[which + ' Region Name'].iloc[i-1])
                    df[which + ' Location'].iloc[i] = copy(df[which + ' Location'].iloc[i-1])

            elif diff_plus_1 <= diff_minus_1 and diff_plus_1 < threshold:
                if df[which + ' Region Name'].iloc[i+1]!='__':
                    df[which + ' Region Name'].iloc[i] = copy(df[which + ' Region Name'].iloc[i+1])
                    df[which + ' Location'].iloc[i] = copy(df[which + ' Location'].iloc[i+1])

# Fill NAs and adjust inconsistency in the original datasets.
def data_cleansing(df):
    # Fill NA in People Moving
    idx = pd.isnull(df['Baseline: People Moving'])
    df['Baseline: People Moving'][idx] = 100 * df[idx]['Difference']/df[idx]['Percent Change']
    idx = pd.isnull(df['Crisis: People Moving'])
    df['Crisis: People Moving'][idx] = df[idx]['Difference'] + df[idx]['Baseline: People Moving']

    # Fill NA in Differene and Percent Change.
    idx = pd.isnull(df['Difference'])
    df['Difference'][idx] = df['Crisis: People Moving'][idx] - df['Baseline: People Moving'][idx]
    idx = pd.isnull(df['Percent Change'])
    df['Percent Change'][idx] = df['Difference'][idx] / df['Baseline: People Moving'][idx] *100

    # Find the median of the known z-score/baseline ratio, which is used for filling NAs in z score later on.
    df_z_calc = df[(pd.isnull(df['Standard (Z) Score'])==False) & (df['Crisis: People Moving'] - df['Baseline: People Moving']==df['Difference']) & (df['Standard (Z) Score']!=0) &(df['Baseline: People Moving']>0)]
    df_z_calc['sd over baseline'] = df_z_calc['Difference'] / df_z_calc['Standard (Z) Score'] / df_z_calc['Baseline: People Moving']
    sd_over_baseline = df_z_calc['sd over baseline'].median()

    # Tackle inconsisitency in People Moving and adjust z score based on new People Moving
    df['Baseline: People Moving'] = df['Crisis: People Moving'] - df['Difference']
    idx = (pd.isnull(df['Standard (Z) Score'])|idx)&(df['Baseline: People Moving']!=0)
    df['Standard (Z) Score'][idx] = df['Difference'][idx]/(sd_over_baseline * df['Baseline: People Moving'][idx])

    # Fill unknown reagions lacking name and/or locaiton code is -1 with nearest neighbor
    fill_unknown_regions(df, 'Starting')
    fill_unknown_regions(df, 'Ending')

    # Fill unknown regions with OpenCage map.
    locations = {}
    for i in df[(df['Starting Region Name']=='__') | (df['Ending Region Name']=='__')].index:
        if df['Starting Region Name'].loc[i] == '__':
            geo_info = geocoder.opencage([df['Starting location lat'].loc[i], df['Starting location lon'].loc[i]], method='reverse', key='8793bad49e134015a1010a75f0bb4a52').json
            try:
                df['Starting Region Name'].loc[i] = geo_info['city'] + '_' + geo_info['county'] + '_' + geo_info['state']
            except KeyError:
                continue
            if df['Starting Region Name'].loc[i] in locations.keys():
                df['Starting Location'].loc[i] = locations.get(df['Starting Region Name'].loc[i])
            else:
                df['Starting Location'].loc[i] = max(df['Starting Location'].max(), df['Ending Location'].max()) + 1
                locations.update({df['Starting Region Name'].loc[i]: df['Starting Location'].loc[i]})
                
        if df['Ending Region Name'].loc[i] == '__':
            geo_info = geocoder.opencage([df['Ending location lat'].loc[i], df['Ending location lon'].loc[i]], method='reverse', key='8793bad49e134015a1010a75f0bb4a52').json
            try:
                df['Ending Region Name'].loc[i] = geo_info['city'] + '_' + geo_info['county'] + '_' + geo_info['state']
            except KeyError:
                continue
            if df['Ending Region Name'].loc[i] in locations.keys():
                df['Ending Location'].loc[i] = locations.get(df['Ending Region Name'].loc[i])
            else:
                df['Ending Location'].loc[i] = max(df['Starting Location'].max(), df['Ending Location'].max()) + 1
                locations.update({df['Ending Region Name'].loc[i]: df['Ending Location'].loc[i]})

    # drop duplicates (in terms of starting location and ending location) by keeping only the record with highest 'Crisis: People Moving'
    df.sort_values(by=['Crisis: People Moving'], ascending=False, inplace=True)
    df.drop_duplicates(subset=['Starting Region Name', 'Ending Region Name'], inplace=True)
    df = df[(df['Starting Region Name']!='__')&(df['Ending Region Name']!='__')]

    # replace location code -1 with unique numbers.
    df_start = df[['Starting Location', 'Starting Region Name']].rename(columns={'Starting Location': 'location', 'Starting Region Name': 'region name'})
    df_end = df[['Ending Location', 'Ending Region Name']].rename(columns={'Ending Location': 'location', 'Ending Region Name': 'region name'})
    df_regions = pd.concat([df_start, df_end]).drop_duplicates()
    to_be_assigned = df_regions[df_regions['location']==-1]['region name']
    for region in to_be_assigned:
        df_regions['location'][df_regions['region name']==region] = df_regions['location'].max() + 1
    df_minus_1 = df[(df['Starting Location']==-1)|(df['Ending Location']==-1)]
    df = df[(df['Starting Location']!=-1)&(df['Ending Location']!=-1)]
    df_minus_1 = df_minus_1.merge(df_regions, how='left', left_on='Starting Region Name', right_on='region name')
    df_minus_1['Starting Location'] = copy(df_minus_1['location'])
    df_minus_1.drop(columns=['location', 'region name'], inplace=True)
    df_minus_1 = df_minus_1.merge(df_regions, how='left', left_on='Ending Region Name', right_on='region name')
    df_minus_1['Ending Location'] = copy(df_minus_1['location'])
    df_minus_1.drop(columns=['location', 'region name'], inplace=True)
    df = df.append(df_minus_1)

# Create edgelist for later graph creation.
def create_edgelist(df, when): # choose 'baseline' or 'crisis' as 'when' argument
    if when.lower()=='baseline':
        return list(zip(df['Starting Region Name'], df['Ending Region Name'], df['Baseline: People Moving']))
    elif when.lower()=='crisis':
        return list(zip(df['Starting Region Name'], df['Ending Region Name'], df['Crisis: People Moving']))
    else:
        raise ValueError('Please choose one between "baseline" and "crisis".')

# Create graph. The graph object is an igraph graph.
def create_graph(edgelist, self_edges=True, zero_weight_edges=False, directed=True, reverse=False):
    G = igraph.Graph(directed=directed)
    weight = []
    el = []
    vx = set()
    for edge in edgelist:
        vx.add(edge[0])
        vx.add(edge[1])
        if zero_weight_edges:
            if self_edges:
                if reverse:
                    el.append((edge[1], edge[0]))
                else:
                    el.append((edge[0], edge[1]))
                weight.append(edge[2])
            elif edge[0]!=edge[1]:
                if reverse:
                    el.append((edge[1], edge[0]))
                else:
                    el.append((edge[0], edge[1]))
                weight.append(edge[2])
        elif edge[2]!=0:
            if self_edges:
                if reverse:
                    el.append((edge[1], edge[0]))
                else:
                    el.append((edge[0], edge[1]))
                weight.append(edge[2])
            elif edge[0]!=edge[1]:
                if reverse:
                    el.append((edge[1], edge[0]))
                else:
                    el.append((edge[0], edge[1]))
                weight.append(edge[2])
    G.add_vertices(list(vx))
    G.add_edges(el)
    G.es['weight'] = weight
    return G

# Print some basic statistics to the graph.
def basic_stats(edgelist):
    G = create_graph(edgelist, zero_weight_edges=False)
    G_no_self = create_graph(edgelist, zero_weight_edges=False, self_edges=False)
    print 'number of nodes: %d' % G.vcount()
    print 'number of weighted edges: %d' % G.ecount()
    print 'total FB users: %d' % np.sum(G.es['weight'])
    print 'total FB users travelling between cities: %d' % np.sum(G_no_self.es['weight'])

# Create a dataframe containing region name, lon, lat.
def df_lonlat(df):
    df_start = df[['Starting Region Name', 'Starting location lon', 'Starting location lat']].rename(columns={'Starting Region Name': 'region name', 'Starting location lon': 'lon', 'Starting location lat':'lat'})
    df_end = df[['Ending Region Name', 'Ending location lon', 'Ending location lat']].rename(columns={'Ending Region Name': 'region name', 'Ending location lon': 'lon', 'Ending location lat': 'lat'})
    df_regions = pd.concat([df_start, df_end]).groupby('region name').mean().reset_index()
    return df_regions

#return pagerank (incl. and excl. loops), in-degree (incl. and excl. loops), out-degree (incl. and excl. loops), 
# total-degree (incl. and excl. loops), closeness, betweenness, cluster coefficient (weighted and unweighted).
def calc_params(df, when='crisis', edgelist=None, reverse=False): # 'edgelist' overwrites 'when'.
    df_regions = df_lonlat(df)
    if edgelist is None:
        edgelist = create_edgelist(df, when)
    G_no_self = create_graph(edgelist, reverse=reverse, self_edges=False)
    G = create_graph(edgelist, reverse=reverse, self_edges=True)
    G_no_self_und = create_graph(edgelist, reverse=reverse, self_edges=False, directed=False)
    prs = G_no_self.pagerank(weights='weight')
    prs_incl_self = G.pagerank(weights='weight')
    indeg = G.strength(mode=2, loops=False, weights='weight')
    indeg_incl_self = G.strength(mode=2, loops=True, weights='weight')
    outdeg = G.strength(mode=1, loops=False, weights='weight')
    outdeg_incl_self = G.strength(mode=1, loops=True, weights='weight')
    totdeg = np.array(indeg) + np.array(outdeg)
    totdeg_incl_self = np.array(indeg_incl_self) + np.array(outdeg_incl_self)
    cls = G_no_self.closeness(weights='weight')
    btw = G_no_self.betweenness(weights='weight')
    weighted_clust_coef = G_no_self_und.transitivity_local_undirected(mode="zero", weights='weight')
    clust_coef = G_no_self_und.transitivity_local_undirected(mode="zero")
    df_pr = pd.DataFrame(data={'region name':G.vs['name'], 'pagerank':prs, 'pagerank_incl_loops': prs_incl_self,
                              'in_degree': indeg, 'in_degree_incl_loops': indeg_incl_self, 'out_degree': outdeg,
                              'out_degree_incl_loops': outdeg_incl_self, 'total_degree': totdeg,
                              'total_degree_incl_loops': totdeg_incl_self, 'closeness': cls, 'betweenness': btw,
                              'weighted_cluster_coef': np.array(weighted_clust_coef)/2.0, 
                               'cluster_coef': clust_coef})
    df_pr = df_pr.merge(df_regions, how='left', on='region name')
    return df_pr

# draw degree distribution
def draw_deg_dist(edgelist):
    G = nx.DiGraph()
    for edge in edgelist:
        if edge[2]!=0 and edge[0]!=edge[1]:
            G.add_weighted_edges_from([edge])
    #total
    deg_dict = {}
    for node, deg in G.degree(weight='weight'):
        deg_dict.update({node:deg})
    plt.figure(figsize=(10,7))
    sns.distplot(np.log(1 + np.array(deg_dict.values())), kde=True, rug=True, hist_kws={'alpha':0.5}, label='total')
    #in
    in_deg_dict = {}
    for node, deg in G.in_degree(weight='weight'):
        in_deg_dict.update({node:deg})
    sns.distplot(np.log(1 + np.array(in_deg_dict.values())), kde=True, rug=True, hist_kws={'alpha':0.5}, label='in')

    #out
    out_deg_dict = {}
    for node, deg in G.out_degree(weight='weight'):
        out_deg_dict.update({node:deg})
    sns.distplot(np.log(1 + np.array(out_deg_dict.values())), kde=True, rug=True, hist_kws={'alpha':0.5}, label='out')
    plt.xlabel('log(1 + Degree)')
    plt.ylabel('proportion')
    plt.legend()

# draw count vs degree plot for total, in, out degrees. Loops are excluded. Linear regression fit parameters are printed.
def draw_deg_count(edgelist):
    G = nx.DiGraph()
    for edge in edgelist:
        if edge[2]!=0 and edge[0]!=edge[1]:
            G.add_weighted_edges_from([edge])
    #total
    deg_dict = defaultdict(int)
    for node, deg in G.degree(weight='weight'):
        deg_dict[deg] += 1
    deg = list(deg_dict.keys())
    count = list(map(lambda x: deg_dict.get(x), deg))
    p_tot = np.polyfit(np.log(np.array(deg)+1), np.log(count), 1)
    plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15,5))
    plt.subplot(131)
    plt.scatter(np.log(deg), np.log(count))
    plt.xlabel('log(degree)')
    plt.ylabel('log(count)')
    plt.title('total degree')
    
    #in
    deg_dict = defaultdict(int)
    for node, deg in G.in_degree(weight='weight'):
        deg_dict[deg] += 1
    deg = list(deg_dict.keys())
    count = list(map(lambda x: deg_dict.get(x), deg))
    p_in = np.polyfit(np.log(np.array(deg)+1), np.log(count), 1)
    plt.subplot(132)
    plt.scatter(np.log(deg), np.log(count))
    plt.title('in degree')
    
    #out
    deg_dict = defaultdict(int)
    for node, deg in G.out_degree(weight='weight'):
        deg_dict[deg] += 1
    deg = list(deg_dict.keys())
    count = list(map(lambda x: deg_dict.get(x), deg))
    p_out = np.polyfit(np.log(np.array(deg)+1), np.log(count), 1)
    plt.subplot(133)
    plt.scatter(np.log(deg), np.log(count))
    plt.title('out degree')
    
    plt.show()
    print 'total degree: slope={}, intercept={}.'.format(p_tot[0], p_tot[1])
    print 'in degree: slope={}, intercept={}.'.format(p_in[0], p_in[1])
    print 'out degree: slope={}, intercept={}.'.format(p_out[0], p_out[1])

# detect communities using igraph's label propagation method. The community is attached to the df, which contains 'region name'
def community_detection(df, edgelist):
    G_no_self = create_graph(edgelist, self_edges=False)
    communities = G_no_self.community_label_propagation(weights='weight')
    sgs = communities.subgraphs()
    clust = [-1]*df.shape[0]
    for k in range(df.shape[0]):
        for i in range(len(sgs)):
            if df['region name'].iloc[k] in sgs[i].vs['name']:
                clust[k] = i
                break
    df['community'] = clust
    
# draw communities using folium
def draw_communities(df_regions, threshold=10, popup=True): # df_regions contains 'lon', 'lat', 'region name', 'community'.
    df_community_size = df_regions['community'].value_counts().reset_index()
    df_community_size = df_community_size[(df_community_size['community'] >=threshold) &(df_community_size['index']!=-1)]
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    picked = np.random.choice(colors.keys(), df_community_size.shape[0], replace=False)
    colors_dict = {df_community_size['index'].iloc[i]:picked[i] for i in range(df_community_size.shape[0])}
    
    m = folium.Map(location=[35.7596, -79.0193], zoom_start=6)
    names = df_regions['region name'].str.replace('_.+County_.+', '').str.replace('[^a-zA-Z]', '')
    if popup:
        for i in range(df_regions.shape[0]):
            if df_regions['community'].iloc[i] in colors_dict.keys():
                lat = df_regions['lat'].iloc[i]
                lon = df_regions['lon'].iloc[i]
                c = colors_dict[df_regions['community'].iloc[i]]
                folium.Circle(
                    radius=200,
                    location=[lat, lon],
                    #tooltip = names.iloc[i],
                    popup = folium.Popup(names.iloc[i],parse_html=True),
                    color=c
                ).add_to(m)
    else:
        for i in range(df_regions.shape[0]):
            if df_regions['community'].iloc[i] in colors_dict.keys():
                lat = df_regions['lat'].iloc[i]
                lon = df_regions['lon'].iloc[i]
                c = colors_dict[df_regions['community'].iloc[i]]
                folium.Circle(
                    radius=200,
                    location=[lat, lon],
                    color=c
                ).add_to(m)
    return m

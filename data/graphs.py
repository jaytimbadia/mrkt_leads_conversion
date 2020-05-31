import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import datetime as dt


mql = pd.read_csv("marketing_qualified_leads.csv")
close = pd.read_csv("olist_closed_deals.csv")

data = pd.merge(mql, close, how = "left", on = "mql_id")
data['lost'] = data['won_date'].isnull()

# print(data.head(50))
mql_origin = mql.groupby('origin').agg({'mql_id':"count"})
origin = list(mql_origin.index)


#Number of MQL by channels
plt.figure(figsize = (6,4))
plt.bar(mql_origin.index, mql_origin['mql_id'])
plt.xticks(rotation = 90)
plt.ylabel('Number of MQL')
plt.title("Number of MQL by channels", size =15)
plt.show()

plt.figure(figsize = (10,6))
fancy_plot = plt.subplot()

for i in origin:
    channel = mql[mql['origin'] == i]
    channel = channel.set_index(pd.to_datetime(channel['first_contact_date']))
    channel_agg = channel.groupby(pd.Grouper(freq = "M")).count().drop(axis = 1, columns =["first_contact_date",
                                                                                           "landing_page_id", "origin"])
    fancy_plot.plot(channel_agg.index, channel_agg, "-o", label = i)
fancy_plot.legend()
plt.title('Number of MQL by channels overtime', size = 15)
# plt.show()


#Opportunity won rate by channel
data_origin = data[["origin","lost", 'mql_id']]
origin_lost = data_origin.groupby(['origin', 'lost']).count()

percentage = []
for i in origin:
    pct = origin_lost.loc[i].loc[False][0]/(origin_lost.loc[i].loc[True][0]+origin_lost.loc[i].loc[False][0])
    percentage.append(pct)

plt.figure(figsize = (7,5))
plt.bar(origin, percentage)
plt.xticks(rotation = 90)
plt.ylabel('won rate')
plt.title("Won rate by channel", size = 15)
plt.show()


# Number of MQL and Won Rate by landing page
mql_lp = mql.groupby('landing_page_id').agg({'mql_id':"count"})
mql_lp = mql_lp[mql_lp['mql_id'] > 30]
data_lp = pd.merge(data, mql_lp, how = "inner", left_on = "landing_page_id", right_index = True)
#ata_lp = data_lp[data_lp['lost']]
lp_lost = data_lp.groupby(['landing_page_id', 'lost']).agg({'mql_id_x':"count"})
landing_page = list(mql_lp.index)

percentage_lp = []
landing_page_2 = []
Num_mql = []
for i in landing_page:
    if mql_lp.loc[i][0] == lp_lost.loc[i].loc[True][0]:
        lp_lost.drop([i])
    else:
        pct = lp_lost.loc[i].loc[False][0]/(lp_lost.loc[i].loc[True][0]+lp_lost.loc[i].loc[False][0])
        percentage_lp.append(pct)
        landing_page_2.append(i)
        Num_mql.append(mql_lp.loc[i][0])

# Landing Page number of MQL and Won rate
fig = plt.figure(figsize = (10,4))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.bar(landing_page_2, percentage_lp, label = "Won rate")
ax2.plot(landing_page_2, Num_mql, color = "red", label = "number of MQL")
ax.set_ylabel('won rate')
ax.legend()
ax2.set_ylabel('number of MQL')
ax2.legend()
plt.title("Number of MQL and Won Rate by landing page", size = 15)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.show()
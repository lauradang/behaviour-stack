#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
import re, os

# Accessing the data file 
emdirectory = r'/Users/lauradang/RyersonHacks/data/emotions.xlsx'
gendirectory = r'/Users/lauradang/RyersonHacks/data/gender.xlsx'
agedirectory = r'/Users/lauradang/RyersonHacks/data/age.xlsx'

df = pd.read_excel(emdirectory)
dgen = pd.read_excel(gendirectory)
dage = pd.read_excel(agedirectory)

# In[ ]:


def get_max_emotion(name, date):
    name_column = df[name]
    date_column = df[df.columns[0]]
    new_max = 0
    date = 0
    count = 0

    for count, index in enumerate(name_column):
        if index > new_max: 
            new_max = index
            date = date_column.iloc[count]
            
    return date


# In[ ]:


def get_max_gender(gender, date):
    gender_column = dgen[gender]
    date_column = dgen[dgen.columns[0]]
    new_max = 0
    count = 0
    date = 0
    
    for count, index in enumerate(gender_column):
        if index > new_max:
            new_max = index
            date = date_column.iloc[count]
            
    return date


# In[34]:


#get_max('Sad', 'Date')
print("The date for the saddest day is: " + str(get_max_emotion('Sad', 'Date')))
print("The date for the happiest day is: " + str(get_max_emotion('Happy', '')))
print("The date for the most disgusting day is: " + str(get_max_emotion('Disgust', 'Date')))
print("The date for the scariest day is: " + str(get_max_emotion('Fear', 'Date')))
print("The date for the most neutral day is: " + str(get_max_emotion('Neutral', 'Date')))
print("The date for the angriest day is: " + str(get_max_emotion('Anger', 'Date')))
print("The date for the most surprising day is: " + str(get_max_emotion('Surprise', 'Date')))



print("The date that had the most females was: " + str(get_max_gender('Female', '')))
print("The date that had the most males was: " + str(get_max_gender('Male', '')))

#print("The most popular age group on this day was: get_max_age('Date')


# In[32]:


#Creating the Graphs and the Charts for the data 


plt.text(1.5, 1.5,'Report for Loblaws- Advertisement: ',
     horizontalalignment='center',
     verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) #This creates text

plt.text(2.0, 2.0, "The happiest ")


title("Emotion Frequency for Ad")
xlabel('Emotion')
ylabel('Number of people')


emotions = {'happy', 'sad', 'tired', 'bored', 'happy'}

pdf = FPDF()
pdf.add_page()
pdf.set_font('arial', 'B', 12)

labels = ['Happiness', 'Sadness', 'Anger', 'Neutral']
sizes = [38.4, 40.6, 20.7, 10.3]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[26]:


import xlrd

path = r'/Users/lauradang/RyersonHacks/data/emotions.xlsx'
output_name = "boba.png"


def generate_barchart(xlsx_path, output_name):

   df = pd.read_excel(xlsx_path)

   columns = []
   sums = []

   is_date = True
   for col in df.columns:
       if (is_date): is_date = False
       else:
           sum = 0
           columns.append(col)
           for f in df[col]:
               sum += int(f)
           sums.append(sum)

   df = pd.DataFrame()
   df['Emotions'] = columns
   df['Frequency'] = sums

   title("How often each emotion is experienced")
   xlabel('Emotions')
   ylabel('Number of Occurrences')

   ncol = len(columns)
   c = []
   for i in range(ncol):
       c.append(i * 2 + 2)

   xticks(c, df['Emotions'])
   bar(c, df['Frequency'], width=0.75, color="#eb879c", label="Frequency")

   legend()
   savefig(output_name)

   # basic formatting
   pdf = FPDF()
   pdf.add_page()
   pdf.set_xy(0, 0)
   pdf.set_font('arial', 'B', 10)

   pdf.image(output_name, x = None, y = None, w = 0, h = 0, type = '', link = '')

   #pdf.output('test.pdf', 'F')

generate_barchart(path, output_name)


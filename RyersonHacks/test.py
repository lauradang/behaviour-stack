import pandas as pd
import matplotlib
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
import xlrd
from pylab import *

def generate_chart(xlsx_path, output_bar, output_pie, cate, titl1, title2):
    
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

    title(titl1)
    xlabel(cate)
    #ylabel('Number of Occurrences')

    ncol = len(columns)
    c = []
    for i in range(ncol):
        c.append(i * 2 + 2)

    xticks(c, columns)
    bar(c, sums, width=0.75, color="#eb879c", label="Frequency", align='center')

    legend()
    savefig(output_bar)
    plt.clf()
    
    ### 
    
    plt.text(1.5, 1.5,'Report for Loblaws- Advertisement: ',
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) #This creates text
    
    title(title2)
    xlabel(cate)
    ylabel('Number of people')
    
    labels = columns
    sizes = sums
    
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'mintcream', 'tan', 'darkgray']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    savefig(output_pie)
    plt.clf()    

path = r"/Users/lauradang/RyersonHacks/data/gender.xlsx"
output = "hatelife.png"
outputp = "hatelifep.png"
generate_chart(path, output, outputp, 'Gender', 'How often each gender viewed', 'Percentage of gender')
path2 = r"/Users/lauradang/RyersonHacks/data/age.xlsx"
output2 = "stillhatinglife.png"
output2p = "stillhatinglifep.png"
generate_chart(path2, output2, output2p, 'Age', 'How often each age group viewed', 'Percentage of age')
path1 = r"/Users/lauradang/RyersonHacks/data/emotions.xlsx"
output3 = "hatinglife.png"
output3p = "hatinglifep.png"
generate_chart(path1, output3, output3p, 'Emotions', 'How often each emotion is experienced', 'Percentage of emotion')
output_pie = "pie.png"

# Accessing the data file 


df = pd.read_excel(path1)
dgen = pd.read_excel(path)
dage = pd.read_excel(path2)    



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


def get_max_gender(gender, date):
    gender_column = dgen[gender]
    date_column = df[df.columns[0]]
    new_max = 0
    count = 0
    date = 0

    for count, index in enumerate(gender_column):
        if index > new_max:
            new_max = index
            date = date_column.iloc[count]

    return date



def get_max_age(date):
    #age_column = dage[age]
    date_column = df[df.columns[0]]
    date_max = 0
    position_i = 0
    position_j = 0
    counter = 0
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1] 
    
    returnlst = []
    
    for i in range(0, count_row):
        
        for j in range (1, count_col):
            
            if dage.iloc[i, j] > date_max:
                date_max = int(dage.iloc[i, j])
                position_i = i
                position_j = j
                
            if j == count_col - 1:
                strreturn = str("The date at: " + str(dage.iloc[i, 0]) + " The greatest number is: " + str(date_max) +
                     " of the age group " + dage.columns[position_j])
                returnlst.append(strreturn)
                date_max = 0
                position_i = 0
                position_j = 0
                
    return returnlst



def generate_text(output_txt):
    
    #get_max('Sad', 'Date')
    t1 = ("saddest day: " + str(get_max_emotion('Sad', 'Date')))
    t2 = ("happiest day: " + str(get_max_emotion('Happy', 'Date')))
    t3 = ("disgusted day: " + str(get_max_emotion('Disgust', 'Date')))
    t4 = ("scariest day: " + str(get_max_emotion('Fear', 'Date')))
    t5 = ("neutralest day: " + str(get_max_emotion('Neutral', 'Date')))
    t6 = ("angriest day: " + str(get_max_emotion('Anger', 'Date')))
    t7 = ("surprisingest day: " + str(get_max_emotion('Surprise', 'Date')))
    
    t8 = ("most females day: " + str(get_max_gender('Female', 'Date')))
    t9 = ("most males day: " + str(get_max_gender('Male', 'Date')))    
    

    
    plt.text(1.5, 1.5, t1,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t2,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t3,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t4,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t5,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t6,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t7,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t8,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5)) 
    plt.text(1.5, 1.5, t9,
         horizontalalignment='center',
         verticalalignment='center', bbox=dict(facecolor='pink', alpha=0.5))     
    
    savefig(output_txt)
    plt.clf()     


outputtxt = "txtihate.png"
generate_text(outputtxt)

t1 = ("Saddest day is: " + str(get_max_emotion('Sad', 'Date')))
t2 = ("Happiest day is: " + str(get_max_emotion('Happy', 'Date')))
t3 = ("The most disgusting day is: " + str(get_max_emotion('Disgust', 'Date')))
t4 = ("Scariest day is: " + str(get_max_emotion('Fear', 'Date')))
t5 = ("Most neutral day is: " + str(get_max_emotion('Neutral', 'Date')))
t6 = ("Angriest day is: " + str(get_max_emotion('Anger', 'Date')))
t7 = ("Most surprising day is: " + str(get_max_emotion('Surprise', 'Date')))



t8 = ("The date that had the most females was: " + str(get_max_gender('Female', 'Date')))
t9 = ("The date that had the most males was: " + str(get_max_gender('Male', 'Date')))

t10_list =  get_max_age('Date')



    
if __name__ == '__main__':
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=20)

    pdf.image(output, x = 120, y = 60, w = 70)
    pdf.image(outputp, x = 30, y = 60, w = 70)
    pdf.image(output3, x = 120, y = 140, w = 70)
    pdf.image(output3p, x = 30, y = 140, w = 70)
    pdf.image(output2, x = 120, y = 240, w = 70)
    pdf.image(output2p, x = 30, y = 240, w = 70)
    #pdf.image(outputtxt, x = 30, y = 200, w = 100)
    
    
    pdf.multi_cell(0, 5, "REPORT ON EMOTIONS AND DEMOGRAPHICS")
    pdf.set_font("Arial", size=8)
    pdf.set_y(20)
    pdf.multi_cell(0, 5, t1, 1)
    pdf.multi_cell(0, 5, t2, 1)
    pdf.multi_cell(0, 5, t3, 1)
    pdf.multi_cell(0, 5, t4, 1)
    pdf.multi_cell(0, 5, t5, 1)
    pdf.multi_cell(0, 5, t6, 1)
    pdf.multi_cell(0, 5, t7, 1)
    pdf.set_y(120)
    pdf.multi_cell(0, 5, t8, 1)
    pdf.multi_cell(0, 5, t9, 1)
    pdf.set_y(200)

    for i in t10_list:
        pdf.multi_cell(0, 5, i, 1)  
        
    
    pdf.output("add_image.pdf")
    
    
    #pdf.ln(85)  # move 85 down
     
    
    #add_image(output_name_emotion, 20)
    #add_image(output_name_age, 120)
    #add_image(output_name_gender, 220)
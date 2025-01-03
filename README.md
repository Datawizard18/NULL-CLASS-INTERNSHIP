import pandas as pd

#Load the data set

data = pd.read_csv('C:/Users/Sindhuja/Downloads/Google-Play-Store-Analytics-main/Google-Play-Store-Analytics-main/heart-disease.csv')

data

print(data)

print(data.head())

data.head()

print(data.tail())

data.tail()

data.type

data.describe()

data.info()

import numpy as np

data.isnull().sum()

df=pd.read_csv('C:/Users/Sindhuja/Downloads/Google-Play-Store-Analytics-main/Google-Play-Store-Analytics-main/Global EV Data 2024.csv')

print(df.isnull().sum())

data['chol'] = data['chol'].replace(np.nan,data['chol'].mean())


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

xpoints = np.array([1,2,6,8])
ypoints = np.array([3,8,1,10])
plt.plot(xpoints, ypoints)
plt.show()


#plotting the pie chart

y=np.array([35,25,25,15])
plt.pie(y)
plt.show()

x=np.random.normal(170,10,250)
print(x)

plt.hist(x)
plt.show()

sns.distplot([0,1,2,3,4,5])
plt.show()


#DATA CLEANING 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os

apps_df = pd.read_csv("C:/Users/Sindhuja/Desktop/SINDHUJA DATA ANALYSIS/GOOGLE PLAY STORE DATA ANALYTICS/Google-Play-Store-Analytics-main/Google-Play-Store-Analytics-main/Play Store Data.csv")
reviews = pd.read_csv("C:/Users/Sindhuja/Desktop/SINDHUJA DATA ANALYSIS/GOOGLE PLAY STORE DATA ANALYTICS/Google-Play-Store-Analytics-main/Google-Play-Store-Analytics-main/User Reviews.csv")
apps_df.head()
apps_df.tail()
reviews.head()
reviews.tail()
apps_df.isna().sum()
reviews.isna().sum()


#DATA CLEANING 
#Handling missing values,Removing duplicates,filtering ratings.
#cleaning the reviews DataFrame by dropping rows with missing translated reviews.
aaps_df = apps_df.dropna(subset = ['Rating'])
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column]. mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df= apps_df= apps_df[apps_df['Rating'] <=5]
reviews.dropna(subset= ["Translated_Review"], inplace = True)






apps_df.dtypes
reviews.dtypes


#convert the installs columns to numeric by removing commas and +
#This code cleans the 'Installs' column in apps_df by removing commas and plus signs, 
#then converts the values to integers for numerical analysis.
apps_df['Installs'] = apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)

#convert price column to numeric after removing $
apps_df['Price'] = apps_df['Price'].str.replace('$','').astype(float)
#This code cleans the 'Price' column in apps_df by removing the dollar sign and converting the values to floating-point numbers

merged_df = pd.merge(apps_df,reviews,on='App', how='inner')
merged_df.head()
#This code performs an inner join between apps_df and reviews on the 'App' column, 
#creating a new DataFrame merged_df that contains only matching records from both datasets.

#DATA TRANSFORMATION 
def convert_size(Size):
    if 'M' in Size:
        return float(Size.replace('M',''))
    elif 'K' in Size:
        return float(Size.replace('K',''))/1024
    else:
        return np.nan
# Apply the function to the 'Size' column in your DataFrame
apps_df['Size'] = apps_df['Size'].apply(convert_size) 
#This code transforms the 'Size' column in apps_df by converting sizes from 
#string format (in megabytes and kilobytes) into numeric values in megabytes using the convert_size function.


#LOGRARITHMIC
apps_df['Log_Installs']= np.log(apps_df['Installs'])
apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Log_Reviews'] = np.log(apps_df['Reviews'])
apps_df.dtypes
#This code calculates the logarithmic values of the 'Installs' and 'Reviews' columns in apps_df, 
#stores them in new columns, converts 'Reviews' to integers, and outputs the data types of all columns in the DataFrame.

#RATING COLUMN TRANSFORMATION
def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['rating_Group']= apps_df['Rating'].apply(rating_group)
#his code defines a function rating_group that categorizes the 'Rating' values in the apps_df DataFrame into groups 
#('Top rated app', 'Above average', 'Average', or 'Below Average') based on their numerical rating

#REVENUE COLUMN
apps_df['Revenue'] = apps_df['Price']*apps_df['Installs']
#This code calculates the total revenue for each app in the apps_df DataFrame by multiplying 
#the 'Price' of the app by the number of 'Installs' and stores the result in a new column called 'Revenue'.

#SENTIMENT ANALYZER
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

#Polarity Scores in SIA
#Positive, Negative, Neutral and Compound

review = "This app is amazing! I Love The new featurs."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)

review = " THIS APP IS NOT GOOD"
sentiment_score = sia.polarity_scores(review)
print(sentiment_score)

review = "App features are not upto the standards its difficult to open the app some times, taking so much of time to open, I dont like the new features"
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


reviews['Sentiment_score'] = reviews['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
reviews.head()


apps_df['Last Updated'] =pd.to_datetime(apps_df['Last Updated'], errors= 'coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year


#PLOTLY - 1
import plotly.express as px
fig=px.bar(x=["A","B","C"], y=[1,3,2],title="Sample Bar Chart")
fig.show()
import plotly.express as px
fig = px.bar(x=["A", "B", "C"], y=[1, 3, 2], title="Sample Bar Chart")
fig.show()


import os

# Set the desired directory path where files will be saved
html_files_path = "C:/Users/Sindhuja/Documents/plotly_graphs/"  # Replace this with your actual desired path

# Create the directory if it doesn't exist
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

plot_containers = ""

# Function to save the plot as an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)  # Join the path and filename
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    
    # Append the plot and insight to the plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
         <div class="plot">{html_content}</div>
         <div class="insights">{insight}</div>
    </div>
    """
    # Save the figure to the specified file path
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


#Fig 1 Generate bar chart for top categories 
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Viridis,
    width=400,
    height=300
)

# Update layout settings
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save the plot as HTML and display the description
fig1.write_html("category_graph_1.html", auto_open=True)
print("The top categories on the play store are dominated by tools, entertainment, and productivity apps.")


#PLOT 2: PIE CHART Paid apps and free apps percentage 
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)

fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig2.write_html("Type_Graph_2.html", auto_open=True)
print("Most apps on the playstore are free, indicating a strategy to attract users first and monetize the apps later.")


#Fig 3 Ratings histogram counts 
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)

fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig3.write_html("Rating_Graph_3.html", auto_open=True)
print("Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.")



#FIG 4 sentiment scores minus side & plus side 
Sentiment_counts = reviews['Sentiment_score'].value_counts()  # Assuming 'Sentiment_score' is the correct column
fig4 = px.bar(
    x=Sentiment_counts.index,
    y=Sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=Sentiment_counts.index,  # Use the correctly defined variable
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)

fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig4.write_html("Sentiment_Graph_4.html", auto_open=True)
save_plot_as_html(fig4," Sentiment Graph 4.html","sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")



#FIG 5
# Define 'installs_by_category' to get the top 10 categories by installs
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)

# Create a bar plot using Plotly Express
fig5 = px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,  # Use the correctly defined variable
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)

# Update layout for better appearance
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML
fig5.write_html("Installs_Graph_5.html", auto_open=True)

# Save with a description (ensure 'save_plot_as_html' is defined or removed if not necessary)
# Uncomment if 'save_plot_as_html' is a defined function in your code
# save_plot_as_html(fig5, "Installs_Graphs_5.html", "The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage")


#fig 6 Line chart no.of updates as per the year 
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=  ['#AB63FA'],
    width=400,
    height=300
)

fig6.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)


# Save plot as HTML with a description
fig6.write_html("Sentiment_Graph_4.html", auto_open=True)
save_plot_as_html(fig6," updates_per_year.html","Updates have been increasing over the years,showing that developers are actively maintaining and improving their apps")


## Fig 7 which category has generated more revenue
revenue_by_categry = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_categry.index,  # Corrected variable
    y=revenue_by_categry.values,  # Corrected variable
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category', 
    color=revenue_by_categry.index,  # Use the correctly defined variable
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)

fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig7.write_html("Revenue_by_Category_7.html", auto_open=True)
save_plot_as_html(fig7, "Revenue_Graph_7.html", "Categories such as Business and productivity lead in revenue generation, indicating their monetization potential.")



#Fig 8 bar Top genres in bar plot 
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,  # Corrected variable
    y=genre_counts.values,  # Corrected variable
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres', 
    color=genre_counts.index,  # Use the correctly defined variable
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)

fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig8.write_html("Genre_Graph_8.html", auto_open=True)
save_plot_as_html(fig8, "Genre_Graph_8.html", "Action and casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")



#Fig 9 Scatter plot between the last updated and rating 
fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,  # Use the correctly defined variable
    width=400,
    height=300
)

fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig9.write_html("Last_updated_9.html", auto_open=True)
save_plot_as_html(fig9, "Last_Updated_9.html", "The Scatter PLot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings.")


#FIG 10 Box plot between the Type and Ratings
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,  # Corrected to a valid color sequence
    width=400,
    height=300
)

fig10.update_layout(  # Updated from fig9 to fig10 to match the figure being created
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size': 16},
    xaxis=dict(title_font={'size': 12}),
    yaxis=dict(title_font={'size': 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML with a description
fig10.write_html("Box_Plot_10.html", auto_open=True)
save_plot_as_html(fig10, "Box_Plot_10.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")

plot_containers_split=plot_containers.split('</div>')

if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers


###DASHBOARD HTML 
dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
     <meta charset="UTF-8">
     <meta name=viewport" content="width=device,initial-scale-1.0">
     <title> Google Play Store Review Analytics</title>
     <style>
        body {{
            font-family:Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
          }}
         .header {{
             display: flex;
             align-items: center;
             justify_content: center;
             padding:20 px;
             background-color: #444
         }}
         .header img{{
             margin:0 10px;
             height:50 px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20 px;
        }}
        .plot-container {{
            border: 2px solid #555
            margin: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
       }}
       .insights {{
           display: none;
           position: absolute;
           right: 10px;
           top: 10px,
           background-color: rgba(0,0,0,0.7);
           padding: 5px;
           border_radius: 5px;
           color: #fff;
       }}
       .plot_container: hover .insights{{
           display:block;
       }}
       </style>
       <script>
           function openplot(filename) {{
               window.open(filename, '_blank');
               }}
       </script>
      </head>
      <body>
        <div class "header">
            <img src= "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-logo_2013_Google.png" alt="Google Logo" >
            <h1>Google Play Store Reviews Analytics<h1>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024" alt="Google Logo">
       </div>
       <div  Class="container">
             {plots}
       </div>
     </body>
     </html>
     """

final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)

dashboard_path=os.path.join(html_files_path,"web page.html")

with open(dashboard_path,"w", encoding="utf-8") as f:
      f.write(final_html)
webbrowser.open('file://'+os.path.realpath(dashboard_path))
             






    
    
    









#19/11/2024
# EXECUTING FIRST TASK ON 19/11/2024
#1. Filetering the 5 star reviews from health and fitness category apps
filtered_df = merged_df[(merged_df['Category'] == 'HEALTH_AND_FITNESS') & (merged_df['Rating'] == 5)]
#REMARKS FOR FILTERING FOR 5 STAR RATING  AS I TRIED TO FILTER THE RATING TO 5 STAR NO REVIEWS / APPS ARE RATED
# 5 STAR, WHEN I CHANGED THE NUMBER TO THE GREATER THAN 4.0 2578 COLUMNS WITH 17 ROWS WAS REFLECTED.
#REMARKS 
#1. AS THE APPS_DF AND USER REVIEWS DATA SETS ARE BOTH MERGED FIRST AND THEN LATER FILTERING ONLY THE 5 STAR
#2.RATING APPS FROM HEALTH & FITNESS CATEGORY, HAS GIVEN ZERO ROWS WITH 17 COLUMNS GOT REFLECTED.
#3.RATINGS PRESENT IN THE APPS_DF COLUMN HAS 5 STAR RATINGS GIVEN FOR HEALTH AND FITNESS CATEGORY APPS.
#4.AFTER MERGING BOTH DATASETS AND APPLYING THE FILTER, THERE ARE NO 5-STAR APPS IN THE HEALTH & FITNESS CATEGORY
#WITH REVIEWS IN THE TRANSLATED REVIEW COLUMN.AS A RESULT,NO 5-STAR RATED APPS ARE AVAILABLE FOR GENERATING A WORD CLOUD 
#FROM THE TRANSLATED REVIEWS
#5.AS I CHANGED THE RATINGS FILTER TO 4.0 AND ABOVE 2578 ROWS AND 17 COLUMNS GOT REFLECTED,4.0 TO 4.8 TRANSLATED REVIEWS ARE PRESENT.


#EXECUTING THE SECOND TASK WITH FILTERS 
def extract_android_version(version):
    try:
        # Split on '.' and take the first part (if it exists) and convert it to a float
        return float(version.split()[0].split('.')[0])
    except Exception as e:
        return None  # Return None if version extraction fails

# Apply the function to the 'Android Ver' column
apps_df['Android Ver'] = apps_df['Android Ver'].apply(lambda x: extract_android_version(str(x)) if isinstance(x, str) else None)


filtered_df = apps_df[
    (apps_df['Installs'] >= 10000) &            # Installs > 10,000
    (apps_df['Size'] > 15) &                  # Size > 15MB
    (apps_df['Content Rating'] == 'Everyone') & # Content Rating = 'Everyone'
    (apps_df['Android Ver'] > 4.0) &            # Android Version > 4.0
    (apps_df['App'].str.len() <= 30) &           # App Name length <= 30 characters
    (apps_df['Price'] > 10)                     # Price > $10
]

#REMARKS
#OBJECTIVE:
#THE TASK IS TO CREATE A DUAL-AXIS CHART COMPARING FREE VS PAID APPS BASED ON THE FOLLOWING CRITERIA:

#FILTERING CRITERIA:
#THE DATA SHOULD MEET THE FOLLOWING CONDITIONS:

#INSTALLS GREATER THAN 10,000.
#APP SIZE GREATER THAN 15 MB.
#CONTENT RATING SHOULD BE "EVERYONE".
#ANDROID VERSION SHOULD BE ABOVE 4.0.
#APP NAME LENGTH SHOULD BE LESS THAN 30 CHARACTERS.
#PRICE SHOULD BE GREATER THAN $10 (FOR PAID APPS).
#OUTCOME OF FILTERING:
#AFTER APPLYING THE ABOVE FILTERS, NO DATA VALUES MET THE SPECIFIED REQUIREMENTS.

#CONCLUSION:
#AS A RESULT, THE DUAL-AXIS CHART CANNOT BE GENERATED DUE TO THE ABSENCE OF QUALIFYING DATA AFTER APPLYING THESE FILTERS.


INTERNSHIP SECOND TASK IS TO CREATE DUAL AXIS CHART
#THE TASK WAS TO CREATE A DUAL-AXIS CHART COMPARING FREE VS PAID APPS BASED ON SPECIFIC FILTERING CRITERIA

#FILTERING CRITERIA
#THE FOLLOWING CONDITIONS WERE APPLIED TO FILTER THE DATA:
#INSTALLS: APPS SHOULD HAVE A SIZE GREATER THAN 15 MB
#CONTENT RATING: THE CONTENT RATING SHOULD BE "EVERYONE".
#ANDROID VERSION: THE ANDROID VERSION SHOULD BE GREATER THAN 4.0.
#APP NAME LENGTH: THE LENGTH OF THE APP NAME SHOULD BE 30 CHARACTERS OR FEWER.
#PRICE: APPS SHOULD HAVE A PRICE GREATER THAN $10(FOR PAID APPS).
#OUTCOME OF FILTERING:
#AFTER APPLYING ALL THE ABOVE FILTERS,NO DATA VALUES MET THE SPECIFIED REQUIREMENTS.
#IN OTHER WORDS,NO APPS QUALIFIED BASED ON THE GIVEN CONDITIONS.
#CONCLUSION:
#AS A RESULT, DUE TO THE ABSENCE OF QUALIFYING DATA AFTER APPLYING THESE FILTERS,
#IT IS NOT POSSIBLE TO GENERATE THE DUAL_AXIS CHART COMPARING FREE VS PAID APPS.
#THE FILTERING CONDITIONS WERE TOO STRINGENT, AND NO APPS IN THE DATASET MET THE CRITERIA FOR INCLUSION.




INTERNSHIP THIRD TASK WITH SUMMARY 
#EXECUTING THE THIRD TASK CHOROPLETH 
#FOR A CHOROPLETH MAP TO WORK CORRECTLY,IT IS CRUCIAL THAT THE DATA POINTS ARE ASSOCIATED WITH SPECIFIC
#GEOGRAPHIC LOCATIONS,TYPICALLY COUNTRIES (OR REGIONS,STATES, etc..). THE MAP NEEDS TO LINK THE DATA
#(SUCH AS POPULATION, GDP, OR ANY OTHER VARIABLE) TO GEOGRAPHIC BOUNDRIES. THIS IS USUALLY DONE USING
#COUNTRY NAMES(e.g. "UNITED STATES", "INDIA","AUSTRALIA") OR COUNTRY CODES (e.g., "US","IN","BR").
#THESE IDENTIFIERS ALLOW THE MAP SOFTWARE TO ACCURATELY PLACE THE DATA IN THE CORRECT GEOGRAPHIC AREA ON THE MAP.

#WITHOUT THESE KEY PIECES OF INFORMATION-COUNTRY NAMES OR COUNTRY CODES-THE DATA CANNOT BE MAPPED TO THE CORRESPONDING
#COUNTRIES,MAKING IT IMPOSSIBLE TO CREATE AN ACCURATE CHOROPLETH MAP. ESSENTIALLY, WITHOUT KNOWING WHERE THE DATA SHOULD BE APPLIED
#GEOGRAPHICALLY, THE MAPPING PROCESS CANNOT PROCEED.

#IN SUMMARY:
#THE DATA NECESSARY FOR BUILDING A CHOROPLETH MAP IS LACKING THE CRUCIAL GEOGRAPHIC IDENTIFIERS:
#COUNTRY NAMES OR COUNTRY CODES ARE ESSENTIAL TO LINK THE DATA WITH ITS RESPECTIVE LOCATIONS ON THE MAP.
#WITHOUT THIS LINKAGE,THE CHOROPLETH MAP CANNOT BE CREATED BECAUSE THE SOFTWARE WILL NOT KNOW WHERE TO ASSIGN THE DATA GEOGRAPHICALLY.
    
    
#INTERNSHIP 4TH TASK IS TO CREATE HEATMAP
apps_df[apps_df["Genres"].str.contains(r"(A|F|E|G|I|K)", case=False, na=False)
new_df=apps_df[~apps_df["Genres"].str.contains(r"(A|F|E|G|I|K)", case=False, na=False)]
new_df2=(new_df[new_df['Installs'] >= 100000])
new_df2.head()
new_df(new_df['Reviews'] >= 1000)
new_df[new_df['Reviews'] >= 1000]
new_df['Reviews'] = new_df['Reviews'].str.replace(',','').str.replace('+','').astype(int)
INTERNSHIP FOURTH TASK WITH SUMMARY
#UPDATED WITHIN THE LAST YEARR(2023): THE DATASET ONLY INCLUDES APPS FROM 2016 TO 2018,
#SO THE FILTER FOR APPS UPDATED IN 2013 IS NOT APPLICABLE IN THIS CASE, AS NONE OF THE APPS MEET THIS CRITERIA.
#INSTALLS > 100,000:ONLY APPS WITH AT LEAST 100,000 INSTALLS WERE CONSIDERED.
#THIS IS IMPLEMENTED USING new_df2 = new_df[new_df['Installs'] >= 100000].
#REVIEWS COUNT > 1,000: APPS THAT HAVE FEWER THAN 1000 REVIEWS WERE EXCLUDED.THE REVIEWS COLUMN IS CLEANED BY REMOVING COMMAS
#AND PLUS SIGNS, AND THEN CONVERTED TO INTEGERS USING new_df['Reviews'] = new_df['Reviews'].str.replace(',','').str.replace('+','').astype(int).
#GENRES NOT STARTING WITH A,F,E,G,I,K: APPLICATIONS WHOSE GENRE STARTS WITH ANY OF THESE LETTERS ARE EXCLUDED
USING REGULAR EXPRESSIONS IN new_df = apps_df[~apps_df["Genres"].str.contains(r"(A|F|E|G|I|K)", case=False, na=False)].
SUMMARY
#NO APPS UPDATED IN THE LAST YEAR(2023): SINCE THE DATASET ONLY CONTAINS DATA DROM 2016 TO 2018,
#THE FILTER TO INLCUDE ONLY APPS UPDATED IN 2023 DOESNT MATCH ANY RECORS, WHUCH MEANS NO DATA IS AVAILABLE FOR THIS PART OF THE ANALYSIS
#DATA CLEANING: THE 'REVIEWS' COLUMN CONTAINED NON-NUMERIC VALUES SUCH AS COMMAS AND PLUS SIGNS.
#THESE HAD TO BE CLEANED AND CONVERTED TO INTEGERS FOR ANALYSIS
#WE NEED A RECENT DATASET THAT INCLUDES APPS UPDATED WITHIN THE LAST YEAR.
#THE HEATMAP VISUALIZATION WOULD BE EFFECTIVE FOR EXAMINING CORRELATIONS ONCE THE DATASET CONTAINS APPS THAT MEET ALL THE FILTERING CRITERIA.
 
# 5 TH TASK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
!pip install datetime
from datetime import datetime

playstore_df=pd.read_csv("C:/Users/Sindhuja/Desktop/SINDHUJA DATA ANALYSIS/GOOGLE PLAY STORE DATA ANALYTICS/Google-Play-Store-Analytics-main/Google-Play-Store-Analytics-main/Play Store Data.csv")
playstore_df.head()
df1=playstore_df[(playstore_df['Content Rating'] == 'Teen') & (playstore_df['App'].str.match(r"^[Ee]"))] 
df1["Installs"] = df1["Installs"].str.replace(r"[+,]", "", regex=True)
df1["Installs"] = df1["Installs"].astype(int)
df1 = df1[df1['Installs'] > 10000]
df1['Last Updated'] = pd.to_datetime(df1['Last Updated'])  
df1['Month_Year'] = df1['Last Updated'].dt.to_period('M')
df1['Month_Year'] = df1['Month_Year'].dt.strftime('%Y-%m')
df1=df1.drop_duplicates()
df1=df1.sort_values(by=["Category","Last Updated"], ascending=True)
result = df1.groupby(["Month_Year", "Category"])["Installs"].sum().reset_index().sort_values("Month_Year")
result["MoM Growth"] = result.groupby("Category")["Installs"].pct_change() * 100
print(result)
categories = result["Category"].unique()
categories
current_hour = datetime.now().hour

if 16 <= current_hour <= 20:
    plt.figure(figsize=(12, 8))
    
    for category in categories:
        # Filter data for the current category
        category_data = result[result["Category"] == category]
        
        # Convert Month_Year to datetime or period if not already
        category_data["Month_Year"] = pd.to_datetime(category_data["Month_Year"], format='%Y-%m')
        
        # Sort by Month_Year
        category_data = category_data.sort_values(by="Month_Year")
        
        # Plot the data
        plt.plot(
            category_data["Month_Year"],
            category_data["Installs"],
            label=category
        )
        
        # Highligh growth
        significant_growth = category_data["MoM Growth"] > 20
        plt.fill_between(
            category_data["Month_Year"],
            category_data["Installs"],
            where=significant_growth,
            color="black",
            alpha=1,
            label=f"Significant Growth ({category})"
        )
    
    # graph formatting
    plt.title("Time Series Chart: Total Installs by App Category", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Total Installs", fontsize=14)
    plt.legend(title="App Category")
    plt.grid(True)
    plt.tight_layout()

    # Display the graph
    plt.show()
else:
    print("This script is configured to run only between 4 PM and 8 PM.")

#INFERENCE:
#1.DROPED THE DUPLICATES EVEN AFTER FILTERING THE DATA.
#2.EVEN AFTER REMOVING DUPLICATES, STILL OBSERVED SIMILAR KIND OF RECORDS FOR THE SAME CATEGORY ON THE SAME DATE
#3.AS PER THE STATISTICS WE DIDN'T OBSERVE DATA FOR CONJUGATIVE PERIOD SO HERE IS THE TIME SERIES LINE CHART WITH THE TREND OF TOTAL 
#INSTALLS SEGMENTED BY APP CATEGORY GRAPH PLOTTED.
#4.SO WE OBSERVED THE RISE IN 20% (GAME CATEGORY AND AS WELL AS FAMILY CATEGORY)





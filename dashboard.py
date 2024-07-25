import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

save_location = './plots/'

# Load the JSON data
with open('email_data.json', 'r') as file:
    data = json.load(file)

# Convert JSON data to DataFrame
contacts = data['contacts']
df = pd.DataFrame(contacts)

# Ensure 'duration_known' is a numeric column
df['duration_known'] = df['duration_known'].fillna(0).astype(float)

# Set up the seaborn style
sns.set(style="whitegrid")

# Bar Chart of Number of Interactions
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='number_of_interactions', data=df, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Interactions per Contact')
plt.xlabel('Contact')
plt.ylabel('Number of Interactions')
plt.tight_layout()
plt.savefig(save_location + 'number_of_interactions.png')  # Save the figure
plt.show()

# Histogram of Email Rate
plt.figure(figsize=(8, 6))
sns.histplot(df['email_rate'], bins=10, kde=True, color='blue')
plt.title('Distribution of Email Rate')
plt.xlabel('Email Rate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(save_location + 'email_rate_distribution.png')  # Save the figure
plt.show()

# Pie Chart of Sentiment Analysis
sentiment_counts = df['sentiment_analysis'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribution of Sentiment Analysis')
plt.savefig(save_location + 'sentiment_analysis_pie_chart.png')  # Save the figure
plt.show()

# Bar Chart of Number of Invitations
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='number_of_invitations', data=df, palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Invitations per Contact')
plt.xlabel('Contact')
plt.ylabel('Number of Invitations')
plt.tight_layout()
plt.savefig(save_location + 'number_of_invitations.png')  # Save the figure
plt.show()

# Scatter Plot of Email Rate vs. Number of Interactions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='email_rate', y='number_of_interactions', data=df, hue='sentiment_analysis', palette='deep')
plt.title('Email Rate vs. Number of Interactions')
plt.xlabel('Email Rate')
plt.ylabel('Number of Interactions')
plt.legend(title='Sentiment Analysis')
plt.tight_layout()
plt.savefig(save_location + 'email_rate_vs_interactions.png')  # Save the figure
plt.show()
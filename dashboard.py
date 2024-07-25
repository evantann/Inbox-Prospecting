import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from JSON file
with open('email_data.json', 'r') as file:
    data = json.load(file)

# Convert data to a DataFrame
df = pd.DataFrame(data['contacts'])

# Setup seaborn style
sns.set(style="whitegrid")

# Plot number of interactions per contact
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='number_of_interactions', data=df, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Interactions per Contact')
plt.xlabel('Contact')
plt.ylabel('Number of Interactions')
plt.tight_layout()
plt.savefig('interactions_per_contact.png')
plt.show()

# Plot number of invitations per contact
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='number_of_invitations', data=df, palette='plasma')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Invitations per Contact')
plt.xlabel('Contact')
plt.ylabel('Number of Invitations')
plt.tight_layout()
plt.savefig('invitations_per_contact.png')
plt.show()

# Plot email rate per contact
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='email_rate', data=df, palette='cividis')
plt.xticks(rotation=45, ha='right')
plt.title('Email Rate per Contact')
plt.xlabel('Contact')
plt.ylabel('Email Rate')
plt.tight_layout()
plt.savefig('email_rate_per_contact.png')
plt.show()

# Plot duration known per contact
plt.figure(figsize=(12, 6))
sns.barplot(x='contact', y='duration_known', data=df, palette='inferno')
plt.xticks(rotation=45, ha='right')
plt.title('Duration Known per Contact')
plt.xlabel('Contact')
plt.ylabel('Duration Known (Days)')
plt.tight_layout()
plt.savefig('duration_known_per_contact.png')
plt.show()

# Plot contacts categorized by influencer status
plt.figure(figsize=(8, 6))
sns.countplot(x='is_influencer', data=df, palette='magma')
plt.title('Influencer Status Distribution')
plt.xlabel('Is Influencer')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Influencer', 'Influencer'])
plt.tight_layout()
plt.savefig('influencer_status_distribution.png')
plt.show()

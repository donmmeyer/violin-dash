#############################
# ViolinAI5.py
#########  Load libraries 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import csv
import random
import logging
import itertools


###################################################################################
# Function to read the results from a CSV file
def read_results_from_file0(filename):
    results = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
#            if row[0] == 'ProHobbyStudent':
#                continue  # Skip this row
            # Convert numeric fields to float or int
            row[2:] = [float(x) if '.' in x else int(x) for x in row[2:]]
            results.append(row)
    return results
############################################################
# Read results from the file
single_results = read_results_from_file0('singleresults.csv')

# Function to read the results from a CSV file
def read_results_from_file1(filename):
    results = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:

            # Convert numeric fields to float or int
            row[2:] = [float(x) if '.' in x else int(x) for x in row[2:]]
            results.append(row)
    return results
####################################################################
# Read results from the file
double_results = read_results_from_file1('doubleresults.csv')

# Function to read the results from a CSV file
def read_results_from_file2(filename):
    results = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:

            # Convert numeric fields to float or int
            row[2:] = [float(x) if '.' in x else int(x) for x in row[2:]]
            results.append(row)
    return results

# Read results from the file
triple_results = read_results_from_file2('tripleresults.csv')
##################################################################################################

def lookup_demographics_three(data, pr_sex, pr_educ, pr_often, pr_age, pr_type):
    # Filter out 'Not specified' and invalid records (respondents <= 3)
    demographics = [(pr_sex, "Sex"), (pr_educ, "Education"), (pr_often, "How often"), (pr_age, "Age"), (pr_type, "ProHobbyStudent")]
    valid_demographics = [dem for dem in demographics if dem[0] != 'Not specified']
    
    # Initialize result list
    results = []
    
    # Ensure there are at least 2 valid demographics
    if len(valid_demographics) >= 3:
        # Generate all combinations of 2 valid demographics
        demographic_combinations = list(itertools.combinations(valid_demographics, 3))
#        print('demographic_combinations=',demographic_combinations)
        # Iterate through each row of data
        for row in data:
            category, values, _, _, _, _, respondents = row

            # Check respondents and demographic combinations
            if respondents > 3:
                for combo in demographic_combinations:
                    # Check if both demographics in the combination are in values
                    if all(dem[0] in values for dem in combo):
                        results.append(row)
                        break  # Stop after finding the first matching combination
    
    return results
###########################################################################################
def lookup_demographics_two(data, pr_sex, pr_educ, pr_often, pr_age, pr_type):
    # Filter out 'Not specified' and invalid records (respondents <= 3)
    demographics = [(pr_sex, "Sex"), (pr_educ, "Education"), (pr_often, "How often"), (pr_age, "Age"), (pr_type, "ProHobbyStudent")]
    valid_demographics = [dem for dem in demographics if dem[0] != 'Not specified']
    
    # Initialize result list
    results = []
    
    # Ensure there are at least 2 valid demographics
    if len(valid_demographics) >= 2:
        # Generate all combinations of 2 valid demographics
        demographic_combinations = list(itertools.combinations(valid_demographics, 2))
#        print('demographic_combinations=',demographic_combinations)
        # Iterate through each row of data
        for row in data:
            category, values, _, _, _, _, respondents = row

            # Check respondents and demographic combinations
            if respondents > 3:
                for combo in demographic_combinations:
                    # Check if both demographics in the combination are in values
                    if all(dem[0] in values for dem in combo):
                        results.append(row)
                        break  # Stop after finding the first matching combination
    
    return results
#################################################################################################    

def lookup_demographics_one(data, pr_sex, pr_educ, pr_often, pr_age, pr_type):
    # Filter out 'Not specified' and invalid records (respondents <= 3)
    demographics = [(pr_sex, "Sex"), (pr_educ, "Education"), (pr_often, "How often"), (pr_age, "Age"), (pr_type, "ProHobbyStudent")]
    valid_demographics = [dem for dem in demographics if dem[0] != 'Not specified']
    
    # Initialize result list
    results = []
    
    # Ensure there are at least 2 valid demographics
    if len(valid_demographics) >= 1:
        # Generate all combinations of 2 valid demographics
        demographic_combinations = list(itertools.combinations(valid_demographics, 1))
#        print('demographic_combinations=',demographic_combinations)
        # Iterate through each row of data
        for row in data:
            category, values, _, _, _, _, respondents = row

            # Check respondents and demographic combinations
            if respondents > 3:
                for combo in demographic_combinations:
                    # Check if both demographics in the combination are in values
                    if all(dem[0] in values for dem in combo):
                        results.append(row)
                        break  # Stop after finding the first matching combination
    
    return results

################################################################################################
def recompute_probabilities(results):
    """
    Recompute the probabilities by summing the rows of probabilities and proportionally
    adjusting them to sum to 1.
    """
    total_brand1 = 0
    total_brand2 = 0
    total_brand3 = 0
    total_no_choice = 0

    # Sum the probabilities for each row
    for row in results:
        _, _, brand1, brand2, brand3, no_choice, _ = row
        total_brand1 += brand1
        total_brand2 += brand2
        total_brand3 += brand3
        total_no_choice += no_choice

    # Compute the total sum of all probabilities
    total_sum = total_brand1 + total_brand2 + total_brand3 + total_no_choice

    # Avoid division by zero
    if total_sum == 0:
        return None

    # Recompute probabilities as proportions of the total
    recomputed_brand1 = total_brand1 / total_sum
    recomputed_brand2 = total_brand2 / total_sum
    recomputed_brand3 = total_brand3 / total_sum
    recomputed_no_choice = total_no_choice / total_sum

    return {
        'brand1': recomputed_brand1,
        'brand2': recomputed_brand2,
        'brand3': recomputed_brand3,
        'no_choice': recomputed_no_choice
    }
##################################################################################


def lookup_demographics(pr_sex, pr_educ, pr_often, pr_age, pr_type):
    # If pr_type is 'Professional', return predefined probabilities and skip the rest

    
    # Filter out 'Not specified'
    demographics = [(pr_sex, "Sex"), (pr_educ, "Education"), (pr_often, "How often"), (pr_age, "Age"),(pr_type, "Type")]
    valid_demographics = [dem for dem in demographics if dem[0] != 'Not specified']
#    print('valid_demographics=',valid_demographics)
    # Check for 3 categories
    if len(valid_demographics) >= 3:
        results = lookup_demographics_three(triple_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)
        if results:
            return recompute_probabilities(results)
        else:
            results = lookup_demographics_two(double_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)
            if results:
                return recompute_probabilities(results)
            else:
                results = lookup_demographics_one(single_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)  
    # Check for 2 categories
    if len(valid_demographics) >= 2:
        results = lookup_demographics_two(double_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)
        if results:
            return recompute_probabilities(results)
        else:
            results = lookup_demographics_one(single_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)    
    # Check for 1 category
    if len(valid_demographics) >= 1:
        results = lookup_demographics_one(single_results, pr_sex, pr_educ, pr_often, pr_age, pr_type)
        if results:
            return recompute_probabilities(results)
    
    return None  # Return None if no valid records are found



###############################################################################

def recommend_violin_brand(probabilities, pr_sex, pr_educ, pr_often, pr_age, pr_type):
    # Check if any demographic data is 'Not specified'
    if pr_sex == 'Not specified' and pr_educ == 'Not specified' and pr_often == 'Not specified' and pr_age == 'Not specified' and pr_type == 'Not specified':
        return (
                "The Allegro Violin has the most appeal with a share of 17%. "
                "Focus on promoting its quality and value to this group. "
                "The Grand Violin is also popular, with a share of 17%. "
                "Consider highlighting its unique qualities as an alternative."
                "No demographic information given. ")

    # Extract probabilities for the three brands
    brand1_prob = probabilities['brand1']  # Grand
    brand2_prob = probabilities['brand2']  # Allegro
    brand3_prob = probabilities['brand3']  # Sonata

    # Sort the brands by their probabilities
    brand_probs = [('Grand', brand1_prob), ('Allegro', brand2_prob), ('Sonata', brand3_prob)]
    brand_probs.sort(key=lambda x: x[1], reverse=True)

    # Recommendation logic
    recommended_brand = brand_probs[0][0]
    highest_prob = brand_probs[0][1]
    second_brand = brand_probs[1][0]
    second_prob = brand_probs[1][1]

    # Check if the second brand is within 0.07 of the highest probability
    if abs(highest_prob - second_prob) <= 0.07:
        consider_brand = second_brand
    else:
        consider_brand = None

    # Format probabilities as rounded percentages
    brand1_percent = round(brand1_prob * 100)
    brand2_percent = round(brand2_prob * 100)
    brand3_percent = round(brand3_prob * 100)

    # If the player is a professional, only that is considered
    if pr_type == 'Professional':
        demographics_info = "The probabilities are based solely on the fact that the player is a Professional."
    else:
        # List of demographic variables
        demographics = [(pr_sex, "Sex"), (pr_educ, "Education"), (pr_often, "How often"), (pr_age, "Age"), (pr_type, "Player type")]
        valid_demographics = [dem[1] for dem in demographics if dem[0] != 'Not specified']

        demographics_info = f"Demographic Information: Age: {pr_age}, Education: {pr_educ}, Type: {pr_type}, Frequency: {pr_often}, Gender: {pr_sex}"
    
    # Generate recommendation sentence randomly
    recommendation_sentences = []
    
    if consider_brand:
        sentences = [
            f"I recommend the {recommended_brand} violin with a probability of {round(highest_prob * 100)}%, but you may also want to consider the {consider_brand}, which has a probability of {round(second_prob * 100)}%.",
            f"The {recommended_brand} violin is a great choice with {round(highest_prob * 100)}%, but the {consider_brand} is also quite close at {round(second_prob * 100)}%.",
            f"Based on the probabilities, the {recommended_brand} violin is best with {round(highest_prob * 100)}%, but do consider the {consider_brand} violin at {round(second_prob * 100)}%."
        ]
    else:
        sentences = [
            f"I recommend the {recommended_brand} violin with a probability of {round(highest_prob * 100)}%.",
            f"Your best option is the {recommended_brand} violin with {round(highest_prob * 100)}% probability.",
            f"Based on the probabilities, I suggest going with the {recommended_brand} violin, which has a {round(highest_prob * 100)}% chance."
        ]

    # Pick a random sentence from the list
    recommendation_sentence = random.choice(sentences)

    # Combine demographics info and recommendation into two separate lines
    return f"{demographics_info}\n{recommendation_sentence}"


################################################################################################    
#
from dash.dependencies import Input, Output, State
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define radio options for each group
age_options = [{'label': '39 and younger', 'value': '39 and younger'}, 
               {'label': '40 and older', 'value': '40 and older'}, 
               {'label': 'Not specified', 'value': 'Not specified'}]

education_options = [{'label': 'College', 'value': 'College'}, 
                     {'label': 'No college', 'value': 'No college'}, 
                     {'label': 'Not specified', 'value': 'Not specified'}]

profession_options = [{'label': 'Professional', 'value': 'Professional'}, 
                      {'label': 'Student', 'value': 'Student'}, 
                      {'label': 'Hobby', 'value': 'Hobby'}, 
                      {'label': 'Not specified', 'value': 'Not specified'}]

frequency_options = [{'label': 'Less than once a day', 'value': 'Less than once a day'}, 
                     {'label': 'Once a day or more', 'value': 'Once a day or more'}, 
                     {'label': 'Not specified', 'value': 'Not specified'}]

gender_options = [{'label': 'Male', 'value': 'Male'}, 
                  {'label': 'Female', 'value': 'Female'}, 
                  {'label': 'Not specified', 'value': 'Not specified'}]

# Layout with radio buttons for each category
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H1("Violin Brand Recommendation"), className="mb-4")
        ),
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="age-group",
                    options=age_options,
                    labelStyle={"display": "block"},
                    style={"margin-bottom": "10px"},
                    value='Not specified',  # Default value set to 'Not specified'
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="education-group",
                    options=education_options,
                    labelStyle={"display": "block"},
                    style={"margin-bottom": "10px"},
                    value='Not specified',  # Default value set to 'Not specified'
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="profession-group",
                    options=profession_options,
                    labelStyle={"display": "block"},
                    style={"margin-bottom": "10px"},
                    value='Not specified',  # Default value set to 'Not specified'
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="frequency-group",
                    options=frequency_options,
                    labelStyle={"display": "block"},
                    style={"margin-bottom": "10px"},
                    value='Not specified',  # Default value set to 'Not specified'
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.RadioItems(
                    id="gender-group",
                    options=gender_options,
                    labelStyle={"display": "block"},
                    style={"margin-bottom": "10px"},
                    value='Not specified',  # Default value set to 'Not specified'
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Submit", id="submit-button", color="primary", className="mt-3"),
                width="auto",
            )
        ),
        dbc.Row(
            dbc.Col(html.Div(id="output-recommendation", className="mt-3"))
        ),
    ],
    className="p-5",
)

# Callback to handle submit action
@app.callback(
    Output("output-recommendation", "children"),
    [Input("submit-button", "n_clicks")],
    [
        State("age-group", "value"),
        State("education-group", "value"),
        State("profession-group", "value"),
        State("frequency-group", "value"),
        State("gender-group", "value"),
    ]
)
def update_recommendation(submit_clicks, age, education, profession, frequency, gender):
    if not submit_clicks:
        return "Please select categories for the customer."

    # Print out the selected values when Submit is clicked
    print(f"Submitted Values:\nAge: {age}\nEducation: {education}\nProfession: {profession}\nFrequency: {frequency}\nGender: {gender}")

    # Get recommendation based on selected values (placeholder)
    probabilities = lookup_demographics(gender, education, frequency, age, profession)
    recommendation = recommend_violin_brand(probabilities, gender, education, frequency, age, profession)

    return f"Recommendation: {recommendation}"

if __name__ == "__main__":
    app.run_server(debug=True)

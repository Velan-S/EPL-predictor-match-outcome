from flask import Flask,render_template,request 
import pickle
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import r2_score

app = Flask(__name__, template_folder="template")

with open('reg_att.pkl', 'rb') as f:
    reg_att = pickle.load(f)

with open('reg_def.pkl', 'rb') as f:
    reg_def = pickle.load(f)

summary = pd.read_csv("summary.csv")
# Prepare the data for regression
X = summary[['avg_poss_x','avg_poss_y']].values
y_att = summary['Att'].values
y_def = summary['Def'].values

@app.route('/')
def home():
    return render_template('form1.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    model_results = pd.DataFrame({
        'Metric': ['Bias', 'Poss Coefficient', 'R^2 Score'],
        'Att': [np.mean([tree.tree_.value[0][0][0] for tree in reg_att.estimators_]), reg_att.feature_importances_[0], r2_score(y_att, reg_att.predict(X))],
        'Def': [np.mean([tree.tree_.value[0][0][0] for tree in reg_def.estimators_]), reg_def.feature_importances_[0], r2_score(y_def, reg_def.predict(X))]
    })
    poss_team1 = summary.loc[summary['team name'] == team1, 'Poss'].values[0]
    poss_team2 = summary.loc[summary['team name'] == team2, 'Poss'].values[0]
    
    # Use the decision tree models to predict the Att and Def variables for each team
    att_team1 = model_results['Att'][0]+(poss_team1 * model_results['Att'][1] )
    att_team2 =  model_results['Att'][0]+(poss_team2 * model_results['Att'][1] )
    def_team1 = model_results['Def'][0]+(poss_team1 * model_results['Def'][1] )
    def_team2 = model_results['Def'][0]+(poss_team2 * model_results['Def'][1] )
    
    # Calculate the log odds ratio for each team
    log_odds_team1 = att_team1 - def_team2
    log_odds_team2 = att_team2 - def_team1
    
    # Convert log odds ratio to expected goals
    xG_team1 = np.exp(log_odds_team1) / (1 + np.exp(log_odds_team1))
    xG_team2 = np.exp(log_odds_team2) / (1 + np.exp(log_odds_team2))
    
    # Create a dataframe to display the results
    table = pd.DataFrame({
        'Team': [team1, team2],
        'Att variable': [att_team1, att_team2],
        'Def variable': [def_team1, def_team2],
        'Expected Goals': [round(xG_team1, 2), round(xG_team2, 2)]
    })
    
    num_sims = 10000

    # Load the Att and Def variables and expected goals for each team
    data = table

    # Simulate num_sims games between the two teams
    results = np.zeros((num_sims, 2))
    for i in range(num_sims):
        # Calculate the number of goals for each team in this simulation
        goals_team1 = binom.rvs(n=num_sims, p=data.loc[0, 'Expected Goals']/10000)
        goals_team2 = binom.rvs(n=num_sims, p=data.loc[1, 'Expected Goals']/10000)

        # Store the results for this simulation
        results[i, 0] = goals_team1
        results[i, 1] = goals_team2

    # Calculate the probability of each outcome (win, draw, loss) for each team
    team1_wins = np.sum(results[:, 0] > results[:, 1]) / num_sims
    team1_draws = np.sum(results[:, 0] == results[:, 1]) / num_sims
    team1_losses = np.sum(results[:, 0] < results[:, 1]) / num_sims

    team2_wins = np.sum(results[:, 1] > results[:, 0]) / num_sims
    team2_draws = np.sum(results[:, 1] == results[:, 0]) / num_sims
    team2_losses = np.sum(results[:, 1] < results[:, 0]) / num_sims
    
    # Display the results
    print('Probabilities of each outcome for Liverpool:')
    print(f'Win: {team1_wins:.2%}')
    print(f'Draw: {team1_draws:.2%}')
    print(f'Loss: {team1_losses:.2%}\n')


    
    outcomes = pd.DataFrame({
    'Outcome': [f'{team1} Win', f'{team2} Win', 'Draw'],
    'Probability': [
        round(team1_wins * 100, 2),
        round(team2_wins * 100, 2),
        round(team1_draws * 100, 2),
    ]
    })
    
    return render_template('form1.html', table_data=outcomes.to_dict('records'), team1=team1, team2=team2)

if __name__ == "__main__":
    app.run(debug=True)
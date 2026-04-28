# Predictive Analysis with Explainability


In business, predictive analysis of existent data is helpful in that it helps understand causality, pinpoint the causes of failure, and prevent these through a restructuring of the processes involved. This project encompasses the analysis and categorisation of an event log, predictive analysis models that will be able to categorize using this event log, and model explainability measures via SHAP or Lime. Through this, not only will one be able to predict the outcome of certain trace prefixes, but even be able to assess which features that the greatest impact on the outcome being the way it is.

This is the primary repository for the Assignment on Explainability in the Practical 'Supervised Process Intelligence'.
Authors: 
Benedikt Koop
Sofia Vishnevskaia
Oleksandr Smuhliakov

Instructor: Dr. Alessandro Berti

## ℹ️ Overview

The project will use the 2013 BPI Challenge Incident Management Data log. The log can be found to download here: https://data.4tu.nl/articles/_/12693914/1


The Project consists of:	

- Formatting of event logs that classifies the traces by their outcome, in a binary manner
- Splitting of event logs into training, test and validation sets 
- Encoding cases into numerical feature vectors suitable for machine learning
- Applying these feature vectors to three predictive models, each of which are implemented during the project. These include a trivial baseline model, a 'white box' model, and a 'black box' model. 
- Using SHAP or LIME to compute feature importance for the black box model. 
- Implementing visualisations relevant to the prior steps
- Provide a prototype that allows demonstration
- Implement unit tests for each part of the project code

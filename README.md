app.py – orchestrates the application flow: it trains the model on startup, then waits for user input (button press) to simulate new orders, applies rules and model to each new order, and updates the dashboard.


 model.py – handles creation of mock data and training of the logistic regression model. This includes a function generate_mock_data to synthesize a dataframe of past orders (with features and a fraud label), and train_model to train the classifier. 
 
 
 rules.py – defines the rule-checking functions (is_address_mismatch, is_suspicious_email, is_suspicious_phone) and an apply_rules helper that aggregates all rule flags for a given order. Having this separate makes it easy to adjust or add rules.
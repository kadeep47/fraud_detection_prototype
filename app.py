

import streamlit as st
import pandas as pd
from model import generate_mock_data, train_model
from rules import apply_rules

# On app startup: generate historical data and train the ML model
st.title("Parania Inc. - Fraud Detection Dashboard")
st.write("## Real-time Order Monitoring for COD Fraud Prevention")

# Train the model on mock historical data (this would be replaced with real data in production)
with st.spinner("Training fraud detection model on historical data..."):
    training_data = generate_mock_data(500)  # generate 500 past orders with labels
    model = train_model(training_data)
st.success("Model trained. Ready to process new orders.")

# Initialize state for incoming orders if not already done
if 'new_orders' not in st.session_state:
    # Generate a list of new incoming orders (without fraud labels, as they would appear in real use)
    new_orders_df = generate_mock_data(20)  # create 20 incoming orders to simulate
    # Remove internal labels and rule flags to simulate raw incoming data
    new_orders_df = new_orders_df.drop(columns=['fraud_label', 'mismatch', 'suspicious_email',
                                                'suspicious_phone', 'repeated_ip'])
    st.session_state.new_orders = new_orders_df.to_dict(orient='records')
    st.session_state.processed_orders = []   # list to store orders that have been processed
    st.session_state.seen_ips = set()        # track seen IPs for repeat IP rule

# Button to ingest (process) the next order in the queue
if st.button("Ingest Next Order"):
    if st.session_state.new_orders:
        # Get the next order data (as a dict)
        next_order = st.session_state.new_orders.pop(0)
        # Apply rule-based checks to this order
        flags = apply_rules(next_order, st.session_state.seen_ips)
        # Prepare features for model prediction (ensure same order as training features)
        features = [[
            int(flags['mismatch']),
            int(flags['suspicious_email']),
            int(flags['suspicious_phone']),
            int(flags['repeated_ip']),
            next_order['amount'] / 1000.0   # scale amount to thousands as done in training
        ]]
        # Get fraud risk probability from the trained model
        risk_prob = model.predict_proba(features)[0][1]
        risk_percent = round(risk_prob * 100, 2)
        # Decide if the order is flagged as high risk:
        # We flag if model probability >= 50% or if any rule flag is True (conservative approach)
        threshold = 50.0  # risk percentage threshold for flagging
        flagged = (risk_percent >= threshold or any(flags.values()))
        # Record this order's info and results for display
        result_entry = {
            'Order ID': next_order['order_id'],
            'Email': next_order['email'],
            'Phone': next_order['phone'],
            'Billing Address': next_order['billing_address'],
            'Shipping Address': next_order['shipping_address'],
            'Amount': next_order['amount'],
            'Risk Score (%)': risk_percent,
            'Flagged': "Yes" if flagged else "No",
            'Alerts': ""
        }
        # If any rules triggered, list them in 'Alerts' (for transparency on why flagged)
        triggered_rules = [name.replace('_', ' ').title() 
                           for name, val in flags.items() if val]  # e.g., "mismatch" -> "Mismatch"
        if triggered_rules:
            result_entry['Alerts'] = ", ".join(triggered_rules)
        # Add the result to processed orders
        st.session_state.processed_orders.append(result_entry)
        # Update seen IPs set for future repeated IP checks
        st.session_state.seen_ips.add(next_order['ip_address'])
    else:
        st.warning("No more new orders to ingest.")

# Display the table of processed orders
if st.session_state.processed_orders:
    st.write("### Recent Orders")
    df_display = pd.DataFrame(st.session_state.processed_orders)
    st.dataframe(df_display)
else:
    st.info("Click **Ingest Next Order** to simulate an incoming order.")

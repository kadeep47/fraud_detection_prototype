import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

# Set random seed for reproducibility (so the same mock data is generated each run)
random.seed(42); np.random.seed(42)

def generate_mock_data(n: int = 500) -> pd.DataFrame:
    """
    Generate a mock dataset of n e-commerce orders with features and a fraud label.
    Each order includes fields like addresses, email, phone, IP, amount, and whether it was fraud.
    """
    # Helper to generate a random IP address (IPv4)
    def random_ip():
        return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
    # Generate a list of random IPs, with some deliberate repeats for testing the repeated_ip feature
    ip_list = [random_ip() for _ in range(n)]
    # Introduce repeated IPs by selecting a few IPs and assigning them to multiple orders
    repeat_ips = random.sample(ip_list, min(3, n))  # pick up to 3 IPs to repeat
    for ip in repeat_ips:
        # Assign this IP to 3 random orders (could overlap if same index chosen twice, but that's fine)
        for idx in random.sample(range(n), min(3, n)):
            ip_list[idx] = ip

    data = []
    # Define some sample domains (normal vs disposable) for email generation
    disposable_domains = ["tempmail.com", "fakeemail.com", "disposable.com"]
    normal_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "example.com"]
    for i in range(n):
        # Random billing and shipping addresses (simulated by number + street + city + pin code)
        pin_code = random.randint(100000, 999999)         # 6-digit postal code
        if random.random() < 0.8:
            # 80% cases: shipping same as billing
            ship_pin = pin_code
        else:
            # 20% cases: different shipping pin (address mismatch)
            ship_pin = random.randint(100000, 999999)
        mismatch = int(ship_pin != pin_code)
        # Construct simple address strings
        if mismatch == 0:
            billing_address = f"{i} Green Street, CityX, {pin_code}"
            shipping_address = billing_address
        else:
            billing_address = f"{i} Green Street, CityX, {pin_code}"
            shipping_address = f"{i} High Street, CityY, {ship_pin}"

        # Email: 10% chance of using a disposable domain
        if random.random() < 0.1:
            domain = random.choice(disposable_domains)
        else:
            domain = random.choice(normal_domains)
        email_user = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))  # random 5-letter string
        email = f"{email_user}@{domain}"
        suspicious_email = int(domain in disposable_domains)

        # Phone: generate a 10-digit number, and flag if it has a suspicious pattern
        phone = ''.join(str(random.randint(0, 9)) for _ in range(10))
        if random.random() < 0.1 or phone.startswith("000") or phone.startswith("999") \
           or phone.endswith("0000") or phone.endswith("1111"):
            suspicious_phone = 1
        else:
            suspicious_phone = 0

        # Order amount: random between 100 and 5000 (assuming currency like INR)
        amount = random.randint(100, 5000)

        data.append({
            "order_id": i+1,
            "billing_address": billing_address,
            "shipping_address": shipping_address,
            "billing_pin": pin_code,
            "shipping_pin": ship_pin,
            "email": email,
            "phone": phone,
            "ip_address": ip_list[i],
            # Rule-based features:
            "mismatch": mismatch,
            "suspicious_email": suspicious_email,
            "suspicious_phone": suspicious_phone,
            "amount": amount
        })
    df = pd.DataFrame(data)
    # Determine repeated_ip feature by checking duplicates in ip_address column
    ip_counts = df['ip_address'].value_counts()
    df['repeated_ip'] = df['ip_address'].apply(lambda ip: 1 if ip_counts[ip] > 1 else 0)

    # --- Label Generation for Training ---
    # We simulate a fraud label based on the features to train the model.
    # For simplicity, we'll use a weighted sum of the rule flags and amount to decide fraud.
    coeffs = {
        'mismatch': 1.0,
        'suspicious_email': 1.2,
        'suspicious_phone': 0.8,
        'repeated_ip': 1.0,
        'amount': 0.0005  # weight per currency unit (0.5 per 1000 amount)
    }
    intercept = -2.4  # bias term to adjust base fraud rate (around 20-30% fraud)
    # Compute a logistic score for each order
    linear_score = (df['mismatch'] * coeffs['mismatch'] +
                    df['suspicious_email'] * coeffs['suspicious_email'] +
                    df['suspicious_phone'] * coeffs['suspicious_phone'] +
                    df['repeated_ip'] * coeffs['repeated_ip'] +
                    df['amount'] * coeffs['amount'] +
                    intercept)
    fraud_prob = 1 / (1 + np.exp(-linear_score))
    df['fraud_label'] = (fraud_prob > 0.5).astype(int)  # label as 1 if probability > 50%
    # Now df contains a realistic-seeming distribution of fraud and legit orders
    return df

def train_model(training_data: pd.DataFrame):
    """
    Train a logistic regression model on the given training dataset.
    Returns the trained model ready for prediction.
    """
    features = ['mismatch', 'suspicious_email', 'suspicious_phone', 'repeated_ip', 'amount']
    X = training_data[features].copy()
    # Scale amount feature to match training assumptions (divide by 1000)
    X['amount'] = X['amount'] / 1000.0
    y = training_data['fraud_label']
    model = LogisticRegression(solver='lbfgs')
    model.fit(X, y)
    return model

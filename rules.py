# Define known disposable email domains for quick lookup
DISPOSABLE_DOMAINS = {"tempmail.com", "fakeemail.com", "disposable.com"}

def is_address_mismatch(billing_address: str, shipping_address: str) -> bool:
    """
    Check if billing and shipping addresses are significantly different.
    Here we consider them mismatched if the full address strings are not identical.
    """
    return billing_address != shipping_address

def is_suspicious_email(email: str) -> bool:
    """
    Check if the email's domain is in the list of disposable/suspicious domains.
    """
    try:
        domain = email.split("@")[1].lower()
    except IndexError:
        return False  # not a valid email format, but not necessarily fraud
    return domain in DISPOSABLE_DOMAINS

def is_suspicious_phone(phone: str) -> bool:
    """
    Check for suspicious phone number patterns:
    e.g., very repetitive or common fake patterns.
    """
    if len(phone) < 4:
        return True  # too short to be real
    # Rule: if all digits are same, or starts/ends with certain sequences
    if phone.count(phone[0]) == len(phone):
        return True  # all digits identical (e.g., "1111111111")
    if phone.startswith("000") or phone.startswith("999"):
        return True  # unrealistic prefixes
    if phone.endswith("0000") or phone.endswith("1111"):
        return True  # ends in repeated sequence
    return False

def apply_rules(order: dict, seen_ips: set) -> dict:
    """
    Apply all fraud rules to an order. Returns a dictionary of flags (True/False).
    Does not modify seen_ips; IP repetition check uses current seen_ips (should update externally).
    """
    flags = {}
    flags['mismatch'] = is_address_mismatch(order.get('billing_address', ""),
                                           order.get('shipping_address', ""))
    flags['suspicious_email'] = is_suspicious_email(order.get('email', ""))
    flags['suspicious_phone'] = is_suspicious_phone(order.get('phone', ""))
    # repeated_ip: True if this IP has been seen before (i.e., already in seen_ips set)
    flags['repeated_ip'] = order.get('ip_address', "") in seen_ips
    # (We handle updating seen_ips outside this function, after processing the order.)
    return flags

def verify_message_data(data):
    for i in range(1, len(data)):
        if not data[i].replace('.', '', 1).isdigit():
            return False
    return True


def hours_to_seconds(t):
    return t * 3600

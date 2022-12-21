import string

NAME_BIAS_WORDS = ['protagonist', 'Protagonist', 'PROTAGONIST', 'unnamed', 'Unnamed', 'UNNAMED', 'unknown', 'Unknown', 'UNKNOWN', 'None', 'none', 'None', 'Mr.', 'Mr ', 'Ms.', 'Ms ', 'Mrs.', 'Mrs ', 'Dr.', 'Dr ', 'TBA', 'TBD', 'N/A'] # technically no ' can filter out some reasonable names, but it's not a big deal and prevents some bad cases
BANNED_NAME_WORDS = NAME_BIAS_WORDS + ['\'', '_', '\n', '"', '#', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'redacted', 'mother', 'father', 'gram', 'grand', 'name', 'appearance', 'occupation', 'age', 'gender', 'sex', 'role', 'profession', 'job', 'friend', 'personality', 'trait', ' and ', 'The ', ' the ', 'national', 'country', 'day', 'date', 'description', 'identification', 'Mayor', 'Detective', 'Officer', 'Sheriff', 'Professor', 'Doctor'] + list(string.punctuation) # sometimes it'll find weird ascii chars to replace these if they're banned via logit bias

def simple_name_check(name):
    if len(name.strip()) == 0:
        return False
    if not all([piece.strip()[0].isupper() for piece in name.strip().split()]):
        return False
    if any([word.lower() in name.lower() for word in BANNED_NAME_WORDS]):
        return False
    if sum([1 for letter in name if letter.isupper()]) != len(name.strip().split()):
        return False
    return True
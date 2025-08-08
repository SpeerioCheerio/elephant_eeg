def count_transitions(series):
    transitions = 0
    prev_state = None
    for state in series:
        if state == 'BAD':
            continue
        if prev_state is not None and state != prev_state:
            transitions += 1
        prev_state = state
    return transitions
def assign_label(features):
    # 0 red_level
    # 1 white_level
    if features[0] < 7000 and features[1] > 0.0003:
        return 1  # Assegna la label 1 se il livello di rosso è inferiore a 10 e il livello di bianco è superiore a 10
    else:
        return 0  # Altrimenti assegna la label 0
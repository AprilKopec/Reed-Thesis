import random

def training_data(n):
    training_data = []
    
    for _ in range(n):
        # Randomly choose a data source (0 or 1)
        data_source = random.randint(0, 1)
        
        if data_source == 0:
            # Generate data from source 0 (mean=0, std=1)
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
        else:
            # Generate data from source 1 (mean=1, std=1)
            if random.randint(0, 1):
                x1 = random.gauss(1, 1)
                x2 = random.gauss(1, 1)
            else:
                x1 = random.gauss(-1, 1)
                x2 = random.gauss(-1, 1)
        
        # Create a tuple containing the data point and its true data source
        data_point = ((x1, x2), data_source)
        training_data.append(data_point)
    
    return training_data
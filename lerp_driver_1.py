def interpolate(data_points, x):
    data_points = sorted(data_points)

    for i in range(len(data_points) - 1):
        x1, y1 = data_points[i]
        x2, y2 = data_points[i + 1]
        
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))
    
    if x <= data_points[0][0]:
        return data_points[0][1]
    
    elif x >= data_points[-1][0]:
        return data_points[-1][1]
    
    return None

data = [(0, 0),
        (1000, 10), 
        (2000, 20), 
        (3000, 30)
        ]
print(interpolate(data, 3001))

from collections import deque

def calculate_rolling_average(new_value, window_size, rolling_values):
    if len(rolling_values) < window_size:
        rolling_values.append(new_value)
    else:
        rolling_values.popleft()
        rolling_values.append(new_value)
    
    return sum(rolling_values) / len(rolling_values)

# Example usage
window_size = 10
rolling_values = deque(maxlen=window_size)

# Simulating incoming angle measurements
angle_measurements = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

for angle in angle_measurements:
    rolling_avg = calculate_rolling_average(angle, window_size, rolling_values)
    print(f"New angle: {angle}, Rolling Average: {rolling_avg}")

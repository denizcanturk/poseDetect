from collections import deque
import serial
import time

def calculate_rolling_average(new_value, window_size, rolling_values):
    if len(rolling_values) < window_size:
        rolling_values.append(new_value)
    else:
        rolling_values.popleft()
        rolling_values.append(new_value)
    
    return sum(rolling_values) / len(rolling_values)

# Example usage
window_size = 5
rolling_values = deque(maxlen=window_size)

# Serial port configuration
serial_port = serial.Serial('COM1', 9600)  # Adjust the port and baudrate as needed

try:
    while True:
        # Simulate new angle measurement (replace with actual sensor reading)
        angle_measurement = 30  # Replace this line with your actual angle measurement
        
        # Calculate rolling average
        rolling_avg = calculate_rolling_average(angle_measurement, window_size, rolling_values)
        print(f"New angle: {angle_measurement}, Rolling Average: {rolling_avg}")
        
        # Send rolling average value through serial port
        serial_port.write(f"{rolling_avg}\n".encode())
        
        # Wait for a short time to simulate real-time behavior
        time.sleep(0.1)

except KeyboardInterrupt:
    # Close the serial port upon keyboard interrupt
    serial_port.close()
    print("Serial port closed.")
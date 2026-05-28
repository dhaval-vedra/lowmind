import lowmind as lm
import numpy as np

class SensorDataAnalyzer:
    def __init__(self):
        self.model = lm.Linear(6, 4)  # 6 sensors -> 4 conditions
        self.conditions = ["NORMAL", "WARNING", "CRITICAL", "EMERGENCY"]
        
    def softmax(self, x):
        """Manual softmax implementation"""
        if hasattr(x, 'data'):
            # Tensor object
            exp_x = np.exp(x.data - np.max(x.data))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            # numpy array
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
    def analyze_sensor_data(self, temperature, humidity, pressure, motion, light, sound):
        sensor_readings = [temperature, humidity, pressure, motion, light, sound]
        input_tensor = lm.Tensor([sensor_readings])
        
        # Get prediction
        output = self.model(input_tensor)
        probabilities = self.softmax(output)
        condition_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][condition_idx]
        
        return self.conditions[condition_idx], confidence, probabilities[0]

def iot_sensor_demo():
    print("🌐 IoT Sensor Data Analysis")
    print("=" * 45)
    
    analyzer = SensorDataAnalyzer()
    
    # Simulate different sensor scenarios
    scenarios = [
        # temp, hum, press, motion, light, sound
        [25.0, 45, 1013, 0.1, 300, 0.2],   # Normal office
        [35.0, 80, 1005, 0.8, 600, 0.9],   # Storm warning
        [45.0, 20, 980, 0.2, 1000, 0.1],   # Fire emergency
        [15.0, 90, 1030, 0.9, 50, 0.7],    # Security breach
    ]
    
    scenario_names = [
        "🏢 Normal Office Environment",
        "🌪️ Approaching Storm", 
        "🔥 Fire Hazard",
        "🚨 Security Alert"
    ]
    
    for name, sensors in zip(scenario_names, scenarios):
        condition, confidence, all_probs = analyzer.analyze_sensor_data(*sensors)
        
        print(f"\n{name}:")
        print(f"  📊 Sensors -> Temp: {sensors[0]}°C, Hum: {sensors[1]}%, "
              f"Press: {sensors[2]}hPa, Motion: {sensors[3]:.1f}")
        print(f"  🎯 Analysis: {condition} (confidence: {confidence:.2f})")
        print(f"  📈 Probabilities: {[f'{p:.2f}' for p in all_probs]}")
        
        # Recommendations based on condition
        if condition == "CRITICAL":
            print("  ⚠️  RECOMMENDATION: Evacuate immediately!")
        elif condition == "WARNING":
            print("  🔔 RECOMMENDATION: Check equipment and monitor")
        elif condition == "EMERGENCY":
            print("  🚨 RECOMMENDATION: Alert authorities!")
        else:
            print("  ✅ RECOMMENDATION: All systems normal")
    
    print("\n" + "=" * 45)
    print("✅ IoT Analysis Completed!")

# Run IoT demo
if __name__ == "__main__":
    iot_sensor_demo()
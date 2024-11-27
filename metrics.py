import matplotlib.pyplot as plt

# Data for the graph
metrics = ['Accuracy', 'Processing Speed (FPS)', 'Memory Usage (MB)']
values = [90, 30, 200]

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Performance Metrics of Face Detection and Blurring')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim(0, 250)  # Extend the y-axis limit
plt.text(0, 95, '90%', ha='center', va='bottom')
plt.text(1, 35, '30 FPS', ha='center', va='bottom')
plt.text(2, 205, '200 MB', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the graph
plt.tight_layout()
plt.show()
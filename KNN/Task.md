Use the diabetes.csv data to implement your own simple KNN classifier using python, (Don’t use any built-in functions), divide your data into 70% for training and 30% for testing. [diabetes.csv] 
1. Objective: 
• Perform multiple iterations of k (e.g., 5 iterations each different k value ex. K=2,3,4…) on the dataset. 
• Use Euclidean distance for computing distances between instances. 
2. Data preprocessing: 
• Normalize each feature column separately for training and test objects using Log Transformation or Min-Max Scaling. 
3. Break ties using Distance-Weighted Voting: 
• When there is a tie, consider the distances between the test instance and the tied classes' neighbors. 
• Assign higher weights to closer neighbors and use these weights to break the tie, reflecting the idea that closer neighbors might have a stronger influence on the classification decision. 
4. Output: 
• For each iteration, output the value of k and the following summary information: 
	o Number of correctly classified instances. 
	o Total number of instances in the test set. 
	o Accuracy. 
• At the end of all iterations, output the average accuracy across all iterations. 

Output Example: 
k value: 3 
Number of correctly classified instances: 418 
Total number of instances: 658 
Accuracy: 63%  
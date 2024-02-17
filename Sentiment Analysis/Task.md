(1) Description: In this project, your objective is to construct a sentiment analysis model capable of distinguishing between negative (0) and positive (1) sentiments. The project will supply a CSV file containing 2,745 sentences, their corresponding labels, ID and source. These sentiments are aggregated from three datasets: YELP, IMDB, and AMAZON. 
(2) Project Objectives: 
• Acquire proficiency in using the SpaCy and the Scikit-Learn library. 
• Grasp the steps involved in text preprocessing. 
• Utilize various techniques to obtain sentiments embedding vectors. 
• Become acquainted with the Linear Support Vector Classifier. 
• Attain a comprehension of the concept of hyperparameter tuning. 
(3) Requirements: 
(a) Data Exploration: 
• Begin by familiarizing yourself with the dataset.  
• Examine the distribution of samples in each class.  
(b) Data Preprocessing: 
• Drop the ID and source columns. 
• With SpaCy, eliminate stop words and perform lemmatization for each sentiment. 
• Generate sentence embeddings using a chosen embedding technique, such as CountVectorizer, Tf-idf, or any other preferred method. 
• Split into training and testing sets. 
(c) Classification and Comparison: 
• Initial Experiment: Implement sentiment classification using a Linear Support Vector Classifier (LinearSVC) with “Grid Search” to identify the optimal parameters for achieving the highest accuracy on the testing dataset. 
• Subsequent Experiment: Use Artificial Neural Network (ANN) for classification. Explore different hyperparameters such as the number of neurons, learning rate, and batch size to enhance the performance of the ANN model. 
• Provide a classification report for each experiment to comprehensively assess the performance of the models. 
• Save the best model, then reload it in a separate file, and use it on new unlabeled data to get predictions.
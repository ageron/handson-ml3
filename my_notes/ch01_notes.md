# Chapter 1: The Machine Learning Landscape

## What Is Machine Learning?

- Machine Learning is the science and art of programming computers so they can learn from data.
- Example: a spam filter learns from many examples of spam and non-spam emails, then uses that knowledge to classify new emails.

## Why Use Machine Learning?

- Training an algorithm with data can involve a shorter and easier-to-maintain program, and it can also be more accurate.
- Machine Learning can tackle super complex problems that are difficult to solve with traditional rules.
- Machine Learning makes it easier to automate retraining for constantly changing environments.
- Machine Learning may offer insights about the problems it solves.

## Types of Machine Learning Systems

### 1. Supervised and Unsupervised Learning

#### Supervised Learning

- In Machine Learning, supervised learning uses training data that contains inputs and the correct outputs, called labels.
- The model learns a mapping from features to a target or label.

Main tasks:

- Classification: predicts a category or class.
- Regression: predicts a numeric value.

Important notes:

- Target and label have nearly the same meaning.
- Target is more common in regression.
- Label is more common in classification.
- Features are input variables, and they are also called predictors or attributes.
- Some algorithms can do both classification and regression. For example, Logistic Regression is called "Regression", but it is mainly used for classification.

#### Unsupervised Learning

- In Machine Learning, unsupervised learning uses training data that is unlabeled.
- The algorithm learns patterns and structures without a teacher.

Main tasks:

1. Clustering

- Clustering groups similar data points together automatically.
- Example: blog visitors can be grouped by behavior or interests.
- Teenagers may read comics after school.
- Adults may read sci-fi on weekends.
- Hierarchical Clustering can further divide large groups into smaller subgroups.

2. Visualization

- Visualization converts complex high-dimensional data into 2D or 3D representations.
- It helps humans understand data structure.
- It helps identify hidden patterns.
- It helps people see clusters clearly.

3. Dimensionality Reduction

- Dimensionality Reduction reduces the number of features while keeping important information.
- It makes training faster.
- It makes storage smaller.
- It can sometimes make models better.

4. Anomaly Detection

- Anomaly Detection detects unusual or rare instances.
- Applications include fraud detection, manufacturing defect detection, and outlier removal.
- The model learns what is "normal" from training data.

5. Novelty Detection

- Novelty Detection detects completely new instances that are different from the training data.
- It requires a very clean dataset.

6. Association Rule Learning

- Association Rule Learning finds relationships between attributes or items in large datasets.
- Example: customers who buy barbecue sauce and potato chips also tend to buy steak.
- It is useful for recommendation systems.
- It is useful for product placement in supermarkets.

#### Self-Supervised Learning

- Self-Supervised Learning is a Machine Learning type where the system creates labels automatically from unlabeled data.
- After generating labels, the model is trained like supervised learning.

Typical workflow:

- Step 1: Pretraining.
- Step 2: Fine-tuning.

Transfer learning:

- Transfer learning means reusing knowledge from one task for another task.
- Self-Supervised Learning is usually treated as its own category.

#### Semi-Supervised Learning

- Semi-Supervised Learning uses a small amount of labeled data and a large amount of unlabeled data because labeling data is expensive and time-consuming.
- This learning type combines supervised learning and unsupervised learning.
- Example: photo recognition, such as Google Photos.

#### Reinforcement Learning

- Reinforcement Learning is a learning method where an agent interacts with an environment.

Core process:

- The agent observes the environment.
- The agent chooses an action.
- The environment returns a reward or penalty, also called a negative reward.
- The agent learns from feedback.
- Goal: maximize total reward over time.

Important concepts:

- Agent: the learner or decision maker.
- Environment: the world the agent interacts with.
- Reward: the feedback signal that measures how good an action is.
- Policy: the strategy that tells the agent what action to choose in a given situation.

Example:

- Robots learning to walk.

Notes:

- During official matches, learning was turned off, and AlphaGo only used the policy it had already learned.
- This is an example of offline learning.

### 2. Batch Versus Online Learning

#### Batch Learning, or Offline Learning

- The model is trained using the entire dataset at once.
- The system cannot learn incrementally.

Workflow:

- Collect all training data.
- Train the model offline.
- Deploy the model into production.
- The model stops learning and only makes predictions.

Problem: model performance decays.

- Over time, the real world changes while the model stays unchanged.
- This causes data drift and model rot.
- Models must be retrained regularly with updated data.
- Retraining can be expensive because it requires CPU, memory, disk space, and network resources.
- Batch learning can adapt slowly.
- Batch learning can be poor for limited devices.

#### Online Learning, or Incremental Learning

- The model learns incrementally from incoming data.
- Data is fed one instance at a time or in small groups called mini-batches.
- Each update is fast and cheap.
- The system can adapt continuously.

Main advantages:

- It adapts quickly.
- It works with limited resources.
- It handles huge datasets.
- It supports Out-of-Core Learning.
- Out-of-Core Learning means the data is too large for RAM, so it cannot fit in memory. The algorithm loads small chunks of data and trains step by step.

Learning rate:

- The learning rate controls how fast the model adapts.
- A high learning rate adapts quickly, forgets old data faster, and is more sensitive to noise or outliers.
- A low learning rate learns slowly, remembers old data better, and is more stable.

Risk of online learning:

- Bad incoming data can quickly damage performance, such as sensor bugs or corrupted data.
- The system needs strong monitoring or anomaly detection.

### 3. Instance-Based and Model-Based Learning

- The main goal is to perform well on unseen or new data.
- This is called generalization.
- There are two major approaches: instance-based learning and model-based learning.

#### Instance-Based Learning

- Instance-Based Learning learns by memorizing training examples and comparing new data with known examples using a similarity measure, such as the L2 norm.

Advantages:

- It has a simple idea.
- It does not need a complex training process.
- It can adapt by storing new examples.

Disadvantages:

- It requires storing many training instances.
- Prediction can become slow on large datasets.
- It is sensitive to the choice of similarity metric.

#### Model-Based Learning

- Model-Based Learning learns by building a model from training data.
- It then uses the model to make predictions on new data.
- Goal: generalize well to unseen examples.

Model selection:

- Choose a mathematical relationship between variables, such as `y = ax + b`.
- This is called model selection.
- Model parameters are values learned by the model.
- A linear model has two parameters: bias and slope.
- Changing the parameters changes the line.

Cost function:

- The cost function measures whether the model is good.
- Purpose: measure the prediction error.
- It tries to minimize the distance between the prediction and the actual training data.

Training:

- Linear Regression finds the best parameter values.
- This process is called training the model.

Prediction, or inference:

- Prediction means using a trained model to make predictions on new data.

Typical Machine Learning workflow:

- Study and visualize data.
- Select data.
- Train the model.
- Make predictions on new data.

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# Input feature (X) and target label (y)
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize data
lifesat.plot(
    kind="scatter",
    grid=True,
    x="GDP per capita (USD)",
    y="Life satisfaction"
)
plt.axis([23500, 62500, 4, 9])
plt.show()

# Create linear regression model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict Cyprus life satisfaction
X_new = [[37655.2]]
prediction = model.predict(X_new)
print(prediction)
```

Code notes:

- `import matplotlib.pyplot as plt`: imports Matplotlib's plotting tools and gives them the short name `plt`.
- `import pandas as pd`: imports Pandas for loading and working with tabular data.
- `import numpy as np`: imports NumPy for numerical arrays and mathematical operations.
- `from sklearn.linear_model import LinearRegression`: imports the Linear Regression model from scikit-learn.
- `data_root = "https://github.com/ageron/data/raw/main/"`: stores the base URL where the dataset is located.
- `pd.read_csv(...)`: loads the CSV dataset into a Pandas DataFrame.
- `lifesat[["GDP per capita (USD)"]].values`: selects the input feature and converts it to a NumPy array.
- `lifesat[["Life satisfaction"]].values`: selects the target value and converts it to a NumPy array.
- `lifesat.plot(...)`: creates a scatter plot to visualize the relationship between GDP per capita and life satisfaction.
- `plt.axis([23500, 62500, 4, 9])`: sets the visible range of the x-axis and y-axis.
- `plt.show()`: displays the plot.
- `model = LinearRegression()`: creates a Linear Regression model.
- `model.fit(X, y)`: trains the model using the input feature `X` and the target `y`.
- `X_new = [[37655.2]]`: creates a new input value for Cyprus GDP per capita.
- `model.predict(X_new)`: predicts the life satisfaction value for the new input.
- `print(prediction)`: prints the model's prediction.

#### Instance-Based Alternative

- An instance-based alternative is to use K-Nearest Neighbors Regression.

```python
# Replace this model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# With this model
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
```

Code notes:

- `from sklearn.linear_model import LinearRegression`: imports a model-based Linear Regression algorithm.
- `model = LinearRegression()`: creates a Linear Regression model.
- `from sklearn.neighbors import KNeighborsRegressor`: imports the K-Nearest Neighbors Regression algorithm.
- `model = KNeighborsRegressor(n_neighbors=3)`: creates a KNN regression model that predicts using the 3 nearest training examples.

## Main Challenges of Machine Learning

- Two major things can go wrong in Machine Learning:
  - Bad data.
  - Bad model.

### Insufficient Quantity of Training Data

- Most Machine Learning algorithms need a large amount of training data to work well.

Human learning versus Machine Learning:

- Humans: a child may learn what an apple is after seeing only a few examples.
- Machine Learning systems: usually require much more data.

The Unreasonable Effectiveness of Data:

- A famous 2001 research paper by Michele Banko and Eric Brill showed that many different Machine Learning algorithms performed similarly well once they were given enough data.
- Main idea: for complex problems, more data can matter more than having a sophisticated algorithm.
- Important insight: for many real-world problems, more high-quality data is extremely important, and the algorithm still matters too.

Practical reality:

- Large datasets are not always available.
- Collecting and labeling data can be expensive, slow, and difficult.
- Thus, the algorithm still matters, especially for small and medium-sized datasets.

### Nonrepresentative Training Data

- For a model to generalize well, the training data must represent the real-world cases the model will face.
- This is important for both instance-based learning and model-based learning.

Problem:

- If training data is not representative, the model learns misleading patterns.
- Predictions become inaccurate.

Sampling noise:

- If the dataset is too small, random chance may create unrepresentative data.
- This problem is called sampling noise.

Sample bias:

- Even large datasets can still be biased if data collection is flawed.
- This means some groups are overrepresented, while others are underrepresented.
- This problem is called sample bias.

### Poor-Quality Data

- Poor-quality training data makes it difficult for Machine Learning models to detect real patterns.
- Problems in data may include errors, noise, incorrect measurements, outliers, and missing values.
- Result: lower model performance and inaccurate predictions.
- Thus, data cleaning is very important and is one of the most important tasks in Machine Learning.

Common data problems:

1. Outliers

- Some data points are extremely abnormal or incorrect.
- Possible solutions: remove the outlier or manually correct the error.

2. Missing Features or Missing Values

- Ignore the feature: remove the whole column or attribute.
- Ignore the instances: remove rows with the missing value.
- Fill the missing value: replace missing values with the mean, median, mode, or predicted values.
- Train multiple models: one model using the feature and another model without the feature, then compare performance.

### Irrelevant Features

- In Machine Learning, model quality depends heavily on the quality of features.

Principle:

- "Garbage in, garbage out."
- If input features are poor or irrelevant, predictions will also be poor.

Problem:

- A model learns well only if the training data contains enough relevant features.
- A model also learns better when unnecessary or irrelevant features are minimized.

Too many irrelevant features can:

- Confuse the model.
- Reduce accuracy.
- Increase training time.
- Cause overfitting.

Feature engineering:

- Feature engineering is the process of creating and improving features.
- It is one of the most important parts of Machine Learning projects.

Main steps in Feature Engineering:

- Feature selection: choose the most useful existing features.
- Goal of feature selection: keep important features and remove useless ones.
- Feature extraction: create better features by combining existing ones.
- Dimensionality reduction techniques can help with feature extraction.
- Creating new features: collect new data to improve the model.
- New relevant features often improve performance significantly.

### Overfitting

- Overfitting happens when a model performs very well on training data but fails to generalize to new or unseen data.
- The model learns noise, random patterns, and accidental relationships instead of real underlying patterns.

Why overfitting happens:

- It usually occurs when the model is too complex.
- It usually occurs when the dataset is too small.
- It usually occurs when the training data is noisy.
- Complex models, such as deep neural networks and high-degree polynomial regression, can easily memorize noise.

Signs of overfitting:

- Training performance has very high accuracy or very low error.
- Test or new data performance has poor accuracy and poor generalization.

Solutions to overfitting:

- Simplify the model by using fewer parameters or simpler algorithms.
- Reduce the number of features by removing irrelevant or noisy attributes.
- Reducing features reduces unnecessary complexity.
- Gather more training data so the model can learn true patterns and reduce memorization of noise.
- Clean the data by fixing errors, removing outliers, and reducing noise.
- Use regularization to reduce overfitting by constraining the model and making it simpler.
- The goal is to balance fitting the training data and keeping the model simple.
- Reducing the degrees of freedom makes the model simpler.

Hyperparameters:

- Hyperparameters control the learning process, not the model parameters themselves.
- They are set before training and stay fixed during training.

Effect of regularization strength:

- Very small regularization makes the model very flexible, which increases overfitting risk.
- Very large regularization makes the model too simple, which can increase underfitting risk.
- Goal: find the right balance.

### Underfitting

- Underfitting happens when the model is too simple and cannot learn the real structure or patterns in the data.

Signs of underfitting:

- Training data has high error and poor accuracy.
- Test data also has poor performance.
- The model fails everywhere because it has not learned enough.

Solutions:

- Use a more powerful model.
- Improve the features with feature engineering.
- Reduce regularization.

## Testing and Validating

- When training a model, split the data into a training set and a test set.
- The training set is used to train the model.
- The test set is used to evaluate the model on unseen data.
- This helps estimate real-world performance.
- Generalization error, also called out-of-sample error, is the error on unseen or new data.
- A low generalization error means the model generalizes well.

Detecting model behavior:

- Ideal: training error is low and test error is low.
- Overfitting: training error is extremely low and test error is extremely high.
- Underfitting: training error is high and test error is high.

Meaning:

- In overfitting, the model memorized the training data.
- In overfitting, the model has poor generalization.
- In underfitting, the model is too simple to learn the patterns.

Typical data split:

- Common practice: 80% training data and 20% test data.
- Large datasets may require much smaller test percentages.
- Example: if the dataset has 1,000,000 instances, a 1% test set gives 10,000 test instances, which can be enough.

Basic workflow:

1. Split Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Code notes:

- `from sklearn.model_selection import train_test_split`: imports the helper function used to split data into training and test sets.
- `train_test_split(...)`: randomly splits features and labels into training and test parts.
- `X`: contains the input features.
- `y`: contains the target values or labels.
- `test_size=0.2`: puts 20% of the data into the test set.
- `random_state=42`: makes the random split reproducible.
- `X_train`: contains training features.
- `X_test`: contains test features.
- `y_train`: contains training targets.
- `y_test`: contains test targets.

2. Train Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

Code notes:

- `from sklearn.linear_model import LinearRegression`: imports the Linear Regression model.
- `model = LinearRegression()`: creates a Linear Regression model object.
- `model.fit(X_train, y_train)`: trains the model using the training features and training targets.

3. Evaluate Model

```python
from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Test MSE:", mse)
```

Code notes:

- `from sklearn.metrics import mean_squared_error`: imports the function used to measure mean squared error.
- `model.predict(X_test)`: uses the trained model to predict values for the test features.
- `predictions`: stores the predicted values.
- `mean_squared_error(y_test, predictions)`: compares the true test targets with the predicted values.
- `mse`: stores the test mean squared error.
- `print("Test MSE:", mse)`: prints the test error.

### Hyperparameter Tuning and Model Selection

- In Machine Learning, after training a model, we must evaluate it, compare models, and tune hyperparameters.
- Goal: find the model that generalizes best to unseen data.

Model selection:

- Suppose we compare Linear Regression and Polynomial Regression.
- Step 1: train both models.
- Step 2: evaluate them on unseen data.
- Step 3: choose the model with lower generalization error.
- This is called model selection.

Hyperparameters:

- A hyperparameter controls the learning process, not the learned parameters themselves.
- Hyperparameters are chosen before training.
- Hyperparameters are not learned automatically from data.

Problem with using the test set repeatedly:

- Suppose we train 100 models, each with different hyperparameter values, and select the one with the best test performance.
- Problem: the model becomes indirectly optimized for the test set.
- Test set information leaks into training decisions.
- Result: overly optimistic evaluation and poorer real-world performance.

#### Holdout Validation

- Solution: split data into a training set, validation set, and test set.
- Training set: train candidate models.
- Validation set, also called the dev set: compare models and hyperparameters.
- Test set: final unbiased evaluation.

Holdout validation workflow:

- Step 1: split the original training data into a reduced training set and a validation set.

```text
Training Data
+-- Reduced Training Set
+-- Validation Set
```

- Step 2: train many candidate models using different hyperparameters.
- Step 3: evaluate each model on the validation set.
- Step 4: select the best-performing model.
- Step 5: retrain the best model on the full training data, meaning training plus validation.
- Step 6: evaluate the final model once on the test set.
- This gives a more reliable estimate of generalization error.

Example workflow in Python:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Split into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Split training into reduced training set and validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42
)

best_model = None
best_error = float("inf")

# Try different hyperparameter values
for alpha in [0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    error = mean_squared_error(y_valid, predictions)
    if error < best_error:
        best_error = error
        best_model = model

print("Best validation error:", best_error)
```

Code notes:

- `from sklearn.model_selection import train_test_split`: imports the function used to split datasets.
- `from sklearn.linear_model import Ridge`: imports Ridge Regression, a regularized linear model.
- `from sklearn.metrics import mean_squared_error`: imports the function used to compute prediction error.
- `train_test_split(X, y, test_size=0.2, random_state=42)`: splits the full dataset into a training part and a final test part.
- `X_train_full`: contains the full training features before creating the validation set.
- `X_test`: contains the final test features.
- `y_train_full`: contains the full training targets before creating the validation set.
- `y_test`: contains the final test targets.
- `train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)`: splits the full training set into a reduced training set and a validation set.
- `X_train`: contains the reduced training features.
- `X_valid`: contains the validation features.
- `y_train`: contains the reduced training targets.
- `y_valid`: contains the validation targets.
- `best_model = None`: creates a variable to store the best model found so far.
- `best_error = float("inf")`: starts the best error at infinity so any real error will be smaller.
- `for alpha in [0.01, 0.1, 1, 10, 100]`: tries several values of the `alpha` hyperparameter.
- `model = Ridge(alpha=alpha)`: creates a Ridge Regression model using the current `alpha`.
- `model.fit(X_train, y_train)`: trains the model on the reduced training set.
- `model.predict(X_valid)`: predicts target values for the validation set.
- `mean_squared_error(y_valid, predictions)`: measures validation error.
- `if error < best_error`: checks whether the current model is better than the previous best model.
- `best_error = error`: saves the new best validation error.
- `best_model = model`: saves the current model as the best model.
- `print("Best validation error:", best_error)`: prints the best validation error found.

### Cross-Validation

- Cross-Validation improves validation reliability.

Idea:

- Use many small validation sets.
- Train and evaluate multiple times.
- Average the results.
- This reduces random evaluation error.

Advantages of Cross-Validation:

- It provides a more accurate performance estimate.
- It helps with better model selection.
- It is especially useful for small datasets.

Drawback:

- It is more computationally expensive because the model must be trained multiple times.

### Data Mismatch

- Data mismatch occurs when the training data distribution is different from the production or real-world data distribution.

Result:

- The model performs well during training.
- The model performs poorly in real-world usage.

Important rule:

- The validation set and test set should be as representative as possible of production data.
- Use real mobile-app images for the validation set and test set.
- Do not carelessly mix web images and production images.

Problem diagnosis:

- Suppose the model trains well on web images but performs poorly on validation data.
- Question: is the problem overfitting or data mismatch?
- It is hard to know directly.

#### Train-Dev Set

- Andrew Ng proposed using an additional dataset called the train-dev set.

Dataset structure:

```text
Web Images
+-- Training Set
+-- Train-Dev Set

Real Mobile Images
+-- Validation (Dev) Set
+-- Test Set
```

Workflow:

- Step 1: train the model on the training set, which contains web images.
- Step 2: evaluate the model on the train-dev set.
- If performance on the train-dev set is poor, the problem is overfitting.
- Possible solutions: regularization, more data, a simpler model, or cleaner data.
- Step 3: if train-dev performance is good, evaluate the model on the dev set, which contains real images.
- If dev set performance is poor, the problem is data mismatch.
- Possible solution: preprocess training images to resemble mobile photos, then retrain the model.
- Step 4: do final evaluation on the test set.
- This estimates real production performance.

### No Free Lunch Theorem

- The No Free Lunch theorem states that no single Machine Learning model is best for every problem.

Meaning:

- Every model makes assumptions about data.
- Example: Linear Regression assumes relationships are approximately linear.
- Different datasets may favor linear models, decision trees, neural networks, SVMs, and other algorithms.

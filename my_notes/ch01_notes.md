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

Important note:

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

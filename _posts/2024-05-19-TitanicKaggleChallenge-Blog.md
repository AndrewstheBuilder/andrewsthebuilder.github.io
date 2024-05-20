---
layout: post
title: "Journey into my first Kaggle Challenge: The Titanic Challenge"
date: 2024-05-19
categories: datascience
---

This is my first kaggle challenge. I needed help to complete it because I have only been studying deep learning and not data analysis. I did not know where to start or what to look for in doing data analysis. So I followed a previously done solution to the Titanic Challenge closely and tried to gain an intuition for everything that was done. This Jupyter notebook is that effort + the machine learning part in the end after the data analysis part!

<a href="https://colab.research.google.com/github/AndrewstheBuilder/DeepLearning/blob/main/TitanicKaggleChallenge_Blog.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

I did a Kaggle Challenge and this is my work related to that. Everything above is my imports and connecting to my Google Drive for use in Google Colaboratory. Here is the Kaggle Challenge Link: https://www.kaggle.com/competitions/titanic/overview

## Step 1: **Explore the dataset and do data analysis**.
Used https://www.kaggle.com/code/startupsci/titanic-data-science-solutions for data analysis inspiration


```python
# Read in data and see a preview of it using .head()
data = pd.read_csv('./drive/MyDrive/Titanic_Challenge/train.csv')
data.head()
```





  <div id="df-ede83f6b-efb9-420d-b665-43acc340ab18" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ede83f6b-efb9-420d-b665-43acc340ab18')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ede83f6b-efb9-420d-b665-43acc340ab18 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ede83f6b-efb9-420d-b665-43acc340ab18');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-cad7e0c9-7fa0-4c31-b7e9-7ee6c0c3c325">
  <button class="colab-df-quickchart" onclick="quickchart('df-cad7e0c9-7fa0-4c31-b7e9-7ee6c0c3c325')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-cad7e0c9-7fa0-4c31-b7e9-7ee6c0c3c325 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Convert categorical datatypes 'Sex' and 'Embarked' into numeric
  # to do correlation calculation
numeric_df = data.select_dtypes(include=[float, int])
numeric_df['Sex'] = data['Sex'].map({'male': 0, 'female':1})
numeric_df['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})

# numeric_df.head()
numeric_df.corr()
```





  <div id="df-24ead6d5-dd9d-40b4-a973-dab55396980f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Sex</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>-0.005007</td>
      <td>-0.035144</td>
      <td>0.036847</td>
      <td>-0.057527</td>
      <td>-0.001652</td>
      <td>0.012658</td>
      <td>-0.042939</td>
      <td>-0.030555</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.005007</td>
      <td>1.000000</td>
      <td>-0.338481</td>
      <td>-0.077221</td>
      <td>-0.035322</td>
      <td>0.081629</td>
      <td>0.257307</td>
      <td>0.543351</td>
      <td>0.108669</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.035144</td>
      <td>-0.338481</td>
      <td>1.000000</td>
      <td>-0.369226</td>
      <td>0.083081</td>
      <td>0.018443</td>
      <td>-0.549500</td>
      <td>-0.131900</td>
      <td>0.043835</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.036847</td>
      <td>-0.077221</td>
      <td>-0.369226</td>
      <td>1.000000</td>
      <td>-0.308247</td>
      <td>-0.189119</td>
      <td>0.096067</td>
      <td>-0.093254</td>
      <td>0.012186</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.057527</td>
      <td>-0.035322</td>
      <td>0.083081</td>
      <td>-0.308247</td>
      <td>1.000000</td>
      <td>0.414838</td>
      <td>0.159651</td>
      <td>0.114631</td>
      <td>-0.060606</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.001652</td>
      <td>0.081629</td>
      <td>0.018443</td>
      <td>-0.189119</td>
      <td>0.414838</td>
      <td>1.000000</td>
      <td>0.216225</td>
      <td>0.245489</td>
      <td>-0.079320</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.012658</td>
      <td>0.257307</td>
      <td>-0.549500</td>
      <td>0.096067</td>
      <td>0.159651</td>
      <td>0.216225</td>
      <td>1.000000</td>
      <td>0.182333</td>
      <td>0.063462</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.042939</td>
      <td>0.543351</td>
      <td>-0.131900</td>
      <td>-0.093254</td>
      <td>0.114631</td>
      <td>0.245489</td>
      <td>0.182333</td>
      <td>1.000000</td>
      <td>0.118593</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>-0.030555</td>
      <td>0.108669</td>
      <td>0.043835</td>
      <td>0.012186</td>
      <td>-0.060606</td>
      <td>-0.079320</td>
      <td>0.063462</td>
      <td>0.118593</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-24ead6d5-dd9d-40b4-a973-dab55396980f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-24ead6d5-dd9d-40b4-a973-dab55396980f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-24ead6d5-dd9d-40b4-a973-dab55396980f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b1c120f6-4eff-40f9-a0ac-c210e73fcbc7">
  <button class="colab-df-quickchart" onclick="quickchart('df-b1c120f6-4eff-40f9-a0ac-c210e73fcbc7')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b1c120f6-4eff-40f9-a0ac-c210e73fcbc7 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Gain an intuition** for how the correlation calculation works by using a tiny example. **What I found**: the best correlations have graphs that depict a monotonic relationship like the sigmoid function and not snake curves like the piece wise function (graphed below)


```python
# Prove to myself that outputs going up and down will give a bad correlation
# Correlation only captures monotonic relationships
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def piecewise(x):
    if x < -5:
        return 0
    elif -5 <= x < 0:
        return 1
    elif 0 <= x < 5:
        return 0
    else:
        return 1

# Generate x values
x = np.linspace(-10, 10, 400)  # 400 points from -10 to 10

# Apply the piecewise function to each x value
y = np.array([piecewise(xi) for xi in x])

# Plotting the Sigmoid function
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Piecewise Function')
plt.title('Piecewise Function')
plt.xlabel('X values')
plt.ylabel('Piecewise Output')
plt.grid(True)
plt.legend()
plt.show()

# Create a DataFrame
data_piecewise = pd.DataFrame({'X': x, 'Y': y})

# Calculate correlation
correlation = data_piecewise['X'].corr(data_piecewise['Y'])
print("Correlation coefficient:", correlation)

# Monotonic function - Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Apply the sigmoid function to each x value
y_sigmoid = np.array([sigmoid(xi) for xi in x])

# Plotting the Sigmoid function on the same plot
plt.plot(x, y_sigmoid, label='Sigmoid Function', linestyle='--')
plt.legend()
plt.show()

# Create a DataFrame for the sigmoid function
data_sigmoid = pd.DataFrame({'X': x, 'Y_Sigmoid': y_sigmoid})

# Calculate correlation for sigmoid
correlation_sigmoid = data_sigmoid['X'].corr(data_sigmoid['Y_Sigmoid'])
print("Sigmoid Correlation coefficient:", correlation_sigmoid)
```



![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_8_0.png)



    Correlation coefficient: 0.43301405506325596




![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_8_2.png)



    Sigmoid Correlation coefficient: 0.9362677445388511


Below plot shows why pandas.corr() may not be the most reliable metric for showing which features(Age, Embarked,etc) are the most important to deciding whether a passenger lived or died. **The data is up and down**: As age goes up there is no clear rise or fall in deaths its behaving more like the piece wise function and not like the sigmoid function above.


```python
# Completely useless plot
# But it shows why the correlation calculation would not work.
# The data is up and down like the piece wise function above
plt.figure(figsize=(50,6))
plt.scatter(data['Age'], data['Survived'], alpha=1, color='red')
plt.title("Scatter Plot of Survived vs. Age")
plt.xlabel('Age')
plt.ylabel('Survived')
plt.xticks(range(1, 101, 1))
plt.grid(True)
plt.show()
```



![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_10_0.png)




```python
# Determine Age bands and check correlation with Survived
# print(pd.cut(data['Age'],5))
# print(data['Age'].head())
data['AgeBand'] = pd.cut(data['Age'], 5)
data[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```





  <div id="df-3e5a5bce-0e1d-49f4-adff-bd7c97589320" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AgeBand</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0.34, 16.336]</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(16.336, 32.252]</td>
      <td>0.369942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(32.252, 48.168]</td>
      <td>0.404255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(48.168, 64.084]</td>
      <td>0.434783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(64.084, 80.0]</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3e5a5bce-0e1d-49f4-adff-bd7c97589320')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3e5a5bce-0e1d-49f4-adff-bd7c97589320 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3e5a5bce-0e1d-49f4-adff-bd7c97589320');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-745dad3c-c165-4a25-b3d0-24d62f56eca2">
  <button class="colab-df-quickchart" onclick="quickchart('df-745dad3c-c165-4a25-b3d0-24d62f56eca2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-745dad3c-c165-4a25-b3d0-24d62f56eca2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




These are the graphs that I mainly relied on for my analysis because it shows the correlation between deaths and the other classes well. "Oh look the lower class people tended to not be survivors"


```python
# These graphs are gold!
# Load and analyze the data
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Columns to plot, excluding 'PassengerId', 'Name', 'Ticket', 'Cabin' due to their specificity or lack of relevance
columns_to_plot = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Define a function to plot survival count based on a given feature
def plot_survival(data, column):
    if column=='Age' or column=='Fare':
        g = sns.FacetGrid(data, col='Survived', height=5, aspect=1)
        g.map(sns.histplot, column, bins=20, kde=False)
    else:
        # Create a count plot for categorical data
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, hue='Survived', data=data, palette='viridis')
        plt.title(f'Survival Count by {column}')
        plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
        plt.show()

# Create a loop to generate plots for each relevant column
for column in columns_to_plot:
    plot_survival(data, column)
```



![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_0.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_1.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_2.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_3.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_4.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_5.png)





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_13_6.png)




```python
# Read in training and test dataset
# Previous analysis above was also on training dataset it just had a different variable name
# Now calling training dataset train_df
train_df = pd.read_csv('./drive/MyDrive/Titanic_Challenge/train.csv')
test_df = pd.read_csv('./drive/MyDrive/Titanic_Challenge/test.csv')
```


```python
# Get info on the train vs. test datasets
train_df.info()
print('_'*40)
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  891 non-null    int64
     1   Survived     891 non-null    int64
     2   Pclass       891 non-null    int64
     3   Name         891 non-null    object
     4   Sex          891 non-null    object
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64
     7   Parch        891 non-null    int64
     8   Ticket       891 non-null    object
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object
     11  Embarked     889 non-null    object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    ________________________________________
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  418 non-null    int64
     1   Pclass       418 non-null    int64
     2   Name         418 non-null    object
     3   Sex          418 non-null    object
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64
     6   Parch        418 non-null    int64
     7   Ticket       418 non-null    object
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object
     10  Embarked     418 non-null    object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB



```python
# Get statistics on the numerical categories in the dataset like Age and PClass(1,2,3)
train_df.describe()
```





  <div id="df-e7dd5e51-215b-4e60-8b2d-c9ac6cb46e6b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e7dd5e51-215b-4e60-8b2d-c9ac6cb46e6b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e7dd5e51-215b-4e60-8b2d-c9ac6cb46e6b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e7dd5e51-215b-4e60-8b2d-c9ac6cb46e6b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f6551860-ab94-48d4-892b-ab95b88347cc">
  <button class="colab-df-quickchart" onclick="quickchart('df-f6551860-ab94-48d4-892b-ab95b88347cc')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f6551860-ab94-48d4-892b-ab95b88347cc button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Describe categorical data
# the include=['O'] means to include types that are objects
train_df.describe(include='O')
```





  <div id="df-6a899532-9b36-47c4-bbd5-6317761300e3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>347082</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6a899532-9b36-47c4-bbd5-6317761300e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6a899532-9b36-47c4-bbd5-6317761300e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6a899532-9b36-47c4-bbd5-6317761300e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5d784098-643e-4f68-8482-4a9f0f7f6525">
  <button class="colab-df-quickchart" onclick="quickchart('df-5d784098-643e-4f68-8482-4a9f0f7f6525')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5d784098-643e-4f68-8482-4a9f0f7f6525 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Mean statistics of each class. Mean Calculation is:
**total of survived(survived = 1) / total(survived=0 + survived=1)**


```python
# P class vs Survived Mean Stat
# Mean stat is where we get the mean of the values for each category in PClass that survived
# This gives the mean of each PClass that survived.
# Look at Class 3!!!
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```





  <div id="df-b7f26912-e6ed-45c2-bdde-15af7c9915f2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b7f26912-e6ed-45c2-bdde-15af7c9915f2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b7f26912-e6ed-45c2-bdde-15af7c9915f2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b7f26912-e6ed-45c2-bdde-15af7c9915f2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-266de63e-e699-4b5b-b1c9-5e2750421a98">
  <button class="colab-df-quickchart" onclick="quickchart('df-266de63e-e699-4b5b-b1c9-5e2750421a98')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-266de63e-e699-4b5b-b1c9-5e2750421a98 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Females were more likely to survive than males
```





  <div id="df-4881aca3-90ca-452f-b5ea-e26036f06543" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4881aca3-90ca-452f-b5ea-e26036f06543')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4881aca3-90ca-452f-b5ea-e26036f06543 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4881aca3-90ca-452f-b5ea-e26036f06543');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9f453b4d-79fe-4c35-b38b-8fac752ff725">
  <button class="colab-df-quickchart" onclick="quickchart('df-9f453b4d-79fe-4c35-b38b-8fac752ff725')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9f453b4d-79fe-4c35-b38b-8fac752ff725 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# The station people left from might be correlated with their PClass
```





  <div id="df-a51e42aa-cee9-458e-87f5-ba83ae7dbd17" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.336957</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a51e42aa-cee9-458e-87f5-ba83ae7dbd17')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a51e42aa-cee9-458e-87f5-ba83ae7dbd17 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a51e42aa-cee9-458e-87f5-ba83ae7dbd17');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bf41154e-352c-493b-bf30-098b8b79960b">
  <button class="colab-df-quickchart" onclick="quickchart('df-bf41154e-352c-493b-bf30-098b8b79960b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bf41154e-352c-493b-bf30-098b8b79960b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Number of Parents+Children => Parch
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True)
```





  <div id="df-6917eee5-24a7-418a-a2bc-b0fed81d2a23" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Parch</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6917eee5-24a7-418a-a2bc-b0fed81d2a23')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6917eee5-24a7-418a-a2bc-b0fed81d2a23 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6917eee5-24a7-418a-a2bc-b0fed81d2a23');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bfc073ff-d097-45c1-b22d-2236f7e78411">
  <button class="colab-df-quickchart" onclick="quickchart('df-bfc073ff-d097-45c1-b22d-2236f7e78411')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bfc073ff-d097-45c1-b22d-2236f7e78411 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Sibsp => Number of siblings + Spouse
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=True)
```





  <div id="df-388367a9-ee5e-4e02-b3e1-68b93def7b69" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.535885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-388367a9-ee5e-4e02-b3e1-68b93def7b69')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-388367a9-ee5e-4e02-b3e1-68b93def7b69 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-388367a9-ee5e-4e02-b3e1-68b93def7b69');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-96cb9cde-2880-4e51-9817-430b8ad67777">
  <button class="colab-df-quickchart" onclick="quickchart('df-96cb9cde-2880-4e51-9817-430b8ad67777')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-96cb9cde-2880-4e51-9817-430b8ad67777 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### I wondered if the amount of SibSp or Parch is correlated with PClass, but I did not find anything compelling there


```python
train_df[['SibSp', 'Pclass']].groupby(['SibSp', 'Pclass'], as_index=False).value_counts()
```





  <div id="df-f6f2d2a7-c528-4ec5-b9b3-454af888198e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SibSp</th>
      <th>Pclass</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>55</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>3</td>
      <td>18</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>3</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f6f2d2a7-c528-4ec5-b9b3-454af888198e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f6f2d2a7-c528-4ec5-b9b3-454af888198e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f6f2d2a7-c528-4ec5-b9b3-454af888198e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b6d3318a-21cd-454b-a99d-5e6d322871f2">
  <button class="colab-df-quickchart" onclick="quickchart('df-b6d3318a-21cd-454b-a99d-5e6d322871f2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b6d3318a-21cd-454b-a99d-5e6d322871f2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
train_df[['Parch', 'Pclass']].groupby(['Parch', 'Pclass'], as_index=False).value_counts()
```





  <div id="df-846dbd06-ede9-40c4-936c-6a651cc90e48" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>381</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>3</td>
      <td>55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>2</td>
      <td>16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>3</td>
      <td>43</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-846dbd06-ede9-40c4-936c-6a651cc90e48')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-846dbd06-ede9-40c4-936c-6a651cc90e48 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-846dbd06-ede9-40c4-936c-6a651cc90e48');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a95504bf-4fcd-4c2b-8c5d-df75c82cd24b">
  <button class="colab-df-quickchart" onclick="quickchart('df-a95504bf-4fcd-4c2b-8c5d-df75c82cd24b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a95504bf-4fcd-4c2b-8c5d-df75c82cd24b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




#### Clearer view graphs. To get a clearer view of who survived vs. who did not for specific classes we are interested in


```python
# Get a clear view of Age range and who did not survive (0) vs. who survived (1)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```




    <seaborn.axisgrid.FacetGrid at 0x7fbc33717490>





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_28_1.png)




```python
# Age range and P Class for who did not survive(0) vs. who survived(1)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.9, bins=20)
grid.add_legend();
```



![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_29_0.png)




```python
# Station embarked, male/female, and P Class for who did not survive vs who survived
# As you can see the male survival rate is higher for the first two stations
# But then there is a higer survival rate
# The survival rate always drops off for lower PClass
  # (Except in station C for female PClass 3)
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
```

    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:718: UserWarning: Using the pointplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:723: UserWarning: Using the pointplot function without specifying `hue_order` is likely to produce an incorrect plot.
      warnings.warn(warning)





    <seaborn.axisgrid.FacetGrid at 0x7fbc3343a7d0>





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_30_2.png)




```python
# Different embarked stations and male vs. female at those stations
  # Survived vs did not survive
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, errorbar=None)
grid.add_legend()
```

    /usr/local/lib/python3.10/dist-packages/seaborn/axisgrid.py:718: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)





    <seaborn.axisgrid.FacetGrid at 0x7fbc3384d000>





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_31_2.png)



#### Transform the data to be fed into Machine Learning Algorithms
- by dropping features that will not have correlation to survival. Like ticket number and cabin
- And transforming categorical features like embarked station to numerical feature because the ML algorithms do not understand categorical features.
- Also creating new features out of the existing features that may be helpful. Like extracting titles from names.
- Filling in null values since some ML classifiers cannot handle null.


```python
# Correcting by dropping features
# Drop ticket and cabin number features
combine = [train_df, test_df]
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
```

    Before (891, 12) (418, 11) (891, 12) (418, 11)





    ('After', (891, 10), (418, 9), (891, 10), (418, 9))



The blog I am following talks about creating a new feature out of the name feature called titles. Because it may have some correlation to who survived and who did not.


```python
# Create title feature
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```





  <div id="df-2d3492e6-12cf-4184-8e6c-61dbb2fc26b2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2d3492e6-12cf-4184-8e6c-61dbb2fc26b2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2d3492e6-12cf-4184-8e6c-61dbb2fc26b2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2d3492e6-12cf-4184-8e6c-61dbb2fc26b2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-305c0cf7-8444-4cde-8099-ba61e2fdc90a">
  <button class="colab-df-quickchart" onclick="quickchart('df-305c0cf7-8444-4cde-8099-ba61e2fdc90a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-305c0cf7-8444-4cde-8099-ba61e2fdc90a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Mean statistic on survival with new title feature
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values('Survived')
```





  <div id="df-6868eee1-4e55-490e-8a89-bbfda6cc67f4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rare</td>
      <td>0.347826</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6868eee1-4e55-490e-8a89-bbfda6cc67f4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6868eee1-4e55-490e-8a89-bbfda6cc67f4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6868eee1-4e55-490e-8a89-bbfda6cc67f4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-cbd52f30-7cd7-468d-b7bb-544c7c8fae29">
  <button class="colab-df-quickchart" onclick="quickchart('df-cbd52f30-7cd7-468d-b7bb-544c7c8fae29')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-cbd52f30-7cd7-468d-b7bb-544c7c8fae29 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Transfer the rarer titles to a overall Rare category
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# See transformed dataset
train_df.head()
```





  <div id="df-75f0d16c-5497-466c-a445-901d7d2ac14f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-75f0d16c-5497-466c-a445-901d7d2ac14f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-75f0d16c-5497-466c-a445-901d7d2ac14f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-75f0d16c-5497-466c-a445-901d7d2ac14f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c27913bf-2003-4b58-b437-8275c16ebc75">
  <button class="colab-df-quickchart" onclick="quickchart('df-c27913bf-2003-4b58-b437-8275c16ebc75')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c27913bf-2003-4b58-b437-8275c16ebc75 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Drop name and PassengerId features for test and training datasets
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
```




    ((891, 9), (418, 9))




```python
# Convert categorical sex feature (male, female) to numeric feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
```





  <div id="df-6ae65876-bc95-41ab-9f65-9de95744afbf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6ae65876-bc95-41ab-9f65-9de95744afbf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6ae65876-bc95-41ab-9f65-9de95744afbf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6ae65876-bc95-41ab-9f65-9de95744afbf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a2828483-4121-4f88-b768-09e6c845f9a5">
  <button class="colab-df-quickchart" onclick="quickchart('df-a2828483-4121-4f88-b768-09e6c845f9a5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a2828483-4121-4f88-b768-09e6c845f9a5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Completing values for missing or null values
# For which columns?
# Age
print(len(train_df['Age']))
train_df['Age'].unique()
print(train_df['Age'].isna().sum())
# More accurate way of guessing missing values is to use other correlated features.
#  In our case we note correlation among Age, Gender, and Pclass.
#  Guess Age values using median values for Age across sets of Pclass and Gender feature combinations.
#  So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on..
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend()
```

    891
    177





    <seaborn.axisgrid.FacetGrid at 0x7fbc330248b0>





![png](/assets/images/TitanicKaggleChallenge_Blog_files/TitanicKaggleChallenge_Blog_40_2.png)




```python
# Guessed values of Age for different combinations of Pclass and Sex
# Fill in the guessed age values where age is null

guess_ages = np.zeros((2,3))
# Iterate over Sex (0,1) and Pclass(1,2,3) to calculate guessed values of Age for the six combinations
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    # Fill in the guessed age values where age is null in the dataset
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
```





  <div id="df-c177cbae-d00d-4e25-9937-86479bb0c24f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c177cbae-d00d-4e25-9937-86479bb0c24f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c177cbae-d00d-4e25-9937-86479bb0c24f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c177cbae-d00d-4e25-9937-86479bb0c24f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-04eebde4-c653-42ab-b592-f18f5d2fce3f">
  <button class="colab-df-quickchart" onclick="quickchart('df-04eebde4-c653-42ab-b592-f18f5d2fce3f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-04eebde4-c653-42ab-b592-f18f5d2fce3f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Create Age Bands and determine correlations with Survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```





  <div id="df-85a07866-f9b6-4cd8-817f-6a4daa284316" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AgeBand</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.08, 16.0]</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(16.0, 32.0]</td>
      <td>0.337374</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(32.0, 48.0]</td>
      <td>0.412037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(48.0, 64.0]</td>
      <td>0.434783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(64.0, 80.0]</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85a07866-f9b6-4cd8-817f-6a4daa284316')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-85a07866-f9b6-4cd8-817f-6a4daa284316 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85a07866-f9b6-4cd8-817f-6a4daa284316');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-10bde643-674b-4d52-b2d0-b7f9f0e012e0">
  <button class="colab-df-quickchart" onclick="quickchart('df-10bde643-674b-4d52-b2d0-b7f9f0e012e0')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-10bde643-674b-4d52-b2d0-b7f9f0e012e0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Replace age with ordinals for these bands
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
```





  <div id="df-1d28cd3c-ba05-4381-a0ca-69f570060f43" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>AgeBand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>(32.0, 48.0]</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1d28cd3c-ba05-4381-a0ca-69f570060f43')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1d28cd3c-ba05-4381-a0ca-69f570060f43 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1d28cd3c-ba05-4381-a0ca-69f570060f43');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5f194656-c0ba-4dbf-a38b-db00aa603b69">
  <button class="colab-df-quickchart" onclick="quickchart('df-5f194656-c0ba-4dbf-a38b-db00aa603b69')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5f194656-c0ba-4dbf-a38b-db00aa603b69 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Remove the age band feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
```





  <div id="df-309da4a6-465f-4135-b9d9-5cbc5330a770" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-309da4a6-465f-4135-b9d9-5cbc5330a770')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-309da4a6-465f-4135-b9d9-5cbc5330a770 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-309da4a6-465f-4135-b9d9-5cbc5330a770');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e84110d6-d9f8-4736-be45-4e1f927d60f4">
  <button class="colab-df-quickchart" onclick="quickchart('df-e84110d6-d9f8-4736-be45-4e1f927d60f4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e84110d6-d9f8-4736-be45-4e1f927d60f4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Create a new feature called 'FamilySize' which combines Parch and SibSp.
# This will enable us to drop Parch and SibSp from the datasets
for dataset in combine:
    # We are adding 1 to count ourselves as a person
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

```





  <div id="df-5cc8c243-5ad7-4979-8ec3-36ad45138614" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FamilySize</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.724138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.578431</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.552795</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.303538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5cc8c243-5ad7-4979-8ec3-36ad45138614')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5cc8c243-5ad7-4979-8ec3-36ad45138614 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5cc8c243-5ad7-4979-8ec3-36ad45138614');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-26a5f698-3a93-474a-8632-9f01bdbca898">
  <button class="colab-df-quickchart" onclick="quickchart('df-26a5f698-3a93-474a-8632-9f01bdbca898')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-26a5f698-3a93-474a-8632-9f01bdbca898 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Calculate statistics for people that survived who were alone
# Vs. ones that had family
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
```





  <div id="df-58c5bdb8-2da6-4004-966b-64ce1855c9ad" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IsAlone</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.505650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.303538</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-58c5bdb8-2da6-4004-966b-64ce1855c9ad')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-58c5bdb8-2da6-4004-966b-64ce1855c9ad button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-58c5bdb8-2da6-4004-966b-64ce1855c9ad');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-be16223b-4634-4dc5-b12a-652aeffb0924">
  <button class="colab-df-quickchart" onclick="quickchart('df-be16223b-4634-4dc5-b12a-652aeffb0924')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-be16223b-4634-4dc5-b12a-652aeffb0924 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Find the most frequently occurring port
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
```




    'S'




```python
# Fill missing port values with the most frequent occurring port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```





  <div id="df-b8a2a095-79e9-4547-9f26-6bc1ca352af1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.339009</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b8a2a095-79e9-4547-9f26-6bc1ca352af1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b8a2a095-79e9-4547-9f26-6bc1ca352af1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b8a2a095-79e9-4547-9f26-6bc1ca352af1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7919d66a-aa96-47d2-a807-c579a9521c6b">
  <button class="colab-df-quickchart" onclick="quickchart('df-7919d66a-aa96-47d2-a807-c579a9521c6b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7919d66a-aa96-47d2-a807-c579a9521c6b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Convert categorical port to numeric port feature
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
```





  <div id="df-85e25e78-dd70-46a8-a60a-e7d17431933b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85e25e78-dd70-46a8-a60a-e7d17431933b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-85e25e78-dd70-46a8-a60a-e7d17431933b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85e25e78-dd70-46a8-a60a-e7d17431933b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8c4fc058-915c-4573-a891-93aed11de53a">
  <button class="colab-df-quickchart" onclick="quickchart('df-8c4fc058-915c-4573-a891-93aed11de53a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8c4fc058-915c-4573-a891-93aed11de53a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# The test data set had a single missing value for fare so we are filling that in
# The model algorithm needs to operate on nonnull values
# Round off the fare to two decimals because it represents currency
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
```





  <div id="df-c7295633-4e29-46b8-9657-113886faf58e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c7295633-4e29-46b8-9657-113886faf58e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c7295633-4e29-46b8-9657-113886faf58e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c7295633-4e29-46b8-9657-113886faf58e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e6bbdb14-29a5-4ad7-b87e-31b75286ad2b">
  <button class="colab-df-quickchart" onclick="quickchart('df-e6bbdb14-29a5-4ad7-b87e-31b75286ad2b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e6bbdb14-29a5-4ad7-b87e-31b75286ad2b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Fareband and survived mean stat

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
```





  <div id="df-24306be8-0ce8-45e9-932e-0b90dfdfdd22" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FareBand</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.001, 7.91]</td>
      <td>0.197309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(7.91, 14.454]</td>
      <td>0.303571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(14.454, 31.0]</td>
      <td>0.454955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(31.0, 512.329]</td>
      <td>0.581081</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-24306be8-0ce8-45e9-932e-0b90dfdfdd22')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-24306be8-0ce8-45e9-932e-0b90dfdfdd22 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-24306be8-0ce8-45e9-932e-0b90dfdfdd22');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-25fb9c1d-ae58-4042-9525-2652c267bad7">
  <button class="colab-df-quickchart" onclick="quickchart('df-25fb9c1d-ae58-4042-9525-2652c267bad7')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-25fb9c1d-ae58-4042-9525-2652c267bad7 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Convert the Fare Feature to ordinal values based on FareBand
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.head(10)
```





  <div id="df-2b055f76-090f-46cb-84c2-69c08924d6e0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2b055f76-090f-46cb-84c2-69c08924d6e0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2b055f76-090f-46cb-84c2-69c08924d6e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2b055f76-090f-46cb-84c2-69c08924d6e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6d0f0e9f-2783-43b7-a5f7-0b91da341869">
  <button class="colab-df-quickchart" onclick="quickchart('df-6d0f0e9f-2783-43b7-a5f7-0b91da341869')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6d0f0e9f-2783-43b7-a5f7-0b91da341869 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# One last look at test_df
test_df.head(10)
```





  <div id="df-1ca9bb31-ba05-4334-a0b9-95d759ab7552" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1ca9bb31-ba05-4334-a0b9-95d759ab7552')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1ca9bb31-ba05-4334-a0b9-95d759ab7552 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1ca9bb31-ba05-4334-a0b9-95d759ab7552');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9740b693-dde0-4c50-82d1-654eee3cb6a9">
  <button class="colab-df-quickchart" onclick="quickchart('df-9740b693-dde0-4c50-82d1-654eee3cb6a9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9740b693-dde0-4c50-82d1-654eee3cb6a9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




## Begin Machine Learning Portion


```python
# Imports for Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
# We are ready to start training a model
# We are performing supervised learning because we are training it on our dataset
# And we are doing Classification and Regression

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
```




    ((891, 10), (891,), (418, 10))




```python
# Logistic Regression
# Explain how it works
# We have an equation like sigmoid(Wx + b) where we are trying to learn W(W is the coefficient) and b
# the sigmoid equation + a threshold for making the output either 0 or 1.
# The coefficient shows how its related to the value we are trying to predict.
  # Its the linear relationship. How changing a variable(W) can affect its output. W1x1 + w2x2 +...
  # W shows how much changing to change the input by to get the correct output

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# logreg.coef_
```




    80.25




```python
# Statistics using the coefficient from Logistic Regression
# This shows the correlation of the features to survived
# A positive correlation means as that feature increases the value of survived
# increases as well.

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
```





  <div id="df-0b871cc3-cdd0-41b8-9a63-75764973511f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>2.140744</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Title</td>
      <td>0.461005</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Embarked</td>
      <td>0.262530</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fare</td>
      <td>0.215071</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Age</td>
      <td>-0.034734</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Parch</td>
      <td>-0.034867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SibSp</td>
      <td>-0.228568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>FamilySize</td>
      <td>-0.263568</td>
    </tr>
    <tr>
      <th>9</th>
      <td>IsAlone</td>
      <td>-0.512509</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pclass</td>
      <td>-0.699627</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0b871cc3-cdd0-41b8-9a63-75764973511f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0b871cc3-cdd0-41b8-9a63-75764973511f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0b871cc3-cdd0-41b8-9a63-75764973511f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c9b20e9b-16ba-42bb-a25f-c654cb76295a">
  <button class="colab-df-quickchart" onclick="quickchart('df-c9b20e9b-16ba-42bb-a25f-c654cb76295a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c9b20e9b-16ba-42bb-a25f-c654cb76295a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Support Vector Machines
# We are trying to find the line of best fit between the two classes.
# We are trying to find the line in between the two closest nodes (closest in relation to the other class) of the two classes

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```




    82.04




```python
# Decision Tree
# A Decision tree starts out with a split of classes based on which split will have the most
# information gain(for entropy calculations) and the most purity (based on gini index calculations)
# The most information gain will be on the class with the least entropy see image above
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```




    89.11



#### Below graph shows how to decide on which class to split up based on entropy calculations for Decision Tree Algorithm.
- There is the Entropy method of splitting the decision tree and there is a gini index way of splitting the decision tree. Both are similar with regards to performance.

![image.png](data:image/png;base64,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)


```python
# Random Forest
# Wikipedia
# Random forests or random decision forests is an ensemble learning method for classification,
# regression and other tasks that operates by constructing a multitude of decision trees at training time.
# For classification tasks, the output of the random forest is the class selected by most trees.
# For regression tasks, the mean or average prediction of the individual trees is returned.
# Random decision forests correct for decision trees' habit of overfitting to their training set.
#

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```




    89.11




```python
# Perceptron
# How is a Perceptron different from Logistic Regression?
# A Perceptron can use the logistic function but that is the only thing Logistic Regression can use
# Also everything about it depends on how its implemented SciKit

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
```




    77.33




```python
# K-Nearest-Neighbors (kNN)
# Choosing K: You select the number of nearest neighbors, kk, which is a critical parameter.
  # The choice of kk can affect the robustness of the model against noise.
# Distance Metric: Calculate the distance between the point you are trying to classify (the query point) and all other points in the data set.
  # Common distance metrics include Euclidean, Manhattan, and Hamming distance.
# Find Nearest Neighbors: Identify the kk closest points to the query point in the feature space.
# Majority Vote for Classification:
  # For classification, the label of the query point is determined by the majority label among its kk nearest neighbors.

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```




    85.07




```python
# Gaussian Naive Bayes
# Regularly Naive Bayes multiplies prior probabilities to get the probability of the input
  # Find probability of classA | given word1 and word2
  # = (Probability that its classA) * (probability that word1 occurs in classA) * (probability that word2 occurs in classA)
# Gaussian Naive Bayes is the same multiplication but the value for probability is taken from the gaussian dist. of the data
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```




    78.45




```python
# Linear SVC - Support Vector Classifier
# What is a Linear SVC?
  # Its like SVM but the decision boundary is linear so its more efficient than SVM
  # This is because calculating the linear decision boundary is computationally simpler
  # than calculating non-linear decision boundaries that require transformations and complex calculations due to the kernel trick.
# Use Cases: It's best used when you have a reason to believe the data can be separated by a linear boundary or when computational efficiency is a concern.
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)
# print(linear_svc.coef_[0])
w = linear_svc.coef_[0]
print(w)
```

    80.13
    [-0.24660274  0.83744421 -0.02822989 -0.10231388 -0.01159634  0.0504932
      0.08623507  0.17555587 -0.08942259 -0.21129048]


    /usr/local/lib/python3.10/dist-packages/sklearn/svm/_base.py:1244: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(



```python
# Stochastic Gradient Descent
# Linear classifiers (SVM, logistic regression, etc.) with SGD training. (From docs)
# The reason its stochastic is because we are using mini-batches for computing the gradient of the loss function
  # as opposed to the entire dataset
  # and then with the gradients we make updates

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```




    72.62




```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
```





  <div id="df-4d8ad750-4c8c-4707-9e9e-1f8f84d26ffc" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>89.11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree</td>
      <td>89.11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>85.07</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Support Vector Machines</td>
      <td>82.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression</td>
      <td>80.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Linear SVC</td>
      <td>80.13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>78.45</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Perceptron</td>
      <td>77.33</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Stochastic Gradient Decent</td>
      <td>72.62</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4d8ad750-4c8c-4707-9e9e-1f8f84d26ffc')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4d8ad750-4c8c-4707-9e9e-1f8f84d26ffc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4d8ad750-4c8c-4707-9e9e-1f8f84d26ffc');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-990c7e48-6a0a-45fb-a5ee-e2e2ddc8259f">
  <button class="colab-df-quickchart" onclick="quickchart('df-990c7e48-6a0a-45fb-a5ee-e2e2ddc8259f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-990c7e48-6a0a-45fb-a5ee-e2e2ddc8259f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('./submission.csv', index=False)
```

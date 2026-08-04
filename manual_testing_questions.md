# Manual Testing Suite for Web Design AI Pipeline

This file contains 40 curated test queries designed to manually validate each of the 5 classification pipelines of the orchestrator. These prompts are structured to test edge cases, custom prompts (such as web aesthetics), execution bounds, and memory lookups.

---

## 1. SIMPLE Pipeline (Router Only)
*Designed to test direct, fast routing and short conversational responses without calling code runtimes.*

1. **Synchronous vs Asynchronous:**
   `Explain the difference between synchronous and asynchronous execution in Python in two paragraphs.`
2. **JSON Serialization:**
   `Translate the following Python dictionary into JSON: {'name': 'John', 'age': 30, 'city': 'New York'}.`
3. **General Knowledge:**
   `Provide a quick summary of what a Docker container is and why it's useful.`
4. **Command Syntax:**
   `How do you check the version of Python installed on your system using the terminal?`
5. **Basic Facts:**
   `What is the capital city of Australia, and what is its current estimated population?`
6. **Best Practices Checklist:**
   `Give me a quick checklist of best practices for writing clean Git commit messages.`
7. **Python Package Concept:**
   `What is the purpose of the __init__.py file in a Python package?`
8. **Regex Query:**
   `Write a regex pattern to validate standard email addresses.`

---

## 2. CODING Pipeline (Actor-Critic + Dual Sandbox)
*Designed to trigger the code generation, syntax checks, test execution loops, and custom design logic.*

9. **Premium Web UI Design (Aesthetics Check):**
   `Create a single-page interactive web dashboard for a fitness tracker. Use a slate/teal color palette, glassmorphism card layouts, a responsive grid for widgets (active time, steps, calories, heart rate), a sticky header navigation, and a clean professional footer. Implement the logic using vanilla JS.`
10. **Recursive Directory Parsing:**
    `Write a Python script that walks through a directory recursively, finds all .log files, extracts lines containing the word 'ERROR', and writes them to a consolidated error_summary.log file.`
11. **Anagram Grouper Algorithm:**
    `Write a Python function find_anagrams(word_list) that groups a list of words into lists of anagrams (e.g. ['eat', 'tea', 'tan', 'ate', 'nat', 'bat'] -> [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]).`
12. **Coffee Shop Page (Web UI):**
    `Create a clean, responsive HTML/CSS landing page for a coffee shop. It must feature a dark mode theme, hero section with a call-to-action button, menu grids with hover effects, and a contact form.`
13. **Priority Queue Manager:**
    `Write a Python class TaskManager that implements a priority queue for tasks, allowing tasks to be added with a priority level (1-5) and executed in priority order.`
14. **Custom Data Structure:**
    `Implement a custom LRU (Least Recently Used) cache class in Python using a doubly linked list and a hash map.`
15. **Debugging & Refactoring:**
    `Debug the following code which is supposed to run binary search but contains an infinite loop:
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid
            else:
                high = mid
        return -1`
16. **XML to CSV Converter:**
    `Write a Python program that parses an XML file of student records and converts it into a structured CSV file.`

---

## 3. REASONING Pipeline (VibeThinker / DeepSeek + Verification)
*Designed to test complex step-by-step logic, math derivations, physics proofs, and playground verification.*

17. **Probability Calculation:**
    `A box contains 5 red balls and 3 blue balls. If we select 3 balls at random without replacement, what is the probability that exactly 2 of them are red? Show your step-by-step derivation.`
18. **Physics Relative Velocity:**
    `A train leaves Station A at 60 mph heading east. Another train leaves Station B at 80 mph heading west. The distance between the stations is 280 miles. If they leave at the same time, how long will it take for them to meet? Derive the equation and verify it.`
19. **Logic Puzzle:**
    `Solve the classic logic puzzle: Three people (Alice, Bob, and Charlie) are suspects in a crime. Alice says: 'Bob is guilty.' Bob says: 'Charlie is guilty.' Charlie says: 'Bob is lying.' If only one statement is true, who is guilty?`
20. **Taylor Series Derivation:**
    `Find the first 5 terms of the Taylor series expansion of f(x) = ln(1+x) around x=0, and show your step-by-step derivation.`
21. **Algorithmic Proof:**
    `Design an O(N log N) algorithm to find the longest increasing subsequence in an array of integers. Provide a step-by-step proof of its correctness.`
22. **Raindrop Moisture Accumulation:**
    `A spherical raindrop of initial radius R_0 falls under gravity. As it falls, it accumulates moisture from a surrounding cloud at a rate proportional to its surface area. Derive the radius of the raindrop as a function of time t.`
23. **Mathematical Limit Verification:**
    `Determine the limit of (1 - cos(x)) / x^2 as x approaches 0 using L'Hôpital's Rule and Taylor series expansion, verifying that both methods yield the same result.`
24. **N-Queens Solver Formulation:**
    `Write a program to solve the N-Queens problem for N=8 and print the number of unique valid configurations.`

---

## 4. PREDICTION Pipeline (Machine Learning / Data Science)
*Designed to evaluate ML models, preprocessing pipelines, training logs, and evaluations (accuracy, MSE, R2 scores).*

25. **Random Forest Classification:**
    `Train a RandomForestClassifier on a synthetic classification dataset using scikit-learn. Use train_test_split, calculate the accuracy and confusion matrix, and output the classification report.`
26. **Linear Regression from Scratch:**
    `Implement a linear regression model from scratch in Python (using numpy) to forecast house prices based on a set of features. Calculate the Mean Squared Error (MSE) and R-squared score.`
27. **Pandas Data Aggregation:**
    `Write a pandas script to load a CSV of sales data, clean missing values, group by 'Category' and 'Month', calculate total revenue, and find the top 3 selling products per category.`
28. **K-Means Clustering Segmentation:**
    `Apply K-Means clustering to segment customers based on their annual income and spending score. Use the elbow method to determine the optimal number of clusters.`
29. **Neural Network Training Loop:**
    `Build a simple feedforward neural network in Python (using NumPy or PyTorch) to classify handwritten digits (like MNIST). Show the training loop and calculate validation loss.`
30. **Churn Prediction with Imbalance:**
    `Create a logistic regression model to predict customer churn. Preprocess categorical features using one-hot encoding, handle imbalanced classes using SMOTE or class weights, and print the ROC-AUC score.`
31. **Time-Series Temperature Prediction:**
    `Implement a time-series forecasting model using an autoregressive (AR) approach to predict the next 10 days of temperature readings from a given history.`
32. **Scikit-Learn Preprocessing Pipeline:**
    `Write a scikit-learn pipeline that preprocesses a dataset (fills numerical missing values with median, scales features, one-hot encodes categoricals) and feeds them into a Support Vector Machine classifier.`

---

## 5. EXTREME WEBSEARCH Pipeline (Deep Web Analysis & Tabular Reporting)
*Designed to trigger active search scraping, page-by-page Jaccard deduplication, and markdown synthesis.*

33. **Tech Stocks Comparison:**
    `Search the web for the latest stock price of NVIDIA (NVDA), Apple (AAPL), and Microsoft (MSFT). Create a Markdown table comparing their current price, daily change, and 52-week high/low.`
34. **AI Launch Summarization:**
    `What are the latest developments, models, and features announced by OpenAI in their most recent developer update or event? Search the web and summarize the top 3 key takeaways.`
35. **Sports Standings Tracker:**
    `Search online for the current standings of the English Premier League (EPL) top 5 teams. Provide their points, matches played, and goal difference.`
36. **Crypto Weekly Change:**
    `What is the current price of Bitcoin (BTC) and Ethereum (ETH) today? Search the web and calculate the percentage difference compared to their prices exactly 7 days ago.`
37. **Flagship Smartphone Face-off:**
    `Search for the latest reviews and ratings of the Samsung Galaxy S24 Ultra and compare it with the iPhone 15 Pro Max. Summarize their pros, cons, and current pricing in a comparison matrix.`
38. **Box Office Earnings:**
    `What is the current box office performance of the top 3 movies playing in theaters right now? Search the web and list their weekend earnings.`
39. **Global Weather Comparison:**
    `Search the web for the weather forecast in Tokyo, London, and New York for the next 3 days. List the high/low temperatures and conditions in a table.`
40. **Space Program Status:**
    `Find the latest news regarding the Artemis space program. What is the scheduled launch date for the next Artemis mission?`

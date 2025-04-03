import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# ML Model Training
# -----------------------------------------------------------------------------
def train_model(csv_path: str) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier on the provided CSV data.
    
    The CSV file is assumed to have a three-row header:
      - Row 1: table identifiers (e.g., table_0, table_1, ...)
      - Row 2: stream type (player or dealer)
      - Row 3: attribute ('spy' or 'card')
      
    This function collects all (spy, card) pairs from all tables/streams and trains
    a decision tree classifier to perfectly capture the deterministic mapping.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        
    Returns:
        DecisionTreeClassifier: A trained classifier.
    """
    # Read CSV file with a multi-index header (3 rows)
    data = pd.read_csv(csv_path, header=[0, 1, 2])
    X = []
    y = []
    
    # Iterate over columns and look for those corresponding to the spy values.
    for col in data.columns:
        if col[2] == 'spy':
            card_col = (col[0], col[1], 'card')
            if card_col in data.columns:
                spy_vals = data[col].values
                card_vals = data[card_col].values
                X.extend(spy_vals.tolist())
                y.extend(card_vals.tolist())
    
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    # Train a Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

# Train the model on the CSV file (make sure "tables.csv" is in your directory)
MODEL = train_model("train.csv")

# -----------------------------------------------------------------------------
# get_card_value_from_spy_value Function
# -----------------------------------------------------------------------------
def get_card_value_from_spy_value(value: float) -> int:
    """
    Given a spy value (float), predict the corresponding card value (int) using a 
    trained machine learning model.
    
    This function uses the global MODEL which has been trained on the (spy, card)
    pairs from all tables.
    
    Parameters:
        value (float): The spy value.
        
    Returns:
        int: The corresponding card value (2–10, or 11 for Ace).
    """
    prediction = MODEL.predict(np.array([[value]]))
    return int(prediction[0])

# -----------------------------------------------------------------------------
# Functionality to Check Outputs from the Tables
# -----------------------------------------------------------------------------
def check_table_outputs(csv_path: str):
    """
    Load the CSV file containing the spy and card values for each table (for both 
    player and dealer streams), apply the get_card_value_from_spy_value function to 
    each spy value, and print a comparison of the predicted and true card values.
    
    Parameters:
        csv_path (str): Path to the CSV file (e.g., "train.csv").
    """
    data = pd.read_csv(csv_path, header=[0, 1, 2])
    results = []
    flag = True
    # Loop through each column that holds spy values.
    for col in data.columns:
        if col[2] == 'spy':
            card_col = (col[0], col[1], 'card')
            if card_col in data.columns:
                spy_vals = data[col].values
                true_cards = data[card_col].values
                for spy, true_card in zip(spy_vals, true_cards):
                    pred_card = get_card_value_from_spy_value(spy)
                    if pred_card != true_card:
                        print(f"Table {col[0]}, {col[1]} stream: Spy={spy}, True Card={true_card}, Predicted Card={pred_card}")
                        flag = False
                    results.append((col[0], col[1], spy, true_card, pred_card))
                    
    if not flag:
        print("Some predictions are incorrect.")
    df_results = pd.DataFrame(results, columns=["Table", "Type", "Spy", "True Card", "Predicted Card"])
    print(df_results)

# -----------------------------------------------------------------------------
# Main: For interactive testing
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # First, check the outputs from all tables:
    print("Checking table outputs:")
    check_table_outputs("train.csv")
    
    # Then, allow for an interactive test of a single spy value:
    try:
        user_value = float(input("Enter a spy value to predict its card value: "))
        print("Predicted card value:", get_card_value_from_spy_value(user_value))
    except Exception as e:
        print("Error in input:", e)

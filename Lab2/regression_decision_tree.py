import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph

class Node:
    def __init__(self, feature=None, node_type=None, threshold=None, branches=None, value=None):
        self.feature = feature  #name of column to split on
        self.node_type = node_type  #'categorical', 'numeric', or 'leaf'
        self.threshold = threshold  #threshold float value fro numeric split
        self.branches = (branches if branches is not None else {})  #dict mapping branch key -> Node
        self.value = value  #leaf prediction value

    def is_leaf(self):
        return self.value is not None


class CustomRegressionTree:
    def __init__(self, max_depth=5, min_samples_split=5, num_thresholds=5, categorical_cols=None, binary_cols=None, numeric_cols=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_thresholds = max(5, num_thresholds)
        self.categorical_cols = categorical_cols or []
        self.binary_cols = binary_cols or []
        self.numeric_cols = numeric_cols or []
        self.root = None

    def _variance(self, y):
        if len(y) <= 1:
            return 0.0
        return np.sum((y - np.mean(y)) ** 2)

    def _evaluate_split(self, y_subsets): #calculate sum of squared errors
        total_sse = sum(self._variance(y_sub) for y_sub in y_subsets)
        return total_sse

    def _find_best_split(self, X, y): #goal - to minimize sse
        best_sse = float("inf")
        best_split = None
        parent_sse = self._variance(y)

        for col in X.columns:
            if col in self.categorical_cols: #if column is categorical feature
                categories = X[col].unique() #get all of values in column without repetition
                if len(categories) <= 1:
                    continue

                subsets_y = [y[X[col] == cat] for cat in categories] #split y into n separate groups (lists of y corresponding to each category: [[prices_cat1],[prices_cat2],[prices_cat3]])
                current_sse = self._evaluate_split(subsets_y)

                if current_sse < best_sse:
                    best_sse = current_sse
                    best_split = {
                        "feature": col,
                        "node_type": "categorical",
                        "categories": categories,
                        "threshold": None,
                    }

            elif col in self.binary_cols:
                vals = X[col].unique()
                if len(vals) <= 1:
                    continue

                val1 = vals[0]
                subsets_y = [y[X[col] == val1], y[X[col] != val1]]
                current_sse = self._evaluate_split(subsets_y)

                if current_sse < best_sse:
                    best_sse = current_sse
                    best_split = {
                        "feature": col,
                        "node_type": "binary",
                        "categories": [val1, "other"],
                        "threshold": None,
                    }

            #here we need to find feature, which ater splitting produces the lowest variance in target variable, 
            #because if we split data into 2 branches where each branch has least deviation between the values, model would perform good
            elif col in self.numeric_cols: #numerical feature
                unique_vals = np.sort(X[col].unique())
                if len(unique_vals) <= 1:
                    continue

                percentiles = np.linspace(10, 90, self.num_thresholds) #generate percentile markers
                thresholds = np.percentile(X[col].dropna(), percentiles) #convert percentiles into values
                thresholds = np.unique(thresholds) 

                for thresh in thresholds:
                    left_mask = X[col] <= thresh
                    right_mask = ~left_mask

                    if not np.any(left_mask) or not np.any(right_mask):
                        continue

                    subsets_y = [y[left_mask], y[right_mask]]
                    current_sse = self._evaluate_split(subsets_y)

                    if current_sse < best_sse:
                        best_sse = current_sse
                        best_split = {
                            "feature": col,
                            "node_type": "numeric",
                            "threshold": thresh,
                            "categories": None,
                        }

        #return best split only if it improves variance
        if best_sse < parent_sse:
            return best_split
        return None

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)

        #stopping cond
        if (depth >= self.max_depth or num_samples < self.min_samples_split or self._variance(y) == 0):
            return Node(value=np.mean(y) if num_samples > 0 else 0.0)

        split = self._find_best_split(X, y)
        if split is None:
            return Node(value=np.mean(y))

        node = Node(feature=split["feature"], node_type=split["node_type"], threshold=split["threshold"])
        col = split["feature"]

        if split["node_type"] == "categorical":
            for cat in split["categories"]:
                child_X = X[X[col] == cat]
                child_y = y[X[col] == cat]
                node.branches[cat] = self._build_tree(child_X, child_y, depth + 1)

        elif split["node_type"] == "binary":
            target_val = split["categories"][0]
            mask = X[col] == target_val

            node.branches[target_val] = self._build_tree(X[mask], y[mask], depth + 1)
            node.branches["other"] = self._build_tree(X[~mask], y[~mask], depth + 1)

        elif split["node_type"] == "numeric":
            thresh = split["threshold"]
            left_mask = X[col] <= thresh

            node.branches["<= threshold"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.branches["> threshold"] = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        X_df = X.copy().reset_index(drop=True)
        y_ser = y.copy().reset_index(drop=True)

        self.root = self._build_tree(X_df, y_ser, depth=0)
        return self

    def _predict_single(self, row, node):
        if node.is_leaf():
            return node.value

        val = row[node.feature]

        if node.node_type == "categorical":
            if val in node.branches:
                return self._predict_single(row, node.branches[val])
            else:
                leaf_vals = [self._get_leaf_mean(b) for b in node.branches.values()]
                return np.mean(leaf_vals)

        elif node.node_type == "binary":
            target_val = list(node.branches.keys())[0]
            if val == target_val:
                return self._predict_single(row, node.branches[target_val])
            else:
                return self._predict_single(row, node.branches["other"])

        elif node.node_type == "numeric":
            if val <= node.threshold:
                return self._predict_single(row, node.branches["<= threshold"])
            else:
                return self._predict_single(row, node.branches["> threshold"])

    def _get_leaf_mean(self, node):
        if node.is_leaf():
            return node.value
        return np.mean([self._get_leaf_mean(b) for b in node.branches.values()])

    def predict(self, X):
        X_df = X.copy().reset_index(drop=True)
        return np.array([self._predict_single(row, self.root) for _, row in X_df.iterrows()])

    def export_graphviz(self, filename="decision_tree"):
        """Generates a perfectly ordered top-down decision tree diagram PNG."""
        dot = Digraph(comment="Regression Tree")
        dot.attr(
            "node", shape="box", style="filled, rounded", fillcolor="lightblue"
        )

        def add_nodes(node, parent_id=None, edge_label=""):
            node_id = str(id(node))

            if node.is_leaf():
                label = f"Predict Price:\n${node.value:,.2f}"
                dot.node(node_id, label=label, fillcolor="#b2ebd2")
            else:
                if node.node_type == "numeric":
                    label = f"{node.feature}\n<= {node.threshold:.2f}"
                else:
                    label = f"Split: {node.feature}"
                dot.node(node_id, label=label)

            if parent_id is not None:
                dot.edge(parent_id, node_id, label=str(edge_label))

            if not node.is_leaf():
                for branch_key, child in node.branches.items():
                    add_nodes(child, parent_id=node_id, edge_label=branch_key)

        add_nodes(self.root)
        dot.render(filename, format="png", view=True)
        print(f"Tree saved as {filename}.png! 📸")
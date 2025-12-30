"""
Order and Sales Analytics Module
Uses Kaggle e-commerce order data
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class OrderAnalytics:
    def __init__(self):
        self.orders = pd.read_csv('data/external/List-of-Orders.csv')
        self.details = pd.read_csv('data/external/Order-Details.csv')
        self.targets = pd.read_csv('data/external/Sales-target.csv')
        
    def get_geographic_distribution(self):
        """Get orders by state"""
        state_dist = self.orders['State'].value_counts()
        return state_dist
    
    def get_profit_analysis(self):
        """Analyze profit by category"""
        profit_df = self.details.groupby('Category').agg({
            'Profit': 'sum',
            'Amount': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        return profit_df
    
    def compare_with_targets(self):
        """Compare actual vs target sales"""
        actual = self.details.groupby('Category')['Amount'].sum()
        targets = self.targets.groupby('Category')['Target'].sum()
        
        comparison = pd.DataFrame({
            'Actual': actual,
            'Target': targets
        })
        return comparison
    
    def get_top_cities(self):
        """Get top performing cities"""
        merged = pd.merge(self.orders, self.details, on='Order ID')
        city_sales = merged.groupby('City')['Amount'].sum().sort_values(ascending=False)
        return city_sales.head(10)

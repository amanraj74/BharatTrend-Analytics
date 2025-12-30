import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class VisualizationExporter:
    def __init__(self, df):
        self.df = df
        self.output_dir = 'data/visualizations'
        
        # CREATE FOLDER IF NOT EXISTS
        os.makedirs(self.output_dir, exist_ok=True)
        
    def category_analysis_chart(self):
        """Top categories visualization"""
        print("📊 Generating category analysis chart...")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top 10 Categories by Volume", "Average Price by Category"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        top_cats = self.df['main_category'].value_counts().head(10)
        
        fig.add_trace(
            go.Bar(x=top_cats.values, y=top_cats.index, orientation='h',
                   marker=dict(color=top_cats.values, colorscale='Blues'),
                   name='Volume', showlegend=False),
            row=1, col=1
        )
        
        price_by_cat = self.df.groupby('main_category')['selling_price'].mean().sort_values().tail(10)
        
        fig.add_trace(
            go.Bar(x=price_by_cat.values, y=price_by_cat.index, orientation='h',
                   marker=dict(color=price_by_cat.values, colorscale='Greens'),
                   name='Avg Price', showlegend=False),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Number of Products", row=1, col=1)
        fig.update_xaxes(title_text="Average Price (₹)", row=1, col=2)
        
        fig.update_layout(height=500, title_text="Category Analysis Dashboard",
                          font=dict(size=11))
        
        fig.write_html(f'{self.output_dir}/01_category_analysis.html')
        print("✅ Saved: 01_category_analysis.html")
    
    def price_distribution_chart(self):
        """Price distribution analysis"""
        print("📊 Generating price distribution chart...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Price Distribution Histogram", "Box Plot by Price Range",
                            "Price vs Discount Scatter", "Cumulative Price Distribution"),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.df['selling_price'], nbinsx=50, name='Price',
                         marker=dict(color='#FF6B35'), showlegend=False),
            row=1, col=1
        )
        
        # Box plot
        for price_range in self.df['price_range'].unique():
            if pd.notna(price_range):
                fig.add_trace(
                    go.Box(y=self.df[self.df['price_range']==price_range]['selling_price'],
                           name=str(price_range), showlegend=True),
                    row=1, col=2
                )
        
        # Scatter
        sample = self.df.sample(min(500, len(self.df)))
        fig.add_trace(
            go.Scatter(x=sample['selling_price'], y=sample['discount'],
                       mode='markers', marker=dict(size=5, color=sample['selling_price'],
                       colorscale='Viridis'), name='Products', showlegend=False),
            row=2, col=1
        )
        
        # Cumulative
        sorted_prices = self.df['selling_price'].sort_values()
        # --- MODIFIED SECTION BELOW ---
        cumulative = list(range(1, len(sorted_prices)+1))  # Convert to list
        
        fig.add_trace(
            go.Scatter(x=sorted_prices, y=cumulative, mode='lines',
                       fill='tozeroy', name='Cumulative', showlegend=False),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Price Range", row=1, col=2)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=2)
        fig.update_xaxes(title_text="Selling Price (₹)", row=2, col=1)
        fig.update_yaxes(title_text="Discount (%)", row=2, col=1)
        fig.update_xaxes(title_text="Price (₹)", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Count", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Price Analysis Dashboard",
                          font=dict(size=10), showlegend=True)
        
        fig.write_html(f'{self.output_dir}/02_price_analysis.html')
        print("✅ Saved: 02_price_analysis.html")
    
    def discount_strategy_chart(self):
        """Discount strategy analysis"""
        print("📊 Generating discount strategy chart...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Discount Distribution", "Discount by Category",
                            "Price vs Discount Relationship", "Discount Impact on Volume"),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.df['discount'], nbinsx=30, name='Discount %',
                         marker=dict(color='#FF006B'), showlegend=False),
            row=1, col=1
        )
        
        # Top discounts
        discount_by_cat = self.df.groupby('main_category')['discount'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=discount_by_cat.values, y=discount_by_cat.index, orientation='h',
                   marker=dict(color=discount_by_cat.values, colorscale='Reds'),
                   showlegend=False),
            row=1, col=2
        )
        
        # Scatter
        sample = self.df.sample(min(500, len(self.df)))
        fig.add_trace(
            go.Scatter(x=sample['selling_price'], y=sample['discount'],
                       mode='markers', marker=dict(size=5, color=sample['discount'],
                       colorscale='Hot'), showlegend=False),
            row=2, col=1
        )
        
        # Box plot
        discount_ranges = pd.cut(self.df['discount'], bins=[0, 25, 50, 75, 100],
                                 labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        for range_name in discount_ranges.unique():
            if pd.notna(range_name):
                fig.add_trace(
                    go.Box(y=self.df[discount_ranges==range_name]['selling_price'],
                           name=str(range_name)),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="Discount (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Average Discount (%)", row=1, col=2)
        fig.update_xaxes(title_text="Selling Price (₹)", row=2, col=1)
        fig.update_yaxes(title_text="Discount (%)", row=2, col=1)
        
        fig.update_layout(height=800, title_text="Discount Strategy Analysis",
                          font=dict(size=10))
        
        fig.write_html(f'{self.output_dir}/03_discount_analysis.html')
        print("✅ Saved: 03_discount_analysis.html")
    
    def market_segment_chart(self):
        """Market segmentation"""
        print("📊 Generating market segmentation chart...")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Market Share by Price Range", "Average Metrics by Segment"),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart
        price_dist = self.df['price_range'].value_counts()
        fig.add_trace(
            go.Pie(labels=price_dist.index, values=price_dist.values,
                   name='Segments', showlegend=True),
            row=1, col=1
        )
        
        # Bar chart
        segment_metrics = self.df.groupby('price_range')['selling_price'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=segment_metrics.index, y=segment_metrics.values,
                   marker=dict(color=segment_metrics.values, colorscale='Viridis'),
                   showlegend=False),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Average Price (₹)", row=1, col=2)
        
        fig.update_layout(height=500, title_text="Market Segmentation Analysis",
                          font=dict(size=11))
        
        fig.write_html(f'{self.output_dir}/04_market_segmentation.html')
        print("✅ Saved: 04_market_segmentation.html")
    
    def generate_all(self):
        """Generate all visualizations"""
        print("\n🎨 GENERATING HIGH-QUALITY VISUALIZATIONS...")
        print("="*60)
        
        self.category_analysis_chart()
        self.price_distribution_chart()
        self.discount_strategy_chart()
        self.market_segment_chart()
        
        print("\n✅ ALL VISUALIZATIONS GENERATED!")
        print(f"📁 Saved to: {self.output_dir}/")

def main():
    # Ensure this path is correct for your environment
    df = pd.read_csv('data/processed/enhanced_products.csv')
    exporter = VisualizationExporter(df)
    exporter.generate_all()

if __name__ == "__main__":
    main()
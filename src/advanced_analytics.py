import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Advanced analytics for professional insights"""
    
    def __init__(self, df):
        self.df = df
        self.insights = {}
        
    def competitive_analysis(self):
        """Analyze competitive landscape"""
        print("\n🏆 COMPETITIVE ANALYSIS")
        print("="*60)
        
        # Source-wise comparison
        source_analysis = self.df.groupby('source').agg({
            'selling_price': ['mean', 'min', 'max', 'std'],
            'discount': 'mean',
            'product': 'count'
        }).round(2)
        
        print("\n📊 Source Comparison:")
        print(source_analysis)
        
        # Category dominance
        print("\n🏪 Category Dominance by Source:")
        for source in self.df['source'].unique():
            source_df = self.df[self.df['source'] == source]
            top_cat = source_df['main_category'].value_counts().index[0]
            print(f"   {source}: {top_cat} ({len(source_df[source_df['main_category']==top_cat])} products)")
        
        return source_analysis
    
    def price_elasticity(self):
        """Analyze price vs demand relationship"""
        print("\n📈 PRICE ELASTICITY ANALYSIS")
        print("="*60)
        
        # Group by price ranges
        price_bins = [0, 500, 1000, 2500, 5000, 10000, np.inf]
        price_labels = ['<₹500', '₹500-1K', '₹1K-2.5K', '₹2.5K-5K', '₹5K-10K', '>₹10K']
        
        self.df['price_bucket'] = pd.cut(self.df['selling_price'], bins=price_bins, labels=price_labels)
        
        elasticity = self.df.groupby('price_bucket').agg({
            'product': 'count',
            'discount': 'mean',
            'selling_price': 'mean'
        })
        
        elasticity.columns = ['Demand (Products)', 'Avg Discount %', 'Avg Price']
        
        print("\n💰 Price Elasticity:")
        print(elasticity)
        
        return elasticity
    
    def customer_segmentation_insights(self):
        """Segment customers by purchasing power"""
        print("\n👥 CUSTOMER SEGMENTATION INSIGHTS")
        print("="*60)
        
        segments = {
            'Budget Conscious': self.df[self.df['price_range'] == 'Budget'],
            'Value Seekers': self.df[self.df['price_range'] == 'Economy'],
            'Middle Class': self.df[self.df['price_range'] == 'Mid-Range'],
            'Premium Buyers': self.df[self.df['price_range'] == 'Premium'],
            'Luxury Segment': self.df[self.df['price_range'] == 'Luxury']
        }
        
        print("\n📊 Segment Analysis:")
        for segment_name, segment_data in segments.items():
            if len(segment_data) > 0:
                print(f"\n🔹 {segment_name}:")
                print(f"   Size: {len(segment_data):,} products ({len(segment_data)/len(self.df)*100:.1f}%)")
                print(f"   Avg Price: ₹{segment_data['selling_price'].mean():.0f}")
                print(f"   Avg Discount: {segment_data['discount'].mean():.1f}%")
                print(f"   Top Category: {segment_data['main_category'].mode().values[0]}")
                print(f"   Price Range: ₹{segment_data['selling_price'].min():.0f} - ₹{segment_data['selling_price'].max():.0f}")
        
        return segments
    
    def discount_strategy_analysis(self):
        """Analyze discount patterns and strategies"""
        print("\n💹 DISCOUNT STRATEGY ANALYSIS")
        print("="*60)
        
        discount_segments = {
            'No Discount': self.df[self.df['discount'] <= 5],
            'Moderate (5-25%)': self.df[(self.df['discount'] > 5) & (self.df['discount'] <= 25)],
            'Heavy (25-50%)': self.df[(self.df['discount'] > 25) & (self.df['discount'] <= 50)],
            'Aggressive (>50%)': self.df[self.df['discount'] > 50]
        }
        
        print("\n🎯 Discount Strategy Distribution:")
        for strategy, data in discount_segments.items():
            if len(data) > 0:
                print(f"\n   {strategy}:")
                print(f"      Products: {len(data):,} ({len(data)/len(self.df)*100:.1f}%)")
                print(f"      Avg Selling Price: ₹{data['selling_price'].mean():.0f}")
                print(f"      Price Range: ₹{data['selling_price'].min():.0f} - ₹{data['selling_price'].max():.0f}")
        
        return discount_segments
    
    def profitability_analysis(self):
        """Estimate profitability metrics"""
        print("\n💰 PROFITABILITY ANALYSIS")
        print("="*60)
        
        # Calculate profit margin (assuming 20% cost of goods sold)
        self.df['cogs'] = self.df['selling_price'] * 0.20
        self.df['profit'] = self.df['selling_price'] - self.df['cogs']
        self.df['profit_margin'] = (self.df['profit'] / self.df['selling_price'] * 100)
        
        print("\n📊 Profitability by Category:")
        profitability = self.df.groupby('main_category').agg({
            'profit': 'mean',
            'profit_margin': 'mean',
            'product': 'count'
        }).sort_values('profit_margin', ascending=False).head(10)
        
        profitability.columns = ['Avg Profit (₹)', 'Profit Margin %', 'Products']
        print(profitability.round(2))
        
        return profitability
    
    def growth_opportunities(self):
        """Identify growth opportunities"""
        print("\n🚀 GROWTH OPPORTUNITIES")
        print("="*60)
        
        # High volume, low discount = mature categories
        mature = self.df.groupby('main_category').agg({
            'product': 'count',
            'discount': 'mean'
        })
        mature = mature[mature['product'] > 100].sort_values('product', ascending=False)
        mature = mature[mature['discount'] < 30]
        
        print(f"\n✅ STABLE GROWTH CATEGORIES (High Volume, Low Discount):")
        for cat in mature.index[:5]:
            count = mature.loc[cat, 'product']
            discount = mature.loc[cat, 'discount']
            print(f"   • {cat}: {int(count)} products, {discount:.1f}% avg discount")
        
        # High discount = aggressive marketing
        aggressive = self.df.groupby('main_category').agg({
            'product': 'count',
            'discount': 'mean'
        })
        aggressive = aggressive[aggressive['product'] > 50]
        aggressive = aggressive[aggressive['discount'] > 40].sort_values('discount', ascending=False)
        
        print(f"\n🎯 AGGRESSIVE MARKETING CATEGORIES (High Volume, High Discount):")
        for cat in aggressive.index[:5]:
            count = aggressive.loc[cat, 'product']
            discount = aggressive.loc[cat, 'discount']
            print(f"   • {cat}: {int(count)} products, {discount:.1f}% avg discount")
        
        # Niche = low volume, premium
        niche = self.df.groupby('main_category').agg({
            'product': 'count',
            'selling_price': 'mean'
        })
        niche = niche[niche['product'] < 100]
        niche = niche[niche['selling_price'] > 3000].sort_values('selling_price', ascending=False)
        
        print(f"\n💎 PREMIUM NICHE OPPORTUNITIES (Low Volume, High Price):")
        for cat in niche.index[:5]:
            count = niche.loc[cat, 'product']
            price = niche.loc[cat, 'selling_price']
            print(f"   • {cat}: {int(count)} products, ₹{price:.0f} avg price")

def main():
    """Main execution"""
    print("🚀 ADVANCED ANALYTICS ENGINE")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed/enhanced_products.csv')
    
    # Run analytics
    analyzer = AdvancedAnalytics(df)
    
    analyzer.competitive_analysis()
    analyzer.price_elasticity()
    analyzer.customer_segmentation_insights()
    analyzer.discount_strategy_analysis()
    analyzer.profitability_analysis()
    analyzer.growth_opportunities()
    
    print("\n" + "="*60)
    print("✅ ADVANCED ANALYTICS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

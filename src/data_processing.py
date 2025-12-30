import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.flipkart_df = None
        self.amazon_df = None
        self.orders_df = None
        self.order_details_df = None
        self.sales_target_df = None
        
    def load_data(self):
        """Load all datasets"""
        print("📂 Loading datasets...")
        
        try:
            self.flipkart_df = pd.read_csv('data/external/flipkart_com-ecommerce_sample.csv')
            print(f"✅ Flipkart: {len(self.flipkart_df)} products loaded")
        except Exception as e:
            print(f"❌ Flipkart Error: {e}")
            
        try:
            self.amazon_df = pd.read_csv('data/external/amazon.csv')
            print(f"✅ Amazon: {len(self.amazon_df)} products loaded")
        except Exception as e:
            print(f"❌ Amazon Error: {e}")
            
        try:
            self.orders_df = pd.read_csv('data/external/List-of-Orders.csv')
            print(f"✅ Orders: {len(self.orders_df)} orders loaded")
        except Exception as e:
            print(f"❌ Orders Error: {e}")
            
        try:
            self.order_details_df = pd.read_csv('data/external/Order-Details.csv')
            print(f"✅ Order Details: {len(self.order_details_df)} items loaded")
        except Exception as e:
            print(f"❌ Order Details Error: {e}")
            
        try:
            self.sales_target_df = pd.read_csv('data/external/Sales-target.csv')
            print(f"✅ Sales Target: {len(self.sales_target_df)} targets loaded")
        except Exception as e:
            print(f"❌ Sales Target Error: {e}")
    
    def explore_flipkart(self):
        """Analyze Flipkart dataset"""
        if self.flipkart_df is None:
            return
            
        print("\n📊 FLIPKART ANALYSIS")
        print("="*50)
        print(f"Total Products: {len(self.flipkart_df)}")
        print(f"\nColumns: {self.flipkart_df.columns.tolist()}")
        print(f"\nData Shape: {self.flipkart_df.shape}")
        print(f"\nSample Data:")
        print(self.flipkart_df.head())
        
        # Category analysis
        if 'product_category_tree' in self.flipkart_df.columns:
            print("\n🏷️ Top 10 Categories:")
            categories = self.flipkart_df['product_category_tree'].value_counts().head(10)
            print(categories)
        
        # Price analysis
        if 'retail_price' in self.flipkart_df.columns and 'discounted_price' in self.flipkart_df.columns:
            self.flipkart_df['retail_price'] = pd.to_numeric(self.flipkart_df['retail_price'], errors='coerce')
            self.flipkart_df['discounted_price'] = pd.to_numeric(self.flipkart_df['discounted_price'], errors='coerce')
            self.flipkart_df['discount_percent'] = ((self.flipkart_df['retail_price'] - self.flipkart_df['discounted_price']) / self.flipkart_df['retail_price'] * 100)
            
            print(f"\n💰 Price Statistics:")
            print(f"Avg Retail Price: ₹{self.flipkart_df['retail_price'].mean():.2f}")
            print(f"Avg Discounted Price: ₹{self.flipkart_df['discounted_price'].mean():.2f}")
            print(f"Avg Discount: {self.flipkart_df['discount_percent'].mean():.2f}%")
    
    def explore_amazon(self):
        """Analyze Amazon dataset"""
        if self.amazon_df is None:
            return
            
        print("\n📊 AMAZON ANALYSIS")
        print("="*50)
        print(f"Total Products: {len(self.amazon_df)}")
        print(f"\nColumns: {self.amazon_df.columns.tolist()}")
        print(f"\nData Shape: {self.amazon_df.shape}")
        print(f"\nSample Data:")
        print(self.amazon_df.head())
        
        # Rating analysis
        if 'rating' in self.amazon_df.columns:
            self.amazon_df['rating'] = pd.to_numeric(self.amazon_df['rating'], errors='coerce')
            print(f"\n⭐ Rating Statistics:")
            print(f"Average Rating: {self.amazon_df['rating'].mean():.2f}")
            print(f"Highest Rated: {self.amazon_df['rating'].max()}")
            print(f"Lowest Rated: {self.amazon_df['rating'].min()}")
        
        # Category analysis
        if 'category' in self.amazon_df.columns:
            print("\n🏷️ Top 10 Categories:")
            print(self.amazon_df['category'].value_counts().head(10))
    
    def explore_orders(self):
        """Analyze Orders dataset"""
        if self.orders_df is None:
            return
            
        print("\n📊 ORDERS ANALYSIS")
        print("="*50)
        print(f"Total Orders: {len(self.orders_df)}")
        print(f"\nColumns: {self.orders_df.columns.tolist()}")
        print(f"\nSample Data:")
        print(self.orders_df.head())
        
        # Date analysis
        date_columns = [col for col in self.orders_df.columns if 'date' in col.lower()]
        if date_columns:
            print(f"\n📅 Date Range:")
            for col in date_columns:
                try:
                    self.orders_df[col] = pd.to_datetime(self.orders_df[col], errors='coerce')
                    print(f"{col}: {self.orders_df[col].min()} to {self.orders_df[col].max()}")
                except:
                    pass
    
    def clean_and_merge(self):
        """Clean and merge datasets for analysis"""
        print("\n🧹 CLEANING AND MERGING DATA...")
        
        products = []
        
        # Process Flipkart
        if self.flipkart_df is not None:
            flipkart_clean = self.flipkart_df.copy()
            flipkart_clean['source'] = 'Flipkart'
            
            # Rename columns
            flipkart_clean = flipkart_clean.rename(columns={
                'product_name': 'product',
                'retail_price': 'original_price',
                'discounted_price': 'selling_price',
                'product_category_tree': 'category'
            })
            
            # Select only needed columns
            flipkart_clean = flipkart_clean[['product', 'original_price', 'selling_price', 'category', 'source']]
            products.append(flipkart_clean)
            print(f"✅ Flipkart: {len(flipkart_clean)} products processed")
        
        # Process Amazon - FIXED!
        if self.amazon_df is not None:
            amazon_clean = self.amazon_df.copy()
            amazon_clean['source'] = 'Amazon'
            
            # Rename columns
            amazon_clean = amazon_clean.rename(columns={
                'product_name': 'product',
                'actual_price': 'original_price',
                'discounted_price': 'selling_price'
            })
            
            # Clean price columns (remove ₹ and commas)
            for col in ['original_price', 'selling_price']:
                if col in amazon_clean.columns:
                    amazon_clean[col] = amazon_clean[col].astype(str).str.replace('₹', '').str.replace(',', '')
                    amazon_clean[col] = pd.to_numeric(amazon_clean[col], errors='coerce')
            
            # Keep rating if available
            if 'rating' in amazon_clean.columns:
                amazon_clean = amazon_clean[['product', 'original_price', 'selling_price', 'category', 'rating', 'source']]
            else:
                amazon_clean = amazon_clean[['product', 'original_price', 'selling_price', 'category', 'source']]
            
            products.append(amazon_clean)
            print(f"✅ Amazon: {len(amazon_clean)} products processed")
        
        if products:
            # Merge all dataframes
            unified_df = pd.concat(products, ignore_index=True)
            
            # Remove duplicates
            unified_df = unified_df.drop_duplicates(subset=['product'], keep='first')
            
            # Save
            unified_df.to_csv('data/processed/unified_products.csv', index=False)
            print(f"✅ Unified dataset created: {len(unified_df)} products")
            print(f"   - Flipkart: {len(unified_df[unified_df['source']=='Flipkart'])}")
            print(f"   - Amazon: {len(unified_df[unified_df['source']=='Amazon'])}")
            
            return unified_df
        
        return None
    
    def generate_insights(self):
        """Generate business insights"""
        print("\n💡 GENERATING INSIGHTS...")
        
        insights = {
            'total_products': 0,
            'total_orders': 0,
            'avg_discount': 0,
            'top_categories': [],
            'price_range': {},
        }
        
        if self.flipkart_df is not None:
            insights['total_products'] += len(self.flipkart_df)
        if self.amazon_df is not None:
            insights['total_products'] += len(self.amazon_df)
        if self.orders_df is not None:
            insights['total_orders'] = len(self.orders_df)
        
        print(f"\n📈 KEY INSIGHTS:")
        print(f"Total Products Available: {insights['total_products']:,}")
        print(f"Total Orders: {insights['total_orders']:,}")
        
        return insights

def main():
    """Main execution"""
    print("🚀 BHARATTREND DATA PROCESSOR")
    print("="*50)
    
    processor = DataProcessor()
    
    # Load all data
    processor.load_data()
    
    # Explore each dataset
    processor.explore_flipkart()
    processor.explore_amazon()
    processor.explore_orders()
    
    # Clean and merge
    unified_df = processor.clean_and_merge()
    
    # Generate insights
    insights = processor.generate_insights()
    
    print("\n✅ DATA PROCESSING COMPLETE!")
    print("\n📁 Processed data saved to: data/processed/unified_products.csv")

if __name__ == "__main__":
    main()
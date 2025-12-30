import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re
import warnings
warnings.filterwarnings('ignore')

class TrendPredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        
    def load_unified_data(self):
        """Load the unified dataset"""
        print("📂 Loading unified products...")
        self.df = pd.read_csv('data/processed/unified_products.csv')
        print(f"✅ Loaded {len(self.df)} products")
        return self.df
    
    def extract_main_category(self, category_str):
        """Extract main category from nested category string"""
        try:
            if pd.isna(category_str):
                return "Unknown"
            
            # For Flipkart format: ["Category >> SubCategory >> Item"]
            if ">>" in str(category_str):
                match = re.search(r'\["([^>]+)', str(category_str))
                if match:
                    return match.group(1).strip()
            
            # For Amazon format: Category|SubCategory|Item
            if "|" in str(category_str):
                return str(category_str).split("|")[0].strip()
            
            return str(category_str)[:50]
        except:
            return "Unknown"
    
    def prepare_data(self):
        """Prepare data for ML"""
        print("\n🔧 PREPARING DATA FOR ML...")
        
        # Extract main categories
        self.df['main_category'] = self.df['category'].apply(self.extract_main_category)
        
        # Clean prices
        self.df['original_price'] = pd.to_numeric(self.df['original_price'], errors='coerce')
        self.df['selling_price'] = pd.to_numeric(self.df['selling_price'], errors='coerce')
        
        # Calculate discount
        self.df['discount'] = ((self.df['original_price'] - self.df['selling_price']) / self.df['original_price'] * 100)
        self.df['discount'] = self.df['discount'].fillna(0)
        
        # Price range categories
        self.df['price_range'] = pd.cut(self.df['selling_price'], 
                                        bins=[0, 500, 1000, 2500, 5000, np.inf],
                                        labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury'])
        
        # Remove invalid rows
        self.df = self.df.dropna(subset=['selling_price', 'main_category'])
        
        print(f"✅ Data prepared: {len(self.df)} valid products")
        
        # Show category distribution
        print("\n📊 Top 10 Main Categories:")
        print(self.df['main_category'].value_counts().head(10))
        
        return self.df
    
    def price_prediction_model(self):
        """Train model to predict optimal selling price"""
        print("\n🤖 TRAINING PRICE PREDICTION MODEL...")
        
        # Prepare features
        df_model = self.df[['main_category', 'original_price', 'source', 'selling_price']].copy()
        df_model = df_model.dropna()
        
        # Encode categories
        le_cat = LabelEncoder()
        le_source = LabelEncoder()
        
        df_model['category_encoded'] = le_cat.fit_transform(df_model['main_category'])
        df_model['source_encoded'] = le_source.fit_transform(df_model['source'])
        
        # Features and target
        X = df_model[['category_encoded', 'original_price', 'source_encoded']]
        y = df_model['selling_price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained!")
        print(f"📊 Mean Absolute Error: ₹{mae:.2f}")
        print(f"📊 R² Score: {r2:.4f}")
        
        self.models['price_predictor'] = {
            'model': model,
            'le_cat': le_cat,
            'le_source': le_source,
            'mae': mae,
            'r2': r2
        }
        
        return model
    
    def product_clustering(self):
        """Cluster products into segments"""
        print("\n🎯 CLUSTERING PRODUCTS...")
        
        # Prepare data
        df_cluster = self.df[['selling_price', 'discount', 'main_category']].copy()
        df_cluster = df_cluster.dropna()
        
        # Encode category
        le = LabelEncoder()
        df_cluster['category_encoded'] = le.fit_transform(df_cluster['main_category'])
        
        # Features for clustering
        features = df_cluster[['selling_price', 'discount', 'category_encoded']]
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_cluster['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        print("\n📊 CLUSTER ANALYSIS:")
        for i in range(5):
            cluster_data = df_cluster[df_cluster['cluster'] == i]
            print(f"\n🔹 Cluster {i+1}:")
            print(f"   Products: {len(cluster_data)}")
            print(f"   Avg Price: ₹{cluster_data['selling_price'].mean():.2f}")
            print(f"   Avg Discount: {cluster_data['discount'].mean():.2f}%")
            print(f"   Top Category: {df_cluster[df_cluster['cluster'] == i]['main_category'].mode().values[0]}")
        
        self.models['clustering'] = {
            'model': kmeans,
            'scaler': scaler,
            'le': le
        }
        
        return kmeans
    
    def trend_analysis(self):
        """Analyze trending categories and price ranges"""
        print("\n📈 TREND ANALYSIS...")
        
        # Top trending categories by volume
        trending_categories = self.df['main_category'].value_counts().head(10)
        
        print("\n🔥 TOP 10 TRENDING CATEGORIES:")
        for cat, count in trending_categories.items():
            avg_price = self.df[self.df['main_category'] == cat]['selling_price'].mean()
            avg_discount = self.df[self.df['main_category'] == cat]['discount'].mean()
            print(f"   {cat}: {count} products | Avg ₹{avg_price:.0f} | {avg_discount:.1f}% off")
        
        # Price range analysis
        print("\n💰 PRICE RANGE DISTRIBUTION:")
        price_dist = self.df['price_range'].value_counts()
        for range_name, count in price_dist.items():
            print(f"   {range_name}: {count} products ({count/len(self.df)*100:.1f}%)")
        
        # Source comparison
        print("\n🏪 SOURCE COMPARISON:")
        for source in self.df['source'].unique():
            source_data = self.df[self.df['source'] == source]
            print(f"   {source}:")
            print(f"      Products: {len(source_data)}")
            print(f"      Avg Price: ₹{source_data['selling_price'].mean():.2f}")
            print(f"      Avg Discount: {source_data['discount'].mean():.2f}%")
    
    def save_insights(self):
        """Save processed data and insights"""
        print("\n💾 SAVING INSIGHTS...")
        
        # Save enhanced dataset
        self.df.to_csv('data/processed/enhanced_products.csv', index=False)
        print("✅ Saved: data/processed/enhanced_products.csv")
        
        # Create summary report
        summary = {
            'total_products': len(self.df),
            'avg_price': self.df['selling_price'].mean(),
            'avg_discount': self.df['discount'].mean(),
            'top_category': self.df['main_category'].value_counts().index[0],
            'categories_count': self.df['main_category'].nunique(),
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('data/processed/summary_insights.csv', index=False)
        print("✅ Saved: data/processed/summary_insights.csv")
        
        return summary

def main():
    """Main execution"""
    print("🚀 BHARATTREND ML MODELS")
    print("="*60)
    
    predictor = TrendPredictor()
    
    # Load and prepare data
    predictor.load_unified_data()
    predictor.prepare_data()
    
    # Train models
    predictor.price_prediction_model()
    predictor.product_clustering()
    
    # Analyze trends
    predictor.trend_analysis()
    
    # Save insights
    summary = predictor.save_insights()
    
    print("\n" + "="*60)
    print("✅ ML MODELS TRAINING COMPLETE!")
    print(f"📊 Total Products Analyzed: {summary['total_products']:,}")
    print(f"💰 Average Selling Price: ₹{summary['avg_price']:.2f}")
    print(f"🎯 Average Discount: {summary['avg_discount']:.2f}%")
    print(f"🔥 Top Category: {summary['top_category']}")
    print("="*60)

if __name__ == "__main__":
    main()

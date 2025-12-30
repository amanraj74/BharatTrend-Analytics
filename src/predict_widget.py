"""
Price Prediction Widget
"""
import pandas as pd
import joblib
import streamlit as st

def price_predictor():
    """Interactive price prediction widget"""
    st.markdown("### 🎯 AI Price Predictor")
    st.write("Get AI-powered price predictions based on market data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        original_price = st.number_input("Original Price (₹)", min_value=100, max_value=100000, value=2000, step=100)
    
    with col2:
        category = st.selectbox(
            "Category",
            ["Clothing", "Electronics", "Furniture", "Beauty & Personal Care", "Footwear"]
        )
    
    if st.button("🔮 Predict Selling Price", type="primary"):
        try:
            # Simple prediction logic based on average discount
            df = pd.read_csv('data/processed/enhanced_products.csv')
            
            # Get average discount for category
            cat_df = df[df['main_category'] == category]
            avg_discount = cat_df['discount'].mean() if len(cat_df) > 0 else 40
            
            # Calculate predicted price
            predicted_price = original_price * (1 - avg_discount/100)
            recommended_discount = avg_discount
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Selling Price", f"₹{predicted_price:,.0f}")
            
            with col2:
                st.metric("Recommended Discount", f"{recommended_discount:.1f}%")
            
            with col3:
                profit_margin = ((original_price - predicted_price) / original_price * 100)
                st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            st.success(f"💡 Based on {len(cat_df):,} similar products in {category}")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

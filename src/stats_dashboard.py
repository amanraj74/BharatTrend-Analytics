"""
Real-time Statistics Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def show_stats():
    """Display real-time statistics"""
    df = pd.read_csv('data/processed/enhanced_products.csv')
    
    st.markdown("## 📊 Real-Time System Statistics")
    
    # System metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📦 Total Products", f"{len(df):,}")
    
    with col2:
        categories = df['main_category'].nunique()
        st.metric("🏷️ Categories", categories)
    
    with col3:
        sources = df['source'].nunique()
        st.metric("🛒 Data Sources", sources)
    
    with col4:
        total_value = df['selling_price'].sum()
        st.metric("💰 Total Value", f"₹{total_value/1000000:.1f}M")
    
    with col5:
        avg_discount = df['discount'].mean()
        st.metric("🎯 Avg Discount", f"{avg_discount:.1f}%")
    
    # Data quality metrics
    st.markdown("### 📈 Data Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Completeness
        completeness = (df.notna().sum() / len(df) * 100).mean()
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = completeness,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Data Completeness"},
            delta = {'reference': 95},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 90], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price range distribution
        price_quality = len(df[df['selling_price'] > 0]) / len(df) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = price_quality,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Price Data Quality"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "gray"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Last updated
    st.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

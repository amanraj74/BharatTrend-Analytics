import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# --- IMPORT HELPER MODULES (Wrap in try-except for robustness) ---
try:
    from src.predict_widget import price_predictor
except ImportError:
    price_predictor = None

try:
    from src.stats_dashboard import show_stats
except ImportError:
    show_stats = None

# Page config
st.set_page_config(
    page_title="BharatTrend Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data"""
    try:
        df = pd.read_csv('data/processed/enhanced_products.csv')
        
        # Clean data
        df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
        df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce')
        
        # Fill NaN values
        df['discount'] = df['discount'].fillna(0)
        df['selling_price'] = df['selling_price'].fillna(0)
        df['original_price'] = df['original_price'].fillna(0)
        
        # Handle rating column if exists
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Remove invalid prices
        df = df[df['selling_price'] > 0]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_metric_card(title, value, icon="📊"):
    """Create a modern metric card"""
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">{icon}</h3>
            <h2 style="color: #333; margin: 0.5rem 0;">{value}</h2>
            <p style="color: #666; margin: 0;">{title}</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🇮🇳 BharatTrend Analytics</h1>
            <p>AI-Powered Market Intelligence for Indian E-commerce</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available. Please run data processing first.")
        st.code("python src/data_processing.py\npython src/ml_models.py")
        return
    
    # ============= SIDEBAR FILTERS =============
    with st.sidebar:
        st.title("🔍 Smart Filters")
        st.markdown("---")
        
        # Source filter
        st.subheader("📦 Data Source")
        sources = ['All'] + sorted(df['source'].dropna().unique().tolist())
        selected_source = st.selectbox("Select Marketplace", sources, key="source")
        
        # Categories
        st.subheader("🏷️ Categories")
        all_categories = sorted(df['main_category'].dropna().unique().tolist())
        popular_cats = df['main_category'].value_counts().head(10).index.tolist()
        
        show_all_cats = st.checkbox("Show all categories", value=False)
        
        if show_all_cats:
            selected_categories = st.multiselect(
                "Select Categories",
                all_categories,
                default=[]
            )
        else:
            selected_categories = st.multiselect(
                "Popular Categories",
                popular_cats,
                default=[]
            )
        
        # Price range
        st.subheader("💰 Price Range")
        max_price = min(int(df['selling_price'].max()), 100000)  # Cap at 1 lakh
        
        price_range = st.slider(
            "Select Price Range (₹)",
            min_value=0,
            max_value=max_price,
            value=(0, 10000),
            step=100
        )
        
        # Discount filter
        st.subheader("🎯 Discount Filter")
        min_discount = st.slider(
            "Minimum Discount (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5
        )
        
        # Rating filter (only if rating column exists)
        if 'rating' in df.columns:
            st.subheader("⭐ Rating")
            include_no_rating = st.checkbox("Include products without ratings", value=True)
            min_rating = st.slider(
                "Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )
        
        # Results limit
        st.subheader("📊 Results")
        results_limit = st.number_input(
            "Max products to show",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset All Filters"):
            st.rerun()

        # 6️⃣ ABOUT PAGE/INFO ⭐
        st.markdown("---")
        with st.expander("ℹ️ About BharatTrend"):
            st.markdown("""
            ### 🇮🇳 BharatTrend Analytics
            
            **Version:** 1.0.0  
            **Author:** Aman Jaiswal
            
            #### 📊 Features:
            - AI-Powered Price Prediction
            - Market Trend Analysis
            - Geographic Sales Insights
            - 13,950+ Products Analysis
            - Real-time Filtering
            
            #### 🎯 Technologies:
            - Python 3.10+
            - Streamlit
            - Scikit-learn
            - Plotly
            """)
    
    # ============= APPLY FILTERS =============
    filtered_df = df.copy()
    
    # Source filter
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    # Category filter
    if selected_categories:
        filtered_df = filtered_df[filtered_df['main_category'].isin(selected_categories)]
    
    # Price filter
    filtered_df = filtered_df[
        (filtered_df['selling_price'] >= price_range[0]) &
        (filtered_df['selling_price'] <= price_range[1])
    ]
    
    # Discount filter
    filtered_df = filtered_df[filtered_df['discount'] >= min_discount]
    
    # Rating filter
    if 'rating' in df.columns and min_rating > 0:
        if include_no_rating:
            filtered_df = filtered_df[
                (filtered_df['rating'] >= min_rating) | 
                (filtered_df['rating'].isna())
            ]
        else:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    # ============= TOP METRICS =============
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        create_metric_card("Products Found", f"{len(filtered_df):,}", "🛍️")
    
    with col2:
        avg_price = filtered_df['selling_price'].mean() if len(filtered_df) > 0 else 0
        create_metric_card("Avg Price", f"₹{avg_price:,.0f}", "💰")
    
    with col3:
        avg_discount = filtered_df['discount'].mean() if len(filtered_df) > 0 else 0
        create_metric_card("Avg Discount", f"{avg_discount:.1f}%", "🎯")
    
    with col4:
        categories = filtered_df['main_category'].nunique() if len(filtered_df) > 0 else 0
        create_metric_card("Categories", f"{categories}", "🏷️")
    
    with col5:
        if len(filtered_df) > 0 and 'price_range' in filtered_df.columns:
            top_range = filtered_df['price_range'].mode().values[0] if len(filtered_df['price_range'].mode()) > 0 else "N/A"
        else:
            top_range = "N/A"
        create_metric_card("Top Range", str(top_range), "📊")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============= TABS =============
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Products", "📊 Analytics", "💡 Insights", "🔥 Trends", "🗺️ Geographic"])
    
    with tab1:
        st.subheader("🛍️ Product Catalog")
        
        # 2️⃣ ADVANCED SEARCH ⭐⭐⭐
        st.markdown("### 🔍 Advanced Product Search")
        search_col1, search_col2, search_col3 = st.columns(3)
        with search_col1:
            search_query = st.text_input("Search by product name", placeholder="e.g., Samsung, Laptop, Shirt")
        with search_col2:
            available_categories = ['All'] + sorted(filtered_df['main_category'].unique().tolist())
            search_category = st.selectbox("Filter by category", available_categories, key="search_cat")
        with search_col3:
            search_source = st.selectbox("Filter by source", ['All', 'Flipkart', 'Amazon'], key="search_src")
            
        # Apply search filters to create a display dataframe
        search_df = filtered_df.copy()
        if search_query:
            search_df = search_df[search_df['product'].str.contains(search_query, case=False, na=False)]
        if search_category != 'All':
            search_df = search_df[search_df['main_category'] == search_category]
        if search_source != 'All':
            search_df = search_df[search_df['source'] == search_source]
            
        st.info(f"🔎 Found **{len(search_df):,}** products matching your search")
        
        if len(search_df) == 0:
            st.warning("⚠️ No products match your filters/search. Try adjusting the criteria.")
        else:
            # Sort options
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Price: Low to High", "Price: High to Low", "Discount: High to Low", "Name: A-Z"]
                )
            
            with col2:
                view_mode = st.radio("View:", ["Cards", "Table"], horizontal=True)
            
            # Apply sorting to the searched dataframe
            display_df = search_df.copy()
            
            if sort_by == "Price: Low to High":
                display_df = display_df.sort_values('selling_price')
            elif sort_by == "Price: High to Low":
                display_df = display_df.sort_values('selling_price', ascending=False)
            elif sort_by == "Discount: High to Low":
                display_df = display_df.sort_values('discount', ascending=False)
            else:
                display_df = display_df.sort_values('product')
            
            # Display products
            final_view_df = display_df.head(results_limit)
            
            if view_mode == "Cards":
                # Card view
                for idx, row in final_view_df.iterrows():
                    product_name = str(row['product'])[:80] if pd.notna(row['product']) else "Unknown Product"
                    category = str(row['main_category']) if pd.notna(row['main_category']) else "N/A"
                    price = row['selling_price'] if pd.notna(row['selling_price']) else 0
                    discount = row['discount'] if pd.notna(row['discount']) else 0
                    
                    st.markdown(f"""
                        <div class="product-card">
                            <h4 style="color: #667eea; margin: 0;">{product_name}...</h4>
                            <p style="margin: 0.5rem 0;">
                                <strong>Category:</strong> {category} | 
                                <strong>Price:</strong> ₹{price:,.0f} | 
                                <strong>Discount:</strong> {discount:.0f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            else:
                # Table view
                columns_to_show = ['product', 'main_category', 'selling_price', 'original_price', 'discount']
                if 'price_range' in final_view_df.columns:
                    columns_to_show.append('price_range')
                if 'source' in final_view_df.columns:
                    columns_to_show.append('source')
                
                st.dataframe(
                    final_view_df[columns_to_show],
                    use_container_width=True,
                    height=600
                )
            
            # 1️⃣ ADD EXPORT FUNCTIONALITY ⭐⭐⭐
            if len(display_df) > 0:
                st.markdown("### 📥 Export Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV Export
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Products as CSV",
                        data=csv,
                        file_name=f"bharattrend_products_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_products"
                    )
                
                with col2:
                    # Summary Report
                    summary = pd.DataFrame({
                        'Metric': ['Total Products', 'Avg Price', 'Avg Discount', 'Categories', 'Total Value'],
                        'Value': [
                            len(display_df),
                            f"₹{display_df['selling_price'].mean():.2f}",
                            f"{display_df['discount'].mean():.2f}%",
                            display_df['main_category'].nunique(),
                            f"₹{display_df['selling_price'].sum():,.2f}"
                        ]
                    })
                    csv_summary = summary.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Summary Report",
                        data=csv_summary,
                        file_name=f"bharattrend_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_summary"
                    )

            # 4️⃣ ADD COMPARISON TOOL ⭐⭐
            st.markdown("---")
            st.markdown("### ⚖️ Product Comparison Tool")
            
            if len(display_df) > 1:
                compare_col1, compare_col2 = st.columns(2)
                with compare_col1:
                    # Limit to first 100 for performance in dropdown
                    product1 = st.selectbox("Select Product 1", display_df['product'].head(100), key="prod1")
                with compare_col2:
                    product2 = st.selectbox("Select Product 2", display_df['product'].head(100), key="prod2")
                
                if st.button("Compare Products"):
                    p1 = display_df[display_df['product'] == product1].iloc[0]
                    p2 = display_df[display_df['product'] == product2].iloc[0]
                    
                    comparison = pd.DataFrame({
                        'Feature': ['Category', 'Selling Price', 'Original Price', 'Discount', 'Source'],
                        'Product 1': [
                            p1['main_category'], 
                            f"₹{p1['selling_price']:,.0f}", 
                            f"₹{p1['original_price']:,.0f}",
                            f"{p1['discount']:.1f}%", 
                            p1['source']
                        ],
                        'Product 2': [
                            p2['main_category'], 
                            f"₹{p2['selling_price']:,.0f}", 
                            f"₹{p2['original_price']:,.0f}",
                            f"{p2['discount']:.1f}%", 
                            p2['source']
                        ]
                    })
                    
                    st.dataframe(comparison, use_container_width=True)
                    
                    # Winner analysis
                    if p1['discount'] > p2['discount']:
                        st.success(f"🏆 Product 1 has better discount ({p1['discount']:.1f}% vs {p2['discount']:.1f}%)")
                    elif p2['discount'] > p1['discount']:
                        st.success(f"🏆 Product 2 has better discount ({p2['discount']:.1f}% vs {p1['discount']:.1f}%)")
                    else:
                        st.info("Both products have the same discount.")
            else:
                st.info("Need at least 2 products to compare.")
    
    with tab2:
        # 5️⃣ ADD STATS DASHBOARD ⭐⭐
        if show_stats:
            show_stats()
            st.markdown("---")
        elif 'src.stats_dashboard' not in str(globals()):
            # Fallback if module is missing
            st.info("Stats Dashboard module not found (src/stats_dashboard.py).")

        st.subheader("📊 Market Analytics")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data to display. Adjust filters.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                st.markdown("#### Top Categories by Volume")
                top_cats = filtered_df['main_category'].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_cats.values,
                        y=top_cats.index.tolist(),
                        orientation='h',
                        marker=dict(
                            color=top_cats.values,
                            colorscale='Viridis'
                        )
                    )
                ])
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    xaxis_title="Number of Products",
                    yaxis_title="",
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price distribution
                st.markdown("#### Price Range Distribution")
                if 'price_range' in filtered_df.columns:
                    price_dist = filtered_df['price_range'].value_counts()
                    
                    fig = px.pie(
                        values=price_dist.values,
                        names=price_dist.index.tolist(),
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Price range data not available")
            
            # Price vs Discount scatter
            st.markdown("#### Price vs Discount Analysis")
            sample = filtered_df.sample(min(500, len(filtered_df)))
            
            fig = px.scatter(
                sample,
                x='selling_price',
                y='discount',
                color='main_category',
                hover_data=['product'],
                labels={
                    'selling_price': 'Selling Price (₹)',
                    'discount': 'Discount (%)',
                    'main_category': 'Category'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💡 Smart Insights")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data for insights. Adjust filters.")
        else:
            # Generate insights
            insights = []
            
            # Top category
            top_cat = filtered_df['main_category'].value_counts().index[0]
            top_cat_count = len(filtered_df[filtered_df['main_category']==top_cat])
            insights.append(f"🔥 **{top_cat}** is the most popular category with **{top_cat_count:,}** products")
            
            # Best discount
            best_discount_cat = filtered_df.groupby('main_category')['discount'].mean().idxmax()
            best_discount_val = filtered_df.groupby('main_category')['discount'].mean().max()
            insights.append(f"💰 **{best_discount_cat}** offers best average discount of **{best_discount_val:.1f}%**")
            
            # Premium category
            premium_cat = filtered_df.groupby('main_category')['selling_price'].mean().idxmax()
            premium_val = filtered_df.groupby('main_category')['selling_price'].mean().max()
            insights.append(f"💎 **{premium_cat}** is the premium category with average price **₹{premium_val:,.0f}**")
            
            # Price range insights
            if 'price_range' in filtered_df.columns:
                budget_count = len(filtered_df[filtered_df['price_range'] == 'Budget'])
                budget_pct = (budget_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                insights.append(f"🎯 **{budget_pct:.1f}%** of products are in the Budget segment (<₹500)")
            
            for insight in insights:
                st.success(insight)
            
            # Recommendations
            st.markdown("### 🎯 Recommendations")
            st.info(f"""
            **For Buyers:**
            - Look for products in {best_discount_cat} for maximum savings
            - {top_cat} has the most variety with {top_cat_count:,} options
            
            **For Sellers:**
            - {top_cat} category shows high demand
            - Average market discount is {filtered_df['discount'].mean():.1f}% - price competitively
            """)
        
        # 3️⃣ ADD PREDICT WIDGET ⭐⭐⭐
        st.markdown("---")
        if price_predictor:
            price_predictor()
        elif 'src.predict_widget' not in str(globals()):
             st.info("Price Predictor module not found (src/predict_widget.py).")
    
    with tab4:
        st.subheader("🔥 Trending Products & Categories")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data for trends. Adjust filters.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### High Discount Deals")
                high_discount = filtered_df.nlargest(10, 'discount')[['product', 'discount', 'selling_price']]
                for idx, row in high_discount.iterrows():
                    product_name = str(row['product'])[:50] if pd.notna(row['product']) else "Unknown"
                    discount = row['discount'] if pd.notna(row['discount']) else 0
                    price = row['selling_price'] if pd.notna(row['selling_price']) else 0
                    st.markdown(f"""
                        - **{product_name}...** Discount: `{discount:.0f}%` | Price: `₹{price:,.0f}`
                    """)
            
            with col2:
                st.markdown("#### Premium Products")
                premium = filtered_df.nlargest(10, 'selling_price')[['product', 'selling_price', 'main_category']]
                for idx, row in premium.iterrows():
                    product_name = str(row['product'])[:50] if pd.notna(row['product']) else "Unknown"
                    price = row['selling_price'] if pd.notna(row['selling_price']) else 0
                    category = str(row['main_category']) if pd.notna(row['main_category']) else "N/A"
                    st.markdown(f"""
                        - **{product_name}...** Price: `₹{price:,.0f}` | Category: `{category}`
                    """)
    
    # GEOGRAPHIC TAB
    with tab5:
        st.subheader("🗺️ Geographic & Sales Analysis")
        
        try:
            # Load order data
            # Note: Checking for both filename versions (hyphenated and spaced) to be robust
            try:
                orders_df = pd.read_csv('data/external/List-of-Orders.csv')
                details_df = pd.read_csv('data/external/Order-Details.csv')
                targets_df = pd.read_csv('data/external/Sales-target.csv')
            except FileNotFoundError:
                orders_df = pd.read_csv('data/external/List of Orders.csv')
                details_df = pd.read_csv('data/external/Order Details.csv')
                targets_df = pd.read_csv('data/external/Sales target.csv')
            
            # Merge order and detail data
            order_data = pd.merge(orders_df, details_df, on='Order ID', how='inner')
            
            # Clean data
            order_data['Amount'] = pd.to_numeric(order_data['Amount'], errors='coerce')
            order_data['Profit'] = pd.to_numeric(order_data['Profit'], errors='coerce')
            order_data['Quantity'] = pd.to_numeric(order_data['Quantity'], errors='coerce')
            
            # Overview metrics
            st.markdown("### 📊 Sales Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_orders = len(orders_df)
                create_metric_card("Total Orders", f"{total_orders:,}", "📦")
            
            with col2:
                total_revenue = order_data['Amount'].sum()
                create_metric_card("Total Revenue", f"₹{total_revenue:,.0f}", "💰")
            
            with col3:
                total_profit = order_data['Profit'].sum()
                create_metric_card("Total Profit", f"₹{total_profit:,.0f}", "📈")
            
            with col4:
                avg_order = order_data['Amount'].mean()
                create_metric_card("Avg Order Value", f"₹{avg_order:,.0f}", "🛒")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Geographic Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🗺️ Top 10 States by Orders")
                state_orders = orders_df['State'].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=state_orders.values,
                        y=state_orders.index.tolist(),
                        orientation='h',
                        marker=dict(
                            color=state_orders.values,
                            colorscale='Blues'
                        ),
                        text=state_orders.values,
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    height=400,
                    xaxis_title="Number of Orders",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🏙️ Top 10 Cities by Revenue")
                city_revenue = order_data.groupby('City')['Amount'].sum().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=city_revenue.values,
                        y=city_revenue.index.tolist(),
                        orientation='h',
                        marker=dict(
                            color=city_revenue.values,
                            colorscale='Greens'
                        ),
                        text=[f"₹{val:,.0f}" for val in city_revenue.values],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    height=400,
                    xaxis_title="Revenue (₹)",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Category Performance
            st.markdown("### 📦 Category Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Revenue by Category")
                cat_revenue = order_data.groupby('Category')['Amount'].sum().sort_values(ascending=False)
                
                fig = px.pie(
                    values=cat_revenue.values,
                    names=cat_revenue.index.tolist(),
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Profit by Category")
                cat_profit = order_data.groupby('Category')['Profit'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=cat_profit.index.tolist(),
                    y=cat_profit.values,
                    color=cat_profit.values,
                    color_continuous_scale='RdYlGn',
                    labels={'x': 'Category', 'y': 'Profit (₹)'}
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Profit (₹)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sales vs Target Analysis
            st.markdown("### 🎯 Sales Target Performance")
            
            # Calculate actual sales by category
            actual_sales = order_data.groupby('Category')['Amount'].sum()
            
            # Parse target data
            targets_df['Target'] = pd.to_numeric(targets_df['Target'], errors='coerce')
            target_sales = targets_df.groupby('Category')['Target'].sum()
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Category': actual_sales.index,
                'Actual Sales': actual_sales.values,
                'Target Sales': [target_sales.get(cat, 0) for cat in actual_sales.index]
            })
            
            comparison_df['Achievement %'] = (comparison_df['Actual Sales'] / comparison_df['Target Sales'] * 100).round(2)
            
            # Bar chart comparison
            fig = go.Figure(data=[
                go.Bar(
                    name='Target Sales',
                    x=comparison_df['Category'],
                    y=comparison_df['Target Sales'],
                    marker_color='lightblue',
                    text=[f"₹{val:,.0f}" for val in comparison_df['Target Sales']],
                    textposition='auto',
                ),
                go.Bar(
                    name='Actual Sales',
                    x=comparison_df['Category'],
                    y=comparison_df['Actual Sales'],
                    marker_color='darkblue',
                    text=[f"₹{val:,.0f}" for val in comparison_df['Actual Sales']],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                barmode='group',
                height=400,
                xaxis_title="Category",
                yaxis_title="Sales Amount (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.markdown("#### 📊 Detailed Performance Metrics")
            
            # Add profit margin
            comparison_df['Profit'] = [cat_profit.get(cat, 0) for cat in comparison_df['Category']]
            comparison_df['Profit Margin %'] = (comparison_df['Profit'] / comparison_df['Actual Sales'] * 100).round(2)
            
            # Format for display
            display_df = comparison_df.copy()
            display_df['Actual Sales'] = display_df['Actual Sales'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Target Sales'] = display_df['Target Sales'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Profit'] = display_df['Profit'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True, height=200)
            
            # Key Insights
            st.markdown("### 💡 Key Geographic Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_state = state_orders.index[0]
                top_state_orders = state_orders.values[0]
                st.success(f"🏆 **Top State:** {top_state} with {top_state_orders} orders")
            
            with col2:
                top_city = city_revenue.index[0]
                top_city_rev = city_revenue.values[0]
                st.success(f"🏙️ **Top City:** {top_city} with ₹{top_city_rev:,.0f} revenue")
            
            with col3:
                best_category = cat_profit.index[0]
                best_cat_profit = cat_profit.values[0]
                st.success(f"📦 **Most Profitable:** {best_category} with ₹{best_cat_profit:,.0f}")
        
        except FileNotFoundError as e:
            st.warning("⚠️ Order data files not found. Please ensure the following files exist in `data/external/`:")
            st.code("""
- List-of-Orders.csv
- Order-Details.csv
- Sales-target.csv
            """)
            st.info("💡 Download from: https://www.kaggle.com/datasets/benroshan/ecommerce-data")
            
        except Exception as e:
            st.error(f"❌ Error loading geographic data: {e}")
            st.info("Please check that all CSV files are properly formatted.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p><strong>BharatTrend Analytics</strong> | Powered by AI & Data Science</p> 
            <p>📧 Contact: aerraj50@gmail.com | 🌐 GitHub: amanraj74<p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
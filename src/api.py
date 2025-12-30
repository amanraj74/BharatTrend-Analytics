"""
FastAPI REST API for BharatTrend predictions
"""
from fastapi import FastAPI, HTTPException
from pathlib import Path  # Add this line
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn

# Global variables
df = None
summary = None

from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    global df, summary
    try:
        # Get project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        
        df = pd.read_csv(project_root / 'data/processed/enhanced_products.csv')
        summary = pd.read_csv(project_root / 'data/processed/summary_insights.csv')
        print("✅ Data loaded successfully!")
        yield
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        yield
    finally:
        # Cleanup (if needed)
        print("🔄 Shutting down...")

app = FastAPI(
    title="BharatTrend API",
    description="AI-Powered Market Trend Analysis API",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan instead of on_event
)

class ProductQuery(BaseModel):
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 10

class PricePredict(BaseModel):
    category: str
    original_price: float

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to BharatTrend API",
        "version": "1.0.0",
        "status": "running",
        "total_products": len(df) if df is not None else 0,
        "endpoints": {
            "GET /stats": "Get overall statistics",
            "GET /categories": "Get all categories",
            "POST /products/search": "Search products",
            "GET /trends": "Get trending categories",
            "POST /predict/price": "Predict optimal selling price",
            "GET /insights": "Get market insights",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get overall market statistics"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_products": int(len(df)),
        "avg_price": float(df['selling_price'].mean()),
        "avg_discount": float(df['discount'].mean()),
        "categories": int(df['main_category'].nunique()),
        "price_range": {
            "min": float(df['selling_price'].min()),
            "max": float(df['selling_price'].max())
        }
    }

@app.get("/categories")
async def get_categories():
    """Get all product categories with counts"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    categories = df['main_category'].value_counts().head(20).to_dict()
    return {"categories": categories}

@app.post("/products/search")
async def search_products(query: ProductQuery):
    """Search products with filters"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    filtered = df.copy()
    
    if query.category:
        filtered = filtered[filtered['main_category'] == query.category]
    
    if query.min_price:
        filtered = filtered[filtered['selling_price'] >= query.min_price]
    
    if query.max_price:
        filtered = filtered[filtered['selling_price'] <= query.max_price]
    
    if len(filtered) == 0:
        raise HTTPException(status_code=404, detail="No products found")
    
    results = filtered.head(query.limit)[['product', 'main_category', 'selling_price', 'discount']].to_dict('records')
    
    return {
        "count": len(filtered),
        "showing": len(results),
        "results": results
    }

@app.get("/trends")
async def get_trends():
    """Get trending categories"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    top_categories = df['main_category'].value_counts().head(10)
    
    trends = []
    for cat, count in top_categories.items():
        cat_data = df[df['main_category'] == cat]
        trends.append({
            "category": cat,
            "products": int(count),
            "avg_price": float(cat_data['selling_price'].mean()),
            "avg_discount": float(cat_data['discount'].mean()),
            "price_range": str(cat_data['price_range'].mode().values[0]) if len(cat_data['price_range'].mode()) > 0 else "Unknown"
        })
    
    return {"trending_categories": trends}

@app.post("/predict/price")
async def predict_price(data: PricePredict):
    """Predict optimal selling price"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    category_data = df[df['main_category'] == data.category]
    
    if len(category_data) == 0:
        raise HTTPException(status_code=404, detail=f"Category '{data.category}' not found")
    
    avg_discount = category_data['discount'].mean()
    predicted_price = data.original_price * (1 - avg_discount/100)
    
    return {
        "category": data.category,
        "original_price": data.original_price,
        "predicted_selling_price": round(predicted_price, 2),
        "recommended_discount": round(avg_discount, 2),
        "market_avg_price": round(category_data['selling_price'].mean(), 2)
    }

@app.get("/insights")
async def get_insights():
    """Get market insights"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    insights = []
    
    # Top trending category
    top_cat = df['main_category'].value_counts().index[0]
    insights.append({
        "type": "trending",
        "message": f"{top_cat} is the most popular category",
        "data": {"category": top_cat, "products": int(df['main_category'].value_counts().values[0])}
    })
    
    # Best discount
    best_discount_cat = df.groupby('main_category')['discount'].mean().idxmax()
    best_discount_val = df.groupby('main_category')['discount'].mean().max()
    insights.append({
        "type": "discount",
        "message": f"{best_discount_cat} offers best discounts",
        "data": {"category": best_discount_cat, "avg_discount": round(best_discount_val, 2)}
    })
    
    # Premium category
    premium_cat = df.groupby('main_category')['selling_price'].mean().idxmax()
    premium_val = df.groupby('main_category')['selling_price'].mean().max() 
    insights.append({
        "type": "premium",
        "message": f"{premium_cat} is the premium category",
        "data": {"category": premium_cat, "avg_price": round(premium_val, 2)}
    })
    
    return {"insights": insights}

if __name__ == "__main__":
    print("🚀 Starting BharatTrend API...")
    print("📍 API will be available at: http://127.0.0.1:8000")
    print("📚 Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

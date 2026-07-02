# ============================================================
# DecodeLabs | Data Science Project 1
# Advanced EDA & Feature Engineering Pipeline
# ============================================================

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera as pa
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 0: LOAD DATA
# ============================================================

df = pd.read_csv("ecommerce_orders.csv", encoding='utf-8-sig')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ShippingAddress & PaymentMethod PDF se extract nahi hui -> placeholder
df['ShippingAddress'] = df['ShippingAddress'].fillna('N/A')
df['PaymentMethod']   = df['PaymentMethod'].fillna('N/A')

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df[['OrderID','Date','Product','Quantity','UnitPrice','OrderStatus','TotalPrice']].head(3).to_string())


# ============================================================
# PHASE 1: INPUT FIDELITY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1: SECURING INPUT FIDELITY")
print("=" * 60)

missing_pct = df.isnull().mean() * 100
print("\n[Missing Values % per Column]")
has_missing = missing_pct[missing_pct > 0]
if len(has_missing) > 0:
    print(has_missing.sort_values(ascending=False).to_string())
else:
    print("No missing values found.")

# ----------------------------------------------------------
# 1B: Numeric Missing Treatment (< 5% -> drop rows)
# ----------------------------------------------------------

print("\n[Missing Value Treatment]")

for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
    pct = df[col].isnull().mean() * 100
    if pct == 0:
        continue
    before = len(df)
    df.dropna(subset=[col], inplace=True)
    print(f"[Drop rows] {col} ({pct:.1f}%) -> dropped {before - len(df)} rows")

# CouponCode -> NO_COUPON
df['CouponCode'] = df['CouponCode'].fillna('NO_COUPON')
print("[Constant] CouponCode NaN -> 'NO_COUPON'")

print(f"\nMissing after treatment: {df.isnull().sum().sum()}")

# ----------------------------------------------------------
# 1C: Outlier Winsorization (IQR)
# ----------------------------------------------------------

print("\n[Outlier Treatment - IQR Winsorization]")

for col in ['UnitPrice', 'Quantity', 'TotalPrice', 'ItemsInCart']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = np.clip(df[col], lower, upper)
    print(f"  {col}: {n_out} outliers clipped [{lower:.2f}, {upper:.2f}]")


# ============================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: FEATURE ENGINEERING (Vectorized)")
print("=" * 60)

# Feature 1: Revenue Per Unit
df['RevenuePerUnit'] = df['TotalPrice'] / df['Quantity']
print("[OK] Feature 1: RevenuePerUnit = TotalPrice / Quantity")

# Feature 2: Cart Utilization Rate
df['CartUtilizationRate'] = (df['Quantity'] / df['ItemsInCart']).round(4)
print("[OK] Feature 2: CartUtilizationRate = Quantity / ItemsInCart")

# Feature 3: Is High Value Order
median_price = df['TotalPrice'].median()
df['IsHighValueOrder'] = (df['TotalPrice'] > median_price).astype(int)
print(f"[OK] Feature 3: IsHighValueOrder (threshold={median_price:.2f})")

# Feature 4: Temporal Features
df['OrderMonth']     = df['Date'].dt.month
df['OrderYear']      = df['Date'].dt.year
df['OrderDayOfWeek'] = df['Date'].dt.dayofweek
print("[OK] Feature 4: OrderMonth, OrderYear, OrderDayOfWeek")

# Feature 5: Has Coupon
df['HasCoupon'] = (df['CouponCode'] != 'NO_COUPON').astype(int)
print("[OK] Feature 5: HasCoupon (1=yes, 0=no)")

# Feature 6: Is Successful Order
df['IsSuccessful'] = df['OrderStatus'].isin(['Delivered', 'Shipped']).astype(int)
print("[OK] Feature 6: IsSuccessful (Delivered/Shipped=1)")

# ----------------------------------------------------------
# One-Hot Encoding
# ----------------------------------------------------------

print("\n[One-Hot Encoding]")
ohe_cols = ['Product', 'ReferralSource', 'OrderStatus']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print(f"[OK] Encoded: {ohe_cols}")

# ----------------------------------------------------------
# Multicollinearity Check
# ----------------------------------------------------------

print("\n[Multicollinearity Check]")

# Protect important features from being dropped
protected = ['TotalPrice', 'RevenuePerUnit']

num_feats   = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[num_feats].corr().abs()
upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [c for c in upper.columns
           if any(upper[c] > 0.80) and c not in protected]

if to_drop:
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    print(f"Dropped collinear columns: {to_drop}")
else:
    print("[OK] No multicollinearity issues.")


# ============================================================
# PHASE 3: PANDERA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3: PANDERA VALIDATION")
print("=" * 60)

all_checks = {
    'Quantity'        : pa.Column(float, pa.Check.between(0, 100)),
    'UnitPrice'       : pa.Column(float, pa.Check.ge(0)),
    'TotalPrice'      : pa.Column(float, pa.Check.ge(0)),
    'RevenuePerUnit'  : pa.Column(float, pa.Check.ge(0)),
    'IsHighValueOrder': pa.Column(int,   pa.Check.isin([0, 1])),
    'HasCoupon'       : pa.Column(int,   pa.Check.isin([0, 1])),
    'IsSuccessful'    : pa.Column(int,   pa.Check.isin([0, 1])),
}

# Sirf existing columns validate karo
active_checks = {k: v for k, v in all_checks.items() if k in df.columns}
schema = pa.DataFrameSchema(active_checks)

try:
    schema.validate(df, lazy=True)
    print("[OK] All Pandera checks PASSED!")
except pa.errors.SchemaErrors as e:
    print("[FAIL] Validation issues:")
    print(e.failure_cases[['column', 'check', 'failure_case']].to_string())


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE - FINAL SUMMARY")
print("=" * 60)
print("Final Shape          :", df.shape)
print("Missing values left  :", df.isnull().sum().sum())
print("Duplicate rows       :", df.duplicated().sum())

new_feats = ['RevenuePerUnit', 'CartUtilizationRate', 'IsHighValueOrder',
             'OrderMonth', 'OrderYear', 'OrderDayOfWeek', 'HasCoupon', 'IsSuccessful']
print("\nEngineered Features Status:")
for f in new_feats:
    status = "[OK]" if f in df.columns else "[MISSING]"
    print(f"  {status} {f}")

df.to_csv("ecommerce_orders_CLEANED.csv", index=False, encoding='utf-8-sig')
print("\nCleaned dataset saved: ecommerce_orders_CLEANED.csv")
print("=" * 60)
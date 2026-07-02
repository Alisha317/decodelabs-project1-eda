import pdfplumber
import pandas as pd
import re

rows = []

COLUMNS = [
    'OrderID', 'Date', 'CustomerID', 'Product', 'Quantity',
    'UnitPrice', 'ShippingAddress', 'PaymentMethod', 'OrderStatus',
    'TrackingNumber', 'ItemsInCart', 'CouponCode', 'ReferralSource', 'TotalPrice'
]

products    = ['Monitor', 'Phone', 'Tablet', 'Chair', 'Printer', 'Laptop', 'Desk']
statuses    = ['Shipped', 'Delivered', 'Cancelled', 'Pending', 'Returned']
pay_methods = ['Credit Card', 'Debit Card', 'Gift Card', 'Online', 'Cash']
referrals   = ['Instagram', 'Facebook', 'Google', 'Email', 'Referral']
coupons     = ['SAVE10', 'FREESHIP', 'WINTER15']

with pdfplumber.open("dataset.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split('\n'):
            line = line.strip()
            if not line.startswith('ORD'):
                continue

            try:
                # --- OrderID (ORD + 6 digits) ---
                order_match = re.search(r'(ORD\d{6})', line)
                order_id = order_match.group(1) if order_match else ''

                # --- Date: extract any 8-digit date after ORD number ---
                # PDF mein date aur OrderID chipke hain: ORD2000002023-01-04
                # So pehle full ORD string se date nikalo
                date_match = re.search(r'ORD\d{6}(\d{4}-\d{2}-\d{2})', line)
                date = date_match.group(1) if date_match else ''

                # --- CustomerID ---
                cust_match = re.search(r'(C\d{5})', line)
                cust_id = cust_match.group(1) if cust_match else ''

                # --- Product ---
                product = next((p for p in products if p in line), '')

                # --- Quantity (1-5) ---
                qty_match = re.search(r'\b([1-5])\s+\d+\.\d+', line)
                quantity = qty_match.group(1) if qty_match else ''

                # --- All decimal prices ---
                all_prices = re.findall(r'\b\d+\.\d+\b', line)
                unit_price  = all_prices[0]  if len(all_prices) >= 1 else ''
                total_price = all_prices[-1] if len(all_prices) >= 2 else ''

                # --- ShippingAddress: 2-3 digit number + Main St ---
                addr_match = re.search(r'(\d{2,4}\s+Main\s+St)', line)
                shipping = addr_match.group(1) if addr_match else ''

                # --- PaymentMethod ---
                pay = ''
                for pm in pay_methods:
                    if pm in line:
                        pay = pm
                        break

                # --- OrderStatus ---
                status = next((s for s in statuses if s in line), '')

                # --- TrackingNumber ---
                trk_match = re.search(r'(TRK\d+)', line)
                tracking = trk_match.group(1) if trk_match else ''

                # --- ItemsInCart (1-10 right after TRK number) ---
                items_match = re.search(r'TRK\d+\s+(\d{1,2})\b', line)
                items = items_match.group(1) if items_match else ''

                # --- CouponCode ---
                coupon = next((c for c in coupons if c in line), '')

                # --- ReferralSource ---
                referral = next((r for r in referrals if r in line), '')

                rows.append({
                    'OrderID'        : order_id,
                    'Date'           : date,
                    'CustomerID'     : cust_id,
                    'Product'        : product,
                    'Quantity'       : quantity,
                    'UnitPrice'      : unit_price,
                    'ShippingAddress': shipping,
                    'PaymentMethod'  : pay,
                    'OrderStatus'    : status,
                    'TrackingNumber' : tracking,
                    'ItemsInCart'    : items,
                    'CouponCode'     : coupon,
                    'ReferralSource' : referral,
                    'TotalPrice'     : total_price,
                })

            except Exception as e:
                continue

df = pd.DataFrame(rows, columns=COLUMNS)

# Empty strings ko NaN banao
df.replace('', pd.NA, inplace=True)

print("Rows extracted:", len(df))
print("\nMissing values:")
print(df.isnull().sum())
print("\nSample (first 5 rows):")
print(df[['OrderID','Date','PaymentMethod','ShippingAddress']].head(5).to_string())

df.to_csv("ecommerce_orders.csv", index=False, encoding='utf-8-sig')
print("\nSaved: ecommerce_orders.csv")
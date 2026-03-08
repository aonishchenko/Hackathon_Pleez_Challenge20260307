# Pleez Growth Engine

REST API + Web UI — analyses a single or multi-restaurant XLSX and returns the top 4 highest-ROI promotional actions per restaurant.

## Model
Scoring trained on 1,458 real promotions across 50 restaurants.
ROI matrix: cuisine × promo-type with rating / uptime / volume multipliers. WIN = >35% ROI.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# Run the server
python3 server.py
# Service runs on http://localhost:8000
```

## If you want to use through API

### Single restaurant (first restaurant_id in file)
```bash
curl -X POST http://localhost:8000/api/analyse \
     -F 'file=@restaurant_data.xlsx'
```

### All restaurants in file (batch)
```bash
curl -X POST http://localhost:8000/api/analyse-all \
     -F 'file=@multi_restaurant_data.xlsx'
```

## Input XLSX Schema

| Sheet | Required | Key Columns |
|-------|----------|-------------|
| `metrics` | **Yes** | restaurant_id, avg_rating, uptime, total_orders, new_users, returning_users, conversion_rate, rejected_orders |
| `restaurant_tags` | No | restaurant_id, cuisine_tag |
| `promotions` | No | restaurant_id, promotion_type, roi, discount_perc |
| `restaurants` | No | restaurant_id, avg_food_cost_perc |

## Dependencies
- Python 3.8+  ·  pandas  ·  openpyxl  ·  numpy  ·  stdlib only for HTTP

## Troubleshooting
ModuleNotFoundError: No module named 'pandas'
Install the required packages:
```bash
pip install -r requirements.txt
```
OSError: [Errno 48] Address already in use
The port 8000 is already in use by another process. To fix:

```bash
Find the process: lsof -i :8000
Kill it: kill <PID>
Restart the server: python3 server.py
```
To stop the server, press Ctrl+C in the terminal where it's running.


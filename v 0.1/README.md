# Pleez Growth Engine

REST API + Web UI that analyses a restaurant XLSX file and returns the top 5 highest-ROI promotional actions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python3 server.py
# Service runs on http://localhost:8000
```

## API Usage

### Analyse a restaurant (curl)
```bash
curl -X POST http://localhost:8000/api/analyse \
     -F 'file=@restaurant_data.xlsx'
```

### Analyse a restaurant (Python)
```python
import requests

with open("restaurant_data.xlsx", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyse",
        files={"file": ("data.xlsx", f)}
    )

report = response.json()["report"]
for action in report["top_actions"]:
    print(f"#{action['rank']} {action['label']}: {action['predicted_roi']}% ROI")
```

## Input XLSX Schema

| Sheet | Required | Key Columns |
|-------|----------|-------------|
| `metrics` | **Yes** | restaurant_id, avg_rating, uptime, total_orders, new_users, returning_users, conversion_rate, rejected_orders |
| `restaurant_tags` | No | restaurant_id, cuisine_tag |
| `promotions` | No | restaurant_id, promotion_type, roi, discount_perc, target_customers |
| `restaurants` | No | restaurant_id, avg_food_cost_perc |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check |
| GET | `/api/schema` | Full API docs (JSON) |
| POST | `/api/analyse` | Analyse restaurant XLSX |

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'
Install the required packages:
```bash
pip install -r requirements.txt
```

If using a virtual environment, activate it first:
```bash
source .venv/bin/activate  # or your venv activation command
pip install -r requirements.txt
```

### OSError: [Errno 48] Address already in use
The port 8000 is already in use by another process. To fix:
1. Find the process: `lsof -i :8000`
2. Kill it: `kill <PID>`
3. Restart the server: `python3 server.py`

To stop the server, press Ctrl+C in the terminal where it's running.

## Dependencies
- Python 3.8+
- pandas
- openpyxl
- numpy
- (all standard library for HTTP server)

## Model

Scoring engine trained on 1,458 real promotions across 50 restaurants (Dec 2025–Mar 2026).
Uses a cuisine × promotion-type ROI matrix with rating, uptime, volume, and context multipliers.
WIN threshold = 35% predicted ROI.

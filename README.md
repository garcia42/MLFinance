# MLfinance

2024학년도 1학기 MLfinLab Project Team repository


## Development

To control the containerized Jupyter server:
```sh
# In the foreground (ctrl-c to stop)
docker compose up

# In the background (`docker compose down` to stop)
docker compose up -d
```

The Jupyter server URL should be `http://localhost:8888`,
which you can use in VSCode to connect.


## TODO

1. Sequential trade models using kaggle dataset 30GB


## ES Data Options

1. DataShop
   - https://datashop.cboe.com/cme-futures-trades
   - Can get all trades since 2008 for 80 bucks, subscription 200
1. Portara
   - https://portaracqg.com/futures/int/ep
   - Tick level 1 is 700GB since 2008
   - 220 for continuous data, need to email for tick data
1. Databento


## Notes

- Need tick data for all contracts at all times.
- Kaggle
    - https://www.kaggle.com/datasets/purveshjain3/es-tick-data
    - This guy uploaded all tick data from 1997 for ES mini
    - Do I need level 1 tick data? For quotes
        - With just tick data I can do sequential trade models
        - Can't do additional features (https://github.com/QuantifiSogang/2024-02SeniorMLFinance/blob/main/Notes/Week10MicrostructuralFeatures/04AdditionalFeatures.ipynb)

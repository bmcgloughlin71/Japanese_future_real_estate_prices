# External data-source candidates

This note records candidate external datasets for future feature engineering. New external features must be merged in a time-safe way: for a transaction in year `Y`, use only data available before or during `Y`, preferably lagged to `Y - 1` where annual publication timing is uncertain.

| Rank | Dataset | URL | Expected value | Candidate features | Granularity | Leakage caveat |
|---:|---|---|---|---|---|---|
| 1 | MLIT Land Market Value Publication / 地価公示 L01 | https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-2024.html | Very high | nearest official land price ¥/㎡, distance to official point, YoY change, land-use attributes | geocoded points, annual | use year-matched or lagged values only |
| 2 | MLIT Prefectural Land Price Survey / 都道府県地価調査 L02 | https://nlftp.mlit.go.jp/ksj/old/datalist/old_KsjTmplt-L02.html | High | complementary nearest official land price and YoY change | geocoded points, annual/July survey | use year-matched or lagged values only |
| 3 | MLIT Station Passenger Counts S12 + Railway N02 | https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2024.html / https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-v3_1.html | High | nearest station, station ridership, line/operator, distance to high-ridership stations | station/rail network | ridership should be lagged by transaction year |
| 4 | MLIT Zoning / 用途地域 A29 | https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-A29-v2_1.html | Medium-high | zoning category, residential/commercial/industrial flag, legal buildability context | polygons | ensure zoning vintage is not after transaction where possible |
| 5 | e-Stat municipal taxable income | https://www.e-stat.go.jp/dbview?sid=0000020103 | Medium | taxable income, taxpayers, income per taxpayer | municipality/year | lag by year |
| 6 | MLIT construction cost deflator | https://www.mlit.go.jp/statistics/details/t-other-2_tk_000362.html | Medium | construction cost index by year/month | national/macro time series | safe if publication/date handled consistently |
| 7 | BOJ average contract interest rates IR04 | https://www.boj.or.jp/en/statistics/dl/loan/yaku/ | Medium | financing-rate proxy, macro time trend | national/monthly | use transaction-month/quarter or lagged values |
| 8 | IPSS regional population projections 2023 | https://www.ipss.go.jp/pp-shicyoson/j/shicyoson23/t-page.asp | Low-medium for historical tests, useful for future demos | projected population/demand | municipality future projections | can leak future assumptions into historical backtests |
| 9 | MLIT land-use 3rd mesh L03-a | https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-a-2021.html | Low-medium | 1km land-use composition/context | mesh/polygon | vintage mismatch for older transactions |

## Recommended first integrations

1. **L01 + L02 lagged nearest official land-price features** — strongest direct local price anchor.
2. **S12/N02 station features** — likely improves accessibility and demand modelling beyond current average-time-to-station field.

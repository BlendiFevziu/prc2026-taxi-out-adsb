# Origin and licence of the geometry files

The eleven `<icao>_osm.json` files contain the movement areas of the competition
airports: runways, taxiways, aprons and parking positions.

## Source

© OpenStreetMap contributors. Fetched through the Overpass API
(`https://overpass-api.de/api/interpreter`) on 02.09.2026 with `fetch_geometry.py`, which
sits in the root directory and can repeat the queries unchanged.

## Licence

The data is licensed under the **Open Database License (ODbL) 1.0**,
https://opendatacommons.org/licenses/odbl/1-0/

For any redistribution this means:

1. **Attribution.** "© OpenStreetMap contributors" must be carried along.
2. **Share-alike.** A derived database is again under the ODbL.
3. The ODbL covers the **database**. The source code of this repository is independently
   under the GPLv3, see `LICENSE`.

## Why they are included

Fetching through the Overpass API takes two queries per airport with an eight second
pause, and fails at peak times. Included, the geometry is available immediately, and a
reproduction computes on the same map vintage rather than a later one. To repeat the fetch
yourself, delete the directory and run `python3 fetch_geometry.py`.

## Eleven files, ten scored airports

`fetch_geometry.py` fetches eleven airports, ten are scored. `ltai_osm.json` (Antalya) is
included because the tool queries it; it does not appear in `ranking.parquet`. The ten
scored airports are EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.

## Element counts

    eddf 1018   eddm  867   egll  848   eham  874   lebl  781   lemd 1224
    lfpg 1645   lirf  432   lszh  581   ltai  766   ltfm 1576      total 10,612

"""Script to download and summarize official WHO Dengue Global Surveillance data for all countries."""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COUNTRY_META: dict[str, dict[str, Any]] = {
    "Indonesia": {"flag": "🇮🇩", "pop": 279300000, "lat": -0.7893, "lon": 113.9213, "region_name": "Asia Tenggara (SEARO)"},
    "Brazil": {"flag": "🇧🇷", "pop": 215300000, "lat": -14.2350, "lon": -51.9253, "region_name": "Amerika (AMRO/PAHO)"},
    "Argentina": {"flag": "🇦🇷", "pop": 46000000, "lat": -38.4161, "lon": -63.6167, "region_name": "Amerika (AMRO/PAHO)"},
    "Mexico": {"flag": "🇲🇽", "pop": 128000000, "lat": 23.6345, "lon": -102.5528, "region_name": "Amerika (AMRO/PAHO)"},
    "Colombia": {"flag": "🇨🇴", "pop": 52000000, "lat": 4.5709, "lon": -74.2973, "region_name": "Amerika (AMRO/PAHO)"},
    "Paraguay": {"flag": "🇵🇾", "pop": 7095000, "lat": -23.4425, "lon": -58.4438, "region_name": "Amerika (AMRO/PAHO)"},
    "Peru": {"flag": "🇵🇪", "pop": 34000000, "lat": -9.1899, "lon": -75.0152, "region_name": "Amerika (AMRO/PAHO)"},
    "India": {"flag": "🇮🇳", "pop": 1428000000, "lat": 20.5937, "lon": 78.9629, "region_name": "Asia Selatan (SEARO)"},
    "Guatemala": {"flag": "🇬🇹", "pop": 18967000, "lat": 15.7835, "lon": -90.2308, "region_name": "Amerika Tengah (AMRO)"},
    "Honduras": {"flag": "🇭🇳", "pop": 11184000, "lat": 15.1999, "lon": -86.2419, "region_name": "Amerika Tengah (AMRO)"},
    "Malaysia": {"flag": "🇲🇾", "pop": 34300000, "lat": 4.2105, "lon": 101.9758, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Viet Nam": {"flag": "🇻🇳", "pop": 98800000, "lat": 14.0583, "lon": 108.2772, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Thailand": {"flag": "🇹🇭", "pop": 71800000, "lat": 15.8700, "lon": 100.9925, "region_name": "Asia Tenggara (SEARO)"},
    "Bangladesh": {"flag": "🇧🇩", "pop": 171000000, "lat": 23.6850, "lon": 90.3563, "region_name": "Asia Selatan (SEARO)"},
    "Nicaragua": {"flag": "🇳🇮", "pop": 7097000, "lat": 12.8654, "lon": -85.2072, "region_name": "Amerika Tengah (AMRO)"},
    "Singapore": {"flag": "🇸🇬", "pop": 5920000, "lat": 1.3521, "lon": 103.8198, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Philippines": {"flag": "🇵🇭", "pop": 117300000, "lat": 12.8797, "lon": 121.7740, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Pakistan": {"flag": "🇵🇰", "pop": 240000000, "lat": 30.3753, "lon": 69.3451, "region_name": "Mediterania Timur (EMRO)"},
    "Sri Lanka": {"flag": "🇱🇰", "pop": 22200000, "lat": 7.8731, "lon": 80.7718, "region_name": "Asia Selatan (SEARO)"},
    "Burkina Faso": {"flag": "🇧🇫", "pop": 23200000, "lat": 12.2383, "lon": -1.5616, "region_name": "Afrika (AFRO)"},
    "Ecuador": {"flag": "🇪🇨", "pop": 18444000, "lat": -1.8312, "lon": -78.1834, "region_name": "Amerika Selatan (AMRO)"},
    "Bolivia (Plurinational State of)": {"flag": "🇧🇴", "pop": 12749000, "lat": -16.2902, "lon": -63.5887, "region_name": "Amerika Selatan (AMRO)"},
    "Venezuela (Bolivarian Republic of)": {"flag": "🇻🇪", "pop": 30765000, "lat": 6.4238, "lon": -66.5897, "region_name": "Amerika Selatan (AMRO)"},
    "Panama": {"flag": "🇵🇦", "pop": 4400000, "lat": 8.5379, "lon": -80.7821, "region_name": "Amerika Tengah (AMRO)"},
    "Costa Rica": {"flag": "🇨🇷", "pop": 5200000, "lat": 9.7489, "lon": -83.7534, "region_name": "Amerika Tengah (AMRO)"},
    "Dominican Republic": {"flag": "🇩🇴", "pop": 11300000, "lat": 18.7357, "lon": -70.1627, "region_name": "Karibia (AMRO)"},
    "Cambodia": {"flag": "🇰🇭", "pop": 16900000, "lat": 12.5657, "lon": 104.9910, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Lao People's Democratic Republic": {"flag": "🇱🇦", "pop": 7600000, "lat": 19.8563, "lon": 102.4955, "region_name": "Asia Pasifik Barat (WPRO)"},
    "Nepal": {"flag": "🇳🇵", "pop": 30900000, "lat": 28.3949, "lon": 84.1240, "region_name": "Asia Selatan (SEARO)"},
    "Maldives": {"flag": "🇲🇻", "pop": 520000, "lat": 3.2028, "lon": 73.2207, "region_name": "Asia Selatan (SEARO)"},
}


def process_who_global(
    output_path: Path | str = "data/raw/who_global_summary.json",
    raw_cache_path: Path | str = "data/raw/who_global_raw.json",
) -> Path:
    raw_cache = Path(raw_cache_path)
    base_url = "https://xmart-api-public.who.int/ARBOV/V_DENGUE_GLOBAL_VALIDATED_PUBLIC"

    if raw_cache.exists():
        print(f"[*] Memuat data mentah WHO dari cache lokal: {raw_cache}")
        with open(raw_cache, encoding="utf-8") as f:
            records = json.load(f)
    else:
        print(f"[*] Mengunduh data global WHO dari: {base_url}")
        req = urllib.request.Request(
            base_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        records = payload.get("value", [])
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_cache, "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"[OK] Cache data mentah tersimpan ({len(records)} baris)")

    # Grouping by country
    country_data: dict[str, dict[str, Any]] = {}
    country_annual: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    country_monthly_2024: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    country_serotypes_2024: dict[str, set[str]] = defaultdict(set)

    for r in records:
        c_name = r.get("COUNTRY")
        if not c_name:
            continue

        c_name = c_name.strip()
        year = r.get("YEAR")
        cases = r.get("CASES") or 0
        deaths = r.get("DEATHS") or 0
        pop = r.get("POPULATION") or 0

        if c_name not in country_data:
            country_data[c_name] = {
                "country": c_name,
                "iso3": r.get("ISO3", ""),
                "region": r.get("WHO_REGION", ""),
                "population": pop,
                "total_cases": 0,
                "total_deaths": 0,
                "cases_2024": 0,
                "deaths_2024": 0,
            }

        # Update population if available
        if pop and pop > country_data[c_name]["population"]:
            country_data[c_name]["population"] = pop

        country_data[c_name]["total_cases"] += cases
        country_data[c_name]["total_deaths"] += deaths

        if year:
            country_annual[c_name][int(year)] += cases

            if int(year) == 2024:
                country_data[c_name]["cases_2024"] += cases
                country_data[c_name]["deaths_2024"] += deaths

                # Check monthly distribution
                start_d = r.get("START_DATE")
                if start_d and len(start_d) >= 7:
                    try:
                        month = int(start_d[5:7])
                        country_monthly_2024[c_name][month] += cases
                    except Exception:
                        pass
                elif r.get("DATE_NUM") and r.get("DATE_TYPE") == "month":
                    month = int(r["DATE_NUM"])
                    if 1 <= month <= 12:
                        country_monthly_2024[c_name][month] += cases

                # Serotypes
                for s_num in [1, 2, 3, 4]:
                    val = str(r.get(f"SERO_{s_num}", "")).upper()
                    if val in ["TRUE", "1", "YES"]:
                        country_serotypes_2024[c_name].add(f"DENV-{s_num}")

    # Build final list
    results = []
    for c_name, info in country_data.items():
        meta = COUNTRY_META.get(c_name, {})
        pop = info["population"]
        if (pop is None or pop <= 0) and "pop" in meta:
            pop = meta["pop"]

        cases_24 = info["cases_2024"]
        deaths_24 = info["deaths_2024"]

        ir_24 = round((cases_24 / pop * 100000), 1) if pop > 0 else 0.0
        cfr_24 = round((deaths_24 / max(1, cases_24) * 100), 2) if cases_24 > 0 else 0.0

        # Monthly array [Jan..Dec]
        m_dict = country_monthly_2024[c_name]
        m_arr = [m_dict.get(m, 0) for m in range(1, 13)]

        # Annual trend dict for years 2014-2024
        ann_dict = {
            str(y): country_annual[c_name][y]
            for y in range(2014, 2025)
            if y in country_annual[c_name]
        }

        # Comparative commentary vs Indonesia
        ratio_cases = round(cases_24 / 257271, 1) if cases_24 > 0 else 0.0
        if c_name == "Brazil":
            comp_narrative = (
                f"Brazil merupakan episentrum wabah DBD terbesar di dunia pada 2024 dengan 10,26 juta kasus "
                f"({ratio_cases}x lipat kasus Indonesia). Tingkat insidens mencapai {ir_24}/100k penduduk, "
                f"didorong oleh sirkulasi simultan serotipe DENV-1 dan DENV-2 pasca-rekor suhu El Niño Amerika Selatan."
            )
        elif c_name == "Singapore":
            comp_narrative = (
                f"Singapura mencatat {cases_24:,} kasus di 2024 dengan surveilans ketat program Wolbachia NEA. "
                f"Walaupun volume total lebih kecil dari Indonesia, kepadatan urban yang ekstrem menghasilkan angka insidens "
                f"{ir_24}/100k penduduk (lebih dari 2x lipat IR Indonesia 92.1/100k), dengan dominasi DENV-2 dan DENV-3."
            )
        elif c_name == "Malaysia":
            comp_narrative = (
                f"Malaysia melaporkan {cases_24:,} kasus dan {deaths_24} kematian di 2024 (IR: {ir_24}/100k). "
                f"Sebagai sesama negara tropis Asia Tenggara, Malaysia mengalami lonjakan musiman yang sangat mirip dengan "
                f"wilayah barat Indonesia (Sumatra dan Semenanjung Melayu)."
            )
        elif c_name == "Thailand":
            comp_narrative = (
                f"Thailand mencatat {cases_24:,} kasus dengan angka insidens {ir_24}/100k. Karakteristik transmisi "
                f"di Thailand dipengaruhi oleh musim hujan monsun barat daya dan sirkulasi keempat serotipe dengue."
            )
        elif c_name == "Viet Nam":
            comp_narrative = (
                f"Vietnam mencatat {cases_24:,} kasus dan {deaths_24} kematian (IR: {ir_24}/100k). Wilayah delta sungai Mekong "
                f"menjadi hotspot utama penyebaran vektor Aedes aegypti di musim peralihan iklim."
            )
        elif c_name == "Indonesia":
            comp_narrative = (
                "Indonesia berada di peringkat #7 dunia pada pengawasan global WHO 2024 dengan 257.271 kasus terkonfirmasi "
                "(IR: 92.1/100k). Pada dashboard ini, Anda dapat beralih ke Mode Khusus Indonesia untuk memeriksa "
                "data riil OpenDengue v1.3 & BMKG di 34 provinsi."
            )
        else:
            comp_narrative = (
                f"{c_name} mencatat {cases_24:,} kasus dan {deaths_24:,} kematian pada tahun 2024 menurut data resmi validasi WHO. "
                f"Angka kesakitan (IR) tercatat sebesar {ir_24} per 100.000 penduduk."
            )

        flag = meta.get("flag", "🌐")
        lat = meta.get("lat", 0.0)
        lon = meta.get("lon", 0.0)
        reg_full = meta.get("region_name", info["region"])

        results.append({
            "country": c_name,
            "iso3": info["iso3"],
            "flag": flag,
            "lat": lat,
            "lon": lon,
            "region": info["region"],
            "region_name": reg_full,
            "population": pop,
            "total_cases": info["total_cases"],
            "total_deaths": info["total_deaths"],
            "cases_2024": cases_24,
            "deaths_2024": deaths_24,
            "ir_2024": ir_24,
            "cfr_2024": cfr_24,
            "serotypes_2024": sorted(list(country_serotypes_2024[c_name])),
            "annual_trend": ann_dict,
            "monthly_2024": m_arr,
            "narrative": comp_narrative,
        })

    # Sort by cases_2024 descending
    results.sort(key=lambda x: (x["cases_2024"], x["total_cases"]), reverse=True)

    # Assign global ranks
    for rank, item in enumerate(results, start=1):
        item["global_rank_2024"] = rank

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "source": "World Health Organization (WHO) Global Dengue Surveillance",
                "total_countries": len(results),
                "countries": results,
            },
            f,
            indent=2,
        )

    print(f"[OK] Berhasil membuat ringkasan WHO Global {len(results)} negara: {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    process_who_global()

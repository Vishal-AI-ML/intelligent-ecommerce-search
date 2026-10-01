"""Deterministic generator for the synthetic seed catalog (dataset `synthetic-seed`, v1).

Every record is SYNTHETIC. Real brand names are used only as labels; series/model names, specs,
prices, ratings and sellers are invented. Nothing is scraped, downloaded or derived from any
marketplace. The output is byte-for-byte reproducible: no randomness, no clock, no network;
pseudo-variation comes from SHA-256 of the product id.

Usage:
    uv run python scripts/generate_seed_catalog.py           # (re)write the seed files
    uv run python scripts/generate_seed_catalog.py --check   # verify committed files match

This script is intentionally self-contained (it does not import the application) so that
application changes cannot silently alter the committed seed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENERATOR_VERSION = "1"
DATASET_ID = "synthetic-seed"
DATASET_VERSION = "1"
TRANSFORM_VERSION = "1"
TAXONOMY_VERSION = "1"
AUTHORED_ON = "2026-09-30"

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed" / "catalog_seed_v1.jsonl"
PROVENANCE_PATH = ROOT / "data" / "seed" / "catalog_seed_v1.provenance.json"


def _h(product_id: str) -> int:
    return int(hashlib.sha256(product_id.encode("utf-8")).hexdigest()[:8], 16)


def _storage_label(gb: int) -> str:
    return f"{gb // 1024}TB" if gb % 1024 == 0 and gb >= 1024 else f"{gb}GB"


def _price(base: int, mult: int, unit: int, h: int) -> int:
    return ((base * mult // 100) // unit + (h % 5)) * unit + unit - 1


def _core(pid, h, title, description, category, subcategory, brand, price):
    if h % 17 == 0:
        rating, reviews = None, 0
    else:
        rating, reviews = (34 + (h >> 8) % 14) / 10, 12 + (h >> 12) % 4500
    if h % 53 == 0:
        availability = "discontinued"
    elif h % 11 == 0:
        availability = "out_of_stock"
    else:
        availability = "in_stock"
    return {
        "product_id": pid,
        "seller_id": None if h % 13 == 0 else f"synthetic-seller-{h % 8 + 1:02d}",
        "title": title,
        "description": description,
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "price": price,
        "currency": "INR",
        "rating": rating,
        "review_count": reviews,
        "availability": availability,
        "source_type": "synthetic",
        "is_synthetic": True,
    }


# ---- laptops ----------------------------------------------------------------------------
LAPTOP_BRANDS = [
    ("HP", 100, ("Aurelia", "Brisk", "Corvid")),
    ("Dell", 105, ("Meridian", "Vantor", "Solace")),
    ("Lenovo", 98, ("Lumen", "Tessera", "Arcadia")),
    ("Asus", 102, ("Zenith", "Kestrel", "Nimbus")),
    ("Acer", 95, ("Orbit", "Pinnacle", "Halcyon")),
    ("Apple", 160, ("Aster", "Borealis", "Cirrus")),
    ("MSI", 110, ("Cobalt", "Ember", "Falcon")),
]
# tier, subcategory, processor, gpu, ram, storage_gb, type, interface, screen, weight, base
LAPTOP_CONFIGS = [
    ("entry", "everyday", "Intel Core i3", None, 8, 256, "SSD", "SATA", 14.0, 1.5, 32000),
    ("entry", "everyday", "Intel Core i3", None, 8, 512, "SSD", "NVME", 15.6, 1.8, 36000),
    ("mid", "business", "Intel Core i5", None, 8, 512, "SSD", "NVME", 15.6, 1.7, 48000),
    ("mid", "business", "Intel Core i5", None, 16, 512, "SSD", "NVME", 14.0, 1.4, 62000),
    ("entry", "everyday", "AMD Ryzen 5", None, 8, 256, "SSD", "NVME", 14.0, 1.5, 42000),
    ("mid", "ultrabook", "AMD Ryzen 5", None, 16, 512, "SSD", "NVME", 14.0, 1.3, 55000),
    (
        "high",
        "ultrabook",
        "AMD Ryzen 7",
        "Integrated Radeon Graphics",
        16,
        1024,
        "SSD",
        "NVME",
        14.0,
        1.2,
        78000,
    ),
    (
        "high",
        "performance",
        "Intel Core i7",
        "NVIDIA GeForce RTX 4050",
        16,
        1024,
        "SSD",
        "NVME",
        16.0,
        2.0,
        98000,
    ),
    (
        "premium",
        "performance",
        "Intel Core i7",
        "NVIDIA GeForce RTX 4060",
        32,
        1024,
        "SSD",
        "NVME",
        16.0,
        2.3,
        145000,
    ),
    ("entry", "everyday", "Intel Core i5", None, 8, 1024, "HDD", "SATA", 15.6, 2.1, 34000),
    ("mid", "business", "Intel Core i5", None, 12, 512, "SSD", "SATA", 15.6, 1.9, 44000),
    ("entry", "everyday", "Intel Core i3", None, 4, 1024, "HDD", "SATA", 15.6, 2.2, 26000),
]
LAPTOP_TAGS = {
    "entry": "A budget-friendly choice for everyday browsing, study and office work.",
    "mid": "A balanced choice for productivity, coding and multitasking.",
    "high": "A fast, portable choice for demanding work and content creation.",
    "premium": "A high-performance choice for gaming and heavy creative workloads.",
}


def gen_laptops(count=80):
    for i in range(count):
        b, k = i % 7, i // 7
        brand, mult, series_names = LAPTOP_BRANDS[b]
        series = series_names[k % 3]
        cfg = LAPTOP_CONFIGS[(5 * k + 3 * b) % 12]
        tier, sub, cpu, gpu, ram, storage, stype, iface, screen, weight, base = cfg
        os_name = "Windows 11"
        if brand == "Apple":
            cpu, gpu, os_name = "Apple M-series", None, "macOS"
            if stype == "HDD":
                stype, iface, storage, ram = "SSD", "NVME", 256, 8
        elif k % 5 == 4:
            os_name = "Linux"
        pid = f"SYN-LAP-{i + 1:04d}"
        h = _h(pid)
        label = _storage_label(storage)
        title = (
            f"{brand} {series} {screen}-inch Laptop - {cpu}, {ram}GB RAM, {label} {stype} "
            f"Model LP{101 + i}"
        )
        gpu_part = f" and {gpu} graphics" if gpu else ""
        desc = (
            f"{series} series laptop with {cpu}, {ram} GB RAM and {label} {stype} storage"
            f"{gpu_part}. {LAPTOP_TAGS[tier]}"
        )
        rec = _core(pid, h, title, desc, "laptop", sub, brand, _price(base, mult, 1000, h))
        rec.update(
            ram_gb=ram,
            storage_gb=storage,
            storage_type=stype,
            storage_interface=iface,
            processor=cpu,
            gpu=gpu,
            screen_size_inches=screen,
            operating_system=os_name,
            weight_kg=weight,
        )
        yield i, rec


# ---- phones -----------------------------------------------------------------------------
PHONE_BRANDS = [
    ("Samsung", 100, ("Vela", "Novara", "Pulse")),
    ("Apple", 170, ("Aster", "Borealis", "Cirrus")),
    ("Xiaomi", 90, ("Lyra", "Mirage", "Quanta")),
    ("OnePlus", 100, ("Ridge", "Summit", "Vertex")),
    ("Realme", 88, ("Zephyr", "Tempo", "Glint")),
    ("Motorola", 92, ("Harbor", "Beacon", "Anchor")),
]
# tier, ram, storage, camera, battery, screen, base
PHONE_CONFIGS = [
    ("entry", 4, 64, "13 MP dual camera", 5000, 6.5, 9000),
    ("entry", 4, 128, "48 MP dual camera", 5000, 6.6, 11000),
    ("entry", 6, 128, "50 MP dual camera", 5000, 6.6, 14000),
    ("mid", 8, 128, "50 MP triple camera", 5000, 6.7, 19000),
    ("mid", 8, 256, "64 MP triple camera", 5000, 6.7, 24000),
    ("mid", 12, 256, "108 MP triple camera", 5000, 6.8, 32000),
    ("upper", 8, 256, "50 MP dual camera with OIS", 4500, 6.4, 38000),
    ("upper", 12, 512, "50 MP triple camera", 5000, 6.8, 55000),
    ("flagship", 16, 512, "200 MP quad camera", 5200, 6.9, 79000),
    ("flagship", 12, 1024, "48 MP triple camera", 4400, 6.7, 99000),
]
PHONE_TAGS = {
    "entry": "An affordable smartphone for calls, messaging and everyday apps.",
    "mid": "A capable smartphone with a good camera and all-day battery life.",
    "upper": "A fast smartphone for photography and smooth multitasking.",
    "flagship": "A top-end smartphone with a premium camera system and display.",
}


def gen_phones(count=60):
    for i in range(count):
        b, k = i % 6, i // 6
        brand, mult, series_names = PHONE_BRANDS[b]
        series = series_names[k % 3]
        tier, ram, storage, camera, battery, screen, base = PHONE_CONFIGS[(3 * k + b) % 10]
        pid = f"SYN-PHN-{i + 1:04d}"
        h = _h(pid)
        label = _storage_label(storage)
        title = f"{brand} {series} ({ram}GB RAM, {label} Storage) Model PH{101 + i}"
        desc = (
            f"{series} smartphone with {ram} GB RAM, {label} storage, {camera} and a {battery} mAh "
            f"battery. {PHONE_TAGS[tier]}"
        )
        rec = _core(pid, h, title, desc, "phone", "smartphone", brand, _price(base, mult, 1000, h))
        rec.update(
            ram_gb=ram,
            storage_gb=storage,
            camera=camera,
            battery_mah=battery,
            screen_size_inches=screen,
            operating_system="iOS" if brand == "Apple" else "Android",
        )
        yield i, rec


# ---- shoes ------------------------------------------------------------------------------
SHOE_BRANDS = [
    ("Nike", 120, ("Strato", "Cadence", "Vector")),
    ("Adidas", 115, ("Kinetic", "Ravine", "Tempo")),
    ("Puma", 100, ("Drift", "Ember", "Rally")),
    ("Reebok", 90, ("Anchor", "Summit", "Flux")),
    ("Skechers", 105, ("Glide", "Stride", "Cloud")),
]
# subcategory, title word, material, base
SHOE_CONFIGS = [
    ("running", "Running", "Mesh", 3500),
    ("running", "Running", "Knit", 5200),
    ("casual", "Casual", "Canvas", 2400),
    ("casual", "Casual", "Leather", 4800),
    ("sports", "Training", "Synthetic", 3900),
    ("sports", "Sports", "Mesh", 4400),
    ("formal", "Formal", "Leather", 5500),
    ("casual", "Sneaker", "Suede", 4100),
    ("running", "Trail Running", "Synthetic", 4700),
    ("sports", "Court", "Leather", 5100),
]
SHOE_COLORS = [
    "Black",
    "White",
    "Grey",
    "Navy",
    "Red",
    "Blue",
    "Green",
    "Brown",
    "Beige",
    "Orange",
]
GENDERS = ["men", "women", "unisex"]
SIZE_SYSTEMS = ["UK", "UK", "US", "EU"]
SIZES = {
    "UK": [6, 7, 8, 9, 10, 11],
    "US": [7, 8, 9, 10, 11, 12],
    "EU": [39, 40, 41, 42, 43, 44],
}


def gen_shoes(count=50):
    for i in range(count):
        b, k = i % 5, i // 5
        brand, mult, series_names = SHOE_BRANDS[b]
        series = series_names[k % 3]
        sub, word, material, base = SHOE_CONFIGS[(3 * k + b) % 10]
        color = SHOE_COLORS[(3 * k + 2 * b) % 10]
        gender = GENDERS[(k + b) % 3]
        system = SIZE_SYSTEMS[i % 4]
        size = float(SIZES[system][i % 6])
        if system != "EU" and i % 7 == 0:
            size += 0.5
        pid = f"SYN-SHO-{i + 1:04d}"
        h = _h(pid)
        title = (
            f"{brand} {series} {word} Shoes for {gender.capitalize()} - {color} Model SH{101 + i}"
        )
        desc = (
            f"{series} {word.lower()} shoes in {color.lower()} with a {material.lower()} upper, "
            f"size {size:g} ({system}). Designed for {sub} use."
        )
        rec = _core(pid, h, title, desc, "shoes", sub, brand, _price(base, mult, 100, h))
        rec.update(size=size, size_system=system, color=color, material=material, gender=gender)
        yield i, rec


# ---- headphones -------------------------------------------------------------------------
HEADPHONE_BRANDS = [
    ("Sony", 115, ("Aura", "Cadenza", "Lyric")),
    ("Bose", 150, ("Serene", "Haven", "Calm")),
    ("JBL", 100, ("Rhythm", "Beat", "Wave")),
    ("boAt", 70, ("Rover", "Nomad", "Drift")),
    ("Sennheiser", 130, ("Clarion", "Timbre", "Resonance")),
]
# subcategory, wireless, anc, battery_hours, connectivity, base
HEADPHONE_CONFIGS = [
    ("in-ear", True, False, 20, "bluetooth", 1500),
    ("in-ear", True, False, 30, "bluetooth", 2500),
    ("in-ear", True, True, 24, "bluetooth", 5500),
    ("in-ear", True, True, 36, "bluetooth", 12000),
    ("on-ear", True, False, 40, "bluetooth", 4500),
    ("over-ear", True, True, 30, "bluetooth", 14000),
    ("over-ear", True, True, 40, "bluetooth", 22000),
    ("over-ear", False, False, None, "wired", 5000),
    ("in-ear", False, False, None, "wired", 900),
    ("over-ear", True, False, 35, "wireless_2_4ghz", 7000),
]
HEADPHONE_TAGS = {
    "in-ear": "Compact and easy to carry for commuting and workouts.",
    "on-ear": "Lightweight on-ear design for everyday listening.",
    "over-ear": "Comfortable over-ear design for long listening sessions and travel.",
}


def gen_headphones(count=50):
    for i in range(count):
        b, k = i % 5, i // 5
        brand, mult, series_names = HEADPHONE_BRANDS[b]
        series = series_names[k % 3]
        sub, wireless, anc, hours, conn, base = HEADPHONE_CONFIGS[(3 * k + b) % 10]
        pid = f"SYN-HDP-{i + 1:04d}"
        h = _h(pid)
        style = "Wireless" if wireless else "Wired"
        title = f"{brand} {series} {sub.capitalize()} {style} Headphones"
        if anc:
            title += " with Active Noise Cancellation"
        title += f" Model HD{101 + i}"
        desc = f"{series} {sub} {style.lower()} headphones"
        if anc:
            desc += " with active noise cancellation"
        if hours:
            desc += f" and up to {hours} hours of battery life"
        desc += f". {HEADPHONE_TAGS[sub]}"
        rec = _core(pid, h, title, desc, "headphones", sub, brand, _price(base, mult, 100, h))
        rec.update(wireless=wireless, anc=anc, battery_life_hours=hours, connectivity=conn)
        yield i, rec


# ---- raw-format variants (deterministic, ~10% of records) -------------------------------


def _variant_price(rec, i):
    rec["price"] = f"{'Rs ' if i % 2 else '₹'}{rec['price']:,}"


def apply_variant(rec, i):
    """Seller-style formatting variants that normalization must undo. Raw stays distinct."""
    v = (i // 10) % 5
    if v == 0:
        rec["brand"] = f" {rec['brand'].lower()} "
        for key in ("storage_type", "storage_interface"):
            if rec.get(key):
                rec[key] = rec[key].lower()
    elif v == 1 and rec.get("ram_gb") is not None and rec.get("storage_gb") is not None:
        rec["ram_gb"] = f"{rec['ram_gb']} GB"
        rec["storage_gb"] = f"{rec['storage_gb']}GB"
    elif v == 2 and rec.get("storage_gb") and rec["storage_gb"] % 1024 == 0:
        rec["storage_gb"] = f"{rec['storage_gb'] // 1024}TB"
    elif v == 4:
        rec["title"] = "  " + rec["title"].replace(" ", "  ", 1) + " "
    else:
        _variant_price(rec, i)


def generate_records():
    for gen in (gen_laptops, gen_phones, gen_shoes, gen_headphones):
        for i, rec in gen():
            if i % 10 == 3:
                apply_variant(rec, i)
            yield rec


def render() -> tuple[bytes, dict]:
    lines = [
        json.dumps(rec, ensure_ascii=False, separators=(",", ":")) for rec in generate_records()
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    by_category: dict[str, int] = {}
    for line in lines:
        cat = json.loads(line)["category"]
        by_category[cat] = by_category.get(cat, 0) + 1
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "source_description": (
            "Project-authored deterministic synthetic catalog produced by "
            "scripts/generate_seed_catalog.py (generator version "
            f"{GENERATOR_VERSION}). Every record is synthetic. There is no third-party source "
            "and no marketplace source."
        ),
        "rights_note": (
            "Authored by this project for development and testing. Not scraped, downloaded or "
            "derived from Amazon, Flipkart or any other marketplace or dataset. Real brand names "
            "are used only as labels; series/model names, specifications, prices, ratings and "
            "sellers are invented, and no affiliation with any brand or marketplace is implied. "
            "No formal licence identifier has been assigned."
        ),
        "license_identifier": None,
        "authored_on": AUTHORED_ON,
        "generator": f"scripts/generate_seed_catalog.py@{GENERATOR_VERSION}",
        "is_synthetic": True,
        "transform_version": TRANSFORM_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "notes": (
            "Templated data is easier than real listings, so retrieval metrics computed on it "
            "will be optimistic. Prices are invented and are not market prices. About 10% of "
            "records use seller-style formatting variants (units, casing, whitespace, currency "
            "markers) to exercise normalization. Reproduce with: uv run python "
            "scripts/generate_seed_catalog.py --check. Counts by category: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items()))
            + "."
        ),
        "record_count": len(lines),
        "checksum_sha256": hashlib.sha256(data).hexdigest(),
    }
    return data, provenance


def provenance_bytes(provenance: dict) -> bytes:
    return (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify committed files, write nothing"
    )
    args = parser.parse_args(argv)
    data, provenance = render()
    prov_bytes = provenance_bytes(provenance)
    if args.check:
        problems = []
        if not SEED_PATH.exists() or SEED_PATH.read_bytes() != data:
            problems.append(f"{SEED_PATH} differs from regenerated output")
        if not PROVENANCE_PATH.exists() or PROVENANCE_PATH.read_bytes() != prov_bytes:
            problems.append(f"{PROVENANCE_PATH} differs from regenerated output")
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(
            "OK: seed and provenance regenerate byte-for-byte "
            f"(sha256 {provenance['checksum_sha256']})"
        )
        return 0
    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_bytes(data)
    PROVENANCE_PATH.write_bytes(prov_bytes)
    print(f"wrote {provenance['record_count']} records, sha256 {provenance['checksum_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

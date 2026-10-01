"""Migration and constraint tests on throwaway databases (never the development database)."""

import re

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, InternalError, ProgrammingError
from sqlalchemy.schema import CreateTable

import ecommerce_search.models  # noqa: F401
from ecommerce_search.catalog import taxonomy as tx
from ecommerce_search.db.base import Base

pytestmark = pytest.mark.integration

SHA = "a" * 64


def tables(engine):
    with engine.connect() as conn:
        return set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            ).scalars()
        )


# ---- migrations -------------------------------------------------------------------------


def test_upgrade_downgrade_round_trip_preserves_pgvector(settings, scratch_database, migrator):
    engine = create_engine(settings.database_url(database=scratch_database))
    try:
        migrator.upgrade(scratch_database)
        assert "products" in tables(engine)
        migrator.downgrade(scratch_database, "0001")
        assert tables(engine) == {"alembic_version"}
        with engine.connect() as conn:  # 0001's extension is untouched by the 0002 downgrade
            assert (
                conn.execute(
                    text("SELECT count(*) FROM pg_extension WHERE extname='vector'")
                ).scalar_one()
                == 1
            )
            assert (
                conn.execute(
                    text("SELECT proname FROM pg_proc WHERE proname='catalog_reviews_append_only'")
                ).first()
                is None
            )
        migrator.upgrade(scratch_database)
        assert "catalog_reviews" in tables(engine)
        migrator.downgrade(scratch_database, "base")
        assert tables(engine) == {"alembic_version"}
        migrator.upgrade(scratch_database)
        migrator.upgrade(scratch_database)  # idempotent
        assert "products" in tables(engine)
    finally:
        engine.dispose()


def test_migrated_schema_matches_model_metadata(migrated_engine):
    with migrated_engine.connect() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "compare_type": True,
                "compare_server_default": False,
                "include_object": lambda obj, name, type_, reflected, compare_to: (
                    not (type_ == "table" and name == "alembic_version")
                ),
            },
        )
        assert compare_metadata(context, Base.metadata) == []


def test_constraint_and_index_names_match_models_and_limits(migrated_engine):
    expected = set()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        expected |= set(re.findall(r"CONSTRAINT (\w+)", ddl))
    with migrated_engine.connect() as conn:
        actual = set(
            conn.execute(
                text(
                    "SELECT c.conname FROM pg_constraint c JOIN pg_class t ON t.oid=c.conrelid "
                    "JOIN pg_namespace n ON n.oid=t.relnamespace "
                    "WHERE n.nspname='public' AND t.relname <> 'alembic_version'"
                )
            ).scalars()
        )
    assert actual == expected
    assert all(len(name) <= 63 for name in actual)


def test_only_constraint_backing_indexes_exist(migrated_engine):
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND tablename <> 'alembic_version'"
            )
        ).all()
        constraint_names = set(
            conn.execute(
                text(
                    "SELECT c.conname FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
                    "JOIN pg_namespace n ON n.oid = t.relnamespace "
                    "WHERE n.nspname = 'public' AND c.contype IN ('p', 'u')"
                )
            ).scalars()
        )
        column_types = set(
            conn.execute(
                text(
                    "SELECT DISTINCT data_type FROM information_schema.columns WHERE table_schema='public'"
                )
            ).scalars()
        )
    assert rows
    assert {name for name, _ in rows} == {
        n for n in constraint_names if not n.startswith("alembic")
    }
    assert all(re.search(r"USING btree", definition) for _, definition in rows)
    assert not column_types & {"tsvector", "USER-DEFINED", "jsonb", "json"}


def test_database_check_values_equal_application_taxonomies(migrated_engine):
    expected = {
        "ck_products_category_valid": tx.Category,
        "ck_products_availability_valid": tx.Availability,
        "ck_products_source_type_valid": tx.SourceType,
        "ck_laptop_specs_storage_type_valid": tx.StorageType,
        "ck_laptop_specs_storage_interface_valid": tx.StorageInterface,
        "ck_shoe_specs_size_system_valid": tx.SizeSystem,
        "ck_shoe_specs_gender_valid": tx.Gender,
        "ck_headphone_specs_connectivity_valid": tx.Connectivity,
        "ck_catalog_reviews_verdict_valid": tx.ReviewVerdict,
    }
    with migrated_engine.connect() as conn:
        for name, enum_cls in expected.items():
            definition = conn.execute(
                text("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = :n"),
                {"n": name},
            ).scalar_one()
            assert set(re.findall(r"'([^']+)'", definition)) == {m.value for m in enum_cls}, name


# ---- constraint behaviour ---------------------------------------------------------------


def add_dataset(conn):
    return conn.execute(
        text(
            "INSERT INTO catalog_datasets (dataset_id, dataset_version, source_description, rights_note, "
            "authored_on, generator, is_synthetic, transform_version, taxonomy_version, rules_version, "
            "checksum_sha256, record_count) VALUES ('d','1','s','r','2026-09-30','g',true,'1','1','1',:sha,1) "
            "RETURNING id"
        ),
        {"sha": SHA},
    ).scalar_one()


def add_product(conn, pid="P1", category="laptop", **over):
    dataset_pk = conn.execute(
        text("SELECT id FROM catalog_datasets LIMIT 1")
    ).scalar() or add_dataset(conn)
    raw_id = conn.execute(
        text(
            "INSERT INTO raw_catalog_records (dataset_pk, source_key, line_number, raw_line, raw_sha256) "
            "VALUES (:d, :k, 1, '{}', :sha) RETURNING id"
        ),
        {"d": dataset_pk, "k": pid, "sha": SHA},
    ).scalar_one()
    values = {
        "product_id": pid,
        "dataset_pk": dataset_pk,
        "raw_id": raw_id,
        "category": category,
        "price": 100,
        "currency": "INR",
        "rating": None,
        "is_synthetic": True,
        "source_type": "synthetic",
        "sha": SHA,
        "availability": "in_stock",
    } | over
    conn.execute(
        text(
            "INSERT INTO products (product_id, dataset_pk, raw_record_id, title, category, brand, price, "
            "currency, rating, availability, source_type, is_synthetic, content_sha256) VALUES "
            "(:product_id, :dataset_pk, :raw_id, 't', :category, 'b', :price, :currency, :rating, "
            ":availability, :source_type, :is_synthetic, :sha)"
        ),
        values,
    )


def fails(engine, name_fragment, action):
    with (
        pytest.raises((IntegrityError, ProgrammingError, InternalError)) as excinfo,
        engine.begin() as conn,
    ):
        action(conn)
    assert name_fragment in str(excinfo.value.orig)


def insert_laptop_spec(conn, pid="P1", category="laptop", stype="SSD", iface="NVME"):
    conn.execute(
        text(
            "INSERT INTO laptop_specs (product_id, category, storage_type, storage_interface) "
            "VALUES (:p, :c, :t, :i)"
        ),
        {"p": pid, "c": category, "t": stype, "i": iface},
    )


@pytest.mark.parametrize(
    ("stype", "iface"),
    [("SSD", "NVME"), ("SSD", "SATA"), ("HDD", "SATA"), ("SSD", None), (None, None)],
)
def test_valid_storage_combinations(migrated_engine, stype, iface):
    with migrated_engine.begin() as conn:
        add_product(conn)
        insert_laptop_spec(conn, stype=stype, iface=iface)


@pytest.mark.parametrize("stype", ["HDD", None])
def test_nvme_requires_ssd(migrated_engine, stype):
    def act(conn):
        add_product(conn)
        insert_laptop_spec(conn, stype=stype, iface="NVME")

    fails(migrated_engine, "ck_laptop_specs_nvme_requires_ssd", act)


def test_spec_attached_to_wrong_category_fails(migrated_engine):
    def laptop_spec_on_phone(conn):
        add_product(conn, category="phone")
        insert_laptop_spec(conn, category="laptop")

    fails(migrated_engine, "fk_laptop_specs_product_id_category_products", laptop_spec_on_phone)

    def laptop_spec_labelled_phone(conn):
        add_product(conn, category="phone")
        insert_laptop_spec(conn, category="phone")

    fails(migrated_engine, "ck_laptop_specs_category_is_laptop", laptop_spec_labelled_phone)

    def phone_spec_on_laptop(conn):
        add_product(conn, category="laptop")
        conn.execute(text("INSERT INTO phone_specs (product_id, category) VALUES ('P1','phone')"))

    fails(migrated_engine, "fk_phone_specs_product_id_category_products", phone_spec_on_laptop)


def test_shoes_cannot_have_ram_and_laptops_cannot_have_a_shoe_size():
    columns = {t: {c.name for c in Base.metadata.tables[t].columns} for t in Base.metadata.tables}
    assert "ram_gb" not in columns["shoe_specs"] and "size" not in columns["laptop_specs"]
    assert "storage_type" not in columns["phone_specs"] and "anc" not in columns["laptop_specs"]


@pytest.mark.parametrize(
    ("override", "constraint"),
    [
        ({"price": 0}, "ck_products_price_positive"),
        ({"price": -1}, "ck_products_price_positive"),
        ({"currency": "inr"}, "ck_products_currency_format"),
        ({"rating": 6}, "ck_products_rating_range"),
        ({"category": "toys"}, "ck_products_category_valid"),
        ({"availability": "gone"}, "ck_products_availability_valid"),
        ({"sha": "xyz"}, "ck_products_content_sha256_format"),
        ({"is_synthetic": False}, "ck_products_synthetic_flag_consistent"),
        ({"product_id": "bad id!"}, "ck_products_product_id_format"),
    ],
)
def test_product_constraints(migrated_engine, override, constraint):
    fails(migrated_engine, constraint, lambda conn: add_product(conn, **override))


def test_shoe_and_headphone_constraints(migrated_engine):
    def size_without_system(conn):
        add_product(conn, category="shoes")
        conn.execute(
            text("INSERT INTO shoe_specs (product_id, category, size) VALUES ('P1','shoes',9)")
        )

    fails(migrated_engine, "ck_shoe_specs_size_and_system_together", size_without_system)

    def wired_and_wireless(conn):
        add_product(conn, category="headphones")
        conn.execute(
            text(
                "INSERT INTO headphone_specs (product_id, category, wireless, connectivity) VALUES ('P1','headphones',true,'wired')"
            )
        )

    fails(migrated_engine, "ck_headphone_specs_wired_not_wireless", wired_and_wireless)

    def bluetooth_not_wireless(conn):
        add_product(conn, category="headphones")
        conn.execute(
            text(
                "INSERT INTO headphone_specs (product_id, category, wireless, connectivity) VALUES ('P1','headphones',false,'bluetooth')"
            )
        )

    fails(
        migrated_engine,
        "ck_headphone_specs_wireless_connectivity_consistent",
        bluetooth_not_wireless,
    )


def test_catalog_reviews_are_append_only(migrated_engine):
    with migrated_engine.begin() as conn:
        add_product(conn)
        dataset_pk = conn.execute(text("SELECT id FROM catalog_datasets")).scalar_one()
        conn.execute(
            text(
                "INSERT INTO catalog_reviews (review_batch_id, product_id, dataset_pk, "
                "product_content_sha256, verdict, reviewer, sample_manifest_sha256) "
                "VALUES ('b','P1',:d,:sha,'accept','TEST-ONLY-REVIEWER',:sha)"
            ),
            {"d": dataset_pk, "sha": SHA},
        )
    for statement in (
        "UPDATE catalog_reviews SET verdict='unsure'",
        "DELETE FROM catalog_reviews",
        "TRUNCATE catalog_reviews",
    ):
        with (
            pytest.raises((InternalError, ProgrammingError, IntegrityError)) as excinfo,
            migrated_engine.begin() as conn,
        ):
            conn.execute(text(statement))
        assert "append-only" in str(excinfo.value.orig)
    with migrated_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM catalog_reviews")).scalar_one() == 1


def test_review_constraints(migrated_engine):
    def bad(verdict="accept", reviewer="R"):
        def act(conn):
            add_product(conn)
            dataset_pk = conn.execute(text("SELECT id FROM catalog_datasets")).scalar_one()
            conn.execute(
                text(
                    "INSERT INTO catalog_reviews (review_batch_id, product_id, dataset_pk, "
                    "product_content_sha256, verdict, reviewer, sample_manifest_sha256) "
                    "VALUES ('b','P1',:d,:sha,:v,:r,:sha)"
                ),
                {"d": dataset_pk, "v": verdict, "r": reviewer, "sha": SHA},
            )

        return act

    fails(migrated_engine, "ck_catalog_reviews_verdict_valid", bad(verdict="approved"))
    fails(migrated_engine, "ck_catalog_reviews_reviewer_not_blank", bad(reviewer="  "))


def test_review_rows_require_a_valid_product_content_hash(migrated_engine):
    def insert(content_hash):
        def act(conn):
            add_product(conn)
            dataset_pk = conn.execute(text("SELECT id FROM catalog_datasets")).scalar_one()
            conn.execute(
                text(
                    "INSERT INTO catalog_reviews (review_batch_id, product_id, dataset_pk, "
                    "product_content_sha256, verdict, reviewer, sample_manifest_sha256) "
                    "VALUES ('b','P1',:d,:h,'accept','TEST-ONLY-REVIEWER',:sha)"
                ),
                {"d": dataset_pk, "h": content_hash, "sha": SHA},
            )

        return act

    fails(migrated_engine, "ck_catalog_reviews_product_content_sha256_format", insert("not-a-hash"))
    with migrated_engine.begin() as conn:
        insert(SHA)(conn)  # a well-formed hash is accepted


@pytest.mark.parametrize("version", ["0", "01", "v1", "1.0", "", "1000000000", "f"])
def test_dataset_version_must_be_a_canonical_positive_integer(migrated_engine, version):
    def act(conn):
        conn.execute(
            text(
                "INSERT INTO catalog_datasets (dataset_id, dataset_version, source_description, "
                "rights_note, authored_on, generator, is_synthetic, transform_version, "
                "taxonomy_version, rules_version, checksum_sha256, record_count) VALUES "
                "('d', :v, 's','r','2026-09-30','g',true,'1','1','1',:sha,1)"
            ),
            {"v": version, "sha": SHA},
        )

    fails(migrated_engine, "ck_catalog_datasets_dataset_version_numeric", act)


def _constraint_definitions(text_blob):
    """name -> normalized definition, using balanced parentheses after each CONSTRAINT."""
    found = {}
    for m in re.finditer(r"CONSTRAINT (\w+) ", text_blob):
        i, depth, seen = m.end(), 0, False
        j = i
        while j < len(text_blob):
            c = text_blob[j]
            if c == "(":
                depth, seen = depth + 1, True
            elif c == ")":
                depth -= 1
            elif (
                c in ",\n"
                and depth == 0
                and seen
                and not text_blob[j : j + 14].lstrip(", \n").startswith(("REFERENCES", "ON "))
            ):
                break
            j += 1
        found[m[1]] = re.sub(r"\s+", " ", text_blob[i:j]).strip()
    return found


def test_model_and_migration_constraint_definitions_are_identical(settings):
    import io

    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.attributes["database"] = "offline_sql_only"  # offline mode never connects
    config.output_buffer = io.StringIO()
    command.upgrade(config, "0001:0002", sql=True)
    migration = _constraint_definitions(config.output_buffer.getvalue())

    model = {}
    for table in Base.metadata.sorted_tables:
        model.update(
            _constraint_definitions(str(CreateTable(table).compile(dialect=postgresql.dialect())))
        )
    assert len(migration) == len(model) > 60
    assert migration == model

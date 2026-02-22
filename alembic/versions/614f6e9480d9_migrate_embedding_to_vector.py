"""migrate_embedding_to_vector

Revision ID: 614f6e9480d9
Revises: cfaffc97d402
Create Date: 2026-02-22

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = '614f6e9480d9'
down_revision: Union[str, Sequence[str], None] = '67743383a8f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.drop_column("faces", "embedding")
    op.add_column("faces", sa.Column("embedding", Vector(512), nullable=False))
    op.execute("""
        CREATE INDEX faces_embedding_hnsw_idx
        ON faces
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS faces_embedding_hnsw_idx")
    op.drop_column("faces", "embedding")
    op.add_column("faces", sa.Column("embedding", sa.ARRAY(sa.Float()), nullable=False))
    op.execute("DROP EXTENSION IF EXISTS vector")
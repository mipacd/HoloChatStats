"""
models.py
SQLAlchemy models mirroring the existing PostgreSQL schema created by the ETL process.
Notes:
- Materialized views (mv_*, chat_language_stats_mv) are mapped as read-only models.
  They are NOT created by db.create_all() in a meaningful way -- they continue to be
  created/refreshed by your ETL's raw SQL (CREATE MATERIALIZED VIEW / REFRESH MATERIALIZED VIEW).
  Mapping them here just lets api.py query them through the ORM.
- Foreign keys / relationships are only added where they existed in the original DDL
  (channels -> streaming_forecasts, channels -> forecast_model_metrics). Other
  logical relationships (e.g. videos.channel_id, user_data.channel_id/user_id/video_id)
  were not enforced with FOREIGN KEY constraints in the original schema, so they are
  left as plain columns to avoid changing existing ETL insert behavior. You can add
  relationship() helpers later using primaryjoin if you want ORM-level joins without
  adding DB constraints.
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import PrimaryKeyConstraint
from pgvector.sqlalchemy import Vector


db = SQLAlchemy()
# ---------------------------------------------------------------------------
# Core tables
# ---------------------------------------------------------------------------
class Channel(db.Model):
    __tablename__ = "channels"
    channel_id = db.Column(db.Text, primary_key=True)
    channel_name = db.Column(db.Text, nullable=False)
    channel_group = db.Column(db.Text)
    # Relationships to tables that *do* declare a real FK to channels
    streaming_forecasts = db.relationship(
        "StreamingForecast", backref="channel", lazy=True
    )
    forecast_model_metrics = db.relationship(
        "ForecastModelMetric", backref="channel", lazy=True
    )
    def __repr__(self):
        return f"<Channel {self.channel_id} {self.channel_name}>"
class User(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.Text, primary_key=True)
    username = db.Column(db.Text, nullable=False)
    def __repr__(self):
        return f"<User {self.user_id} {self.username}>"
class Video(db.Model):
    __tablename__ = "videos"
    video_id = db.Column(db.Text, primary_key=True)
    channel_id = db.Column(db.Text)  # no FK constraint in original DDL
    title = db.Column(db.Text, nullable=False)
    end_time = db.Column(db.TIMESTAMP(timezone=True), nullable=False)
    duration = db.Column(db.Interval)
    processed_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    has_chat_log = db.Column(db.Boolean, default=False)
    funniest_timestamp = db.Column(db.Integer)
    def __repr__(self):
        return f"<Video {self.video_id} {self.title}>"
class UserData(db.Model):
    __tablename__ = "user_data"
    user_id = db.Column(db.Text, nullable=False)
    channel_id = db.Column(db.Text, nullable=False)
    last_message_at = db.Column(db.TIMESTAMP(timezone=True), nullable=False)
    video_id = db.Column(db.Text, nullable=False)
    membership_rank = db.Column(db.Integer)
    jp_count = db.Column(db.Integer, default=0)
    kr_count = db.Column(db.Integer, default=0)
    ru_count = db.Column(db.Integer, default=0)
    emoji_count = db.Column(db.Integer, default=0)
    es_en_id_count = db.Column(db.Integer, default=0)
    total_message_count = db.Column(db.Integer, default=0)
    is_gift = db.Column(db.Boolean, default=False)
    __table_args__ = (
        PrimaryKeyConstraint(
            "user_id", "channel_id", "last_message_at", "video_id",
            name="user_data_pkey"
        ),
    )
    def __repr__(self):
        return f"<UserData {self.user_id} {self.channel_id} {self.last_message_at}>"
# ---------------------------------------------------------------------------
# Forecasting tables
# ---------------------------------------------------------------------------
class StreamingForecast(db.Model):
    __tablename__ = "streaming_forecasts"
    forecast_id = db.Column(db.Integer, primary_key=True)  # SERIAL
    channel_id = db.Column(
        db.Text, db.ForeignKey("channels.channel_id"), nullable=False
    )
    forecast_month = db.Column(db.Date, nullable=False)
    forecasted_hours = db.Column(db.Numeric(10, 2), nullable=False)
    confidence_lower = db.Column(db.Numeric(10, 2))
    confidence_upper = db.Column(db.Numeric(10, 2))
    confidence_p25 = db.Column(db.Numeric(10, 2))
    confidence_p75 = db.Column(db.Numeric(10, 2))
    model_version = db.Column(db.String(50))
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    __table_args__ = (
        db.UniqueConstraint(
            "channel_id", "forecast_month", "created_at",
            name="streaming_forecasts_channel_id_forecast_month_created_at_key"
        ),
    )
    def __repr__(self):
        return f"<StreamingForecast {self.channel_id} {self.forecast_month}>"
class ForecastModelMetric(db.Model):
    __tablename__ = "forecast_model_metrics"
    metric_id = db.Column(db.Integer, primary_key=True)  # SERIAL
    channel_id = db.Column(db.Text, db.ForeignKey("channels.channel_id"))
    mae = db.Column(db.Numeric(10, 4))
    rmse = db.Column(db.Numeric(10, 4))
    mape = db.Column(db.Numeric(10, 4))
    model_version = db.Column(db.String(50))
    training_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    def __repr__(self):
        return f"<ForecastModelMetric {self.channel_id} {self.model_version}>"
# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
class MembershipDataSummary(db.Model):
    __tablename__ = "membership_data_summary"
    channel_group = db.Column(db.Text)
    channel_name = db.Column(db.Text, nullable=False)
    observed_month = db.Column(db.Date, nullable=False)
    membership_rank = db.Column(db.Integer, nullable=False)
    membership_count = db.Column(db.BigInteger)
    percentage_total = db.Column(db.DECIMAL(5, 2))
    updated_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    __table_args__ = (
        PrimaryKeyConstraint(
            "channel_name", "observed_month", "membership_rank",
            name="membership_data_summary_pkey"
        ),
    )
    def __repr__(self):
        return f"<MembershipDataSummary {self.channel_name} {self.observed_month}>"
# ---------------------------------------------------------------------------
# Materialized views (read-only via ORM; managed by ETL's raw SQL)
# ---------------------------------------------------------------------------
class MvUserMonthlyActivity(db.Model):
    """Mirrors mv_user_monthly_activity materialized view."""
    __tablename__ = "mv_user_monthly_activity"
    __table_args__ = {"info": {"is_view": True}}
    # Materialized views have no real primary key; SQLAlchemy requires one
    # to map a class, so we designate a composite "pseudo" PK matching the
    # view's GROUP BY columns. Do not issue INSERT/UPDATE/DELETE via this model.
    user_id = db.Column(db.Text, primary_key=True)
    channel_id = db.Column(db.Text, primary_key=True)
    observed_month = db.Column(db.TIMESTAMP(timezone=True), primary_key=True)
    monthly_message_count = db.Column(db.BigInteger)
class MvUserActivity(db.Model):
    """Mirrors mv_user_activity materialized view."""
    __tablename__ = "mv_user_activity"
    __table_args__ = {"info": {"is_view": True}}
    user_id = db.Column(db.Text, primary_key=True)
    activity_month = db.Column(db.TIMESTAMP(timezone=True), primary_key=True)
    channel_id = db.Column(db.Text, primary_key=True)
    channel_group = db.Column(db.Text)
class ChatLanguageStatsMv(db.Model):
    """Mirrors chat_language_stats_mv materialized view."""
    __tablename__ = "chat_language_stats_mv"
    __table_args__ = {"info": {"is_view": True}}
    channel_id = db.Column(db.Text, primary_key=True)
    observed_month = db.Column(db.TIMESTAMP(timezone=True), primary_key=True)
    jp_count = db.Column(db.BigInteger)
    kr_count = db.Column(db.BigInteger)
    ru_count = db.Column(db.BigInteger)
    emoji_count = db.Column(db.BigInteger)
    es_en_id_count = db.Column(db.BigInteger)
    total_messages = db.Column(db.BigInteger)
class MvUserLanguagePerMonth(db.Model):
    """Mirrors mv_user_language_per_month materialized view."""
    __tablename__ = "mv_user_language_per_month"
    __table_args__ = {"info": {"is_view": True}}
    user_id = db.Column(db.Text, primary_key=True)
    channel_id = db.Column(db.Text, primary_key=True)
    month = db.Column(db.TIMESTAMP(timezone=True), primary_key=True)
    total_jp_messages = db.Column(db.BigInteger)
    total_non_emoji_messages = db.Column(db.BigInteger)

class VideoHighlight(db.Model):
    __tablename__ = "video_highlights"
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Text, db.ForeignKey("videos.video_id"))
    topic_tag = db.Column(db.Text)
    generated_summary = db.Column(db.Text)
    start_seconds = db.Column(db.Integer)
    summary_embedding = db.Column(Vector(384))
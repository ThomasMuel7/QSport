import logging
from datetime import datetime
from typing import Optional
from tzfpy import get_tz

def fetch_timezone_info(latitude: float, longitude: float, match_datetime_utc: datetime) -> Optional[dict]:
    try:
        
        timezone_id = get_tz(longitude, latitude)
        if not timezone_id:
            return None
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_id)
        offset_seconds = match_datetime_utc.astimezone(tz).utcoffset().total_seconds()
        return {
            "timeZoneId": timezone_id,
            "rawOffset": int(offset_seconds),
            "dstOffset": 0
        }
    except Exception as e:
        logging.error(f"[timezone_collector] Erreur: {e}")
        return None

def get_offset_hours(tz_info: dict) -> float:
    if tz_info is None:
        return None
    return (tz_info.get("rawOffset", 0) + tz_info.get("dstOffset", 0)) / 3600

"""DeepSeek's billing windows, evaluated only when dispatching new subagents."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("tools.delegate_tool")
_BEIJING = timezone(timedelta(hours=8), "Asia/Shanghai")


def is_deepseek_off_peak(now: datetime | None = None) -> bool:
    """Whether an instant falls in DeepSeek's discounted Beijing-time window.

    Only non-holiday Monday-Friday 09:00-12:00 and 14:00-18:00 are peak.
    Weekend make-up workdays are still off-peak under DeepSeek's billing rule.
    Missing/expired holiday data conservatively keeps ordinary weekday peak
    hours on the configured route; the other windows need no holiday lookup.
    """
    if now is None:
        now = datetime.now(_BEIJING)
    if now.utcoffset() is None:
        raise ValueError("DeepSeek billing time must be timezone-aware")
    local = now.astimezone(_BEIJING)
    minute = local.hour * 60 + local.minute
    weekday_peak = local.weekday() < 5 and (
        9 * 60 <= minute < 12 * 60 or 14 * 60 <= minute < 18 * 60
    )
    if not weekday_peak:
        return True
    try:
        from chinese_calendar import is_holiday

        return is_holiday(local.date())
    except (ImportError, NotImplementedError, ValueError):
        logger.warning(
            "DeepSeek off-peak: holiday calendar unavailable for %s; keeping the "
            "configured delegation route during weekday peak hours. Update chinesecalendar.",
            local.date(),
        )
        return False

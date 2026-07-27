// Deliberately a no-op pass-through, not a real offline cache — this app
// is only ever useful with a live connection to mobile_api.py (account
// equity, ER, journal), so caching stale trading data would be actively
// misleading rather than helpful. This file exists purely because some
// versions of Chrome's "installable PWA" / "Add to Home Screen -> full
// app" criteria historically wanted a registered service worker present;
// costs nothing to include, does nothing beyond letting requests through.
self.addEventListener("fetch", () => {});

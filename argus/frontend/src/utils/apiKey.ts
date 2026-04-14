// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — API Key Manager
// SECURITY: API key is stored in sessionStorage (cleared when tab closes)
// and cached in a module-level variable for fast access.
// Never use localStorage for credentials — it persists across sessions.
// ═══════════════════════════════════════════════════════════════════════

const STORAGE_KEY = 'ARGUS_API_KEY';

// In-memory cache (fastest access, cleared on page reload)
let _cachedKey: string | null = null;

/**
 * Get the API key. Checks in-memory cache first, then sessionStorage.
 */
export function getApiKey(): string {
  if (_cachedKey !== null) return _cachedKey;
  if (typeof window !== 'undefined') {
    _cachedKey = sessionStorage.getItem(STORAGE_KEY) || '';

    // MIGRATION: If key exists in localStorage (old behavior), move it
    const legacyKey = localStorage.getItem(STORAGE_KEY);
    if (legacyKey && !_cachedKey) {
      _cachedKey = legacyKey;
      sessionStorage.setItem(STORAGE_KEY, legacyKey);
      localStorage.removeItem(STORAGE_KEY);
    }

    return _cachedKey;
  }
  return '';
}

/**
 * Set the API key. Stores in sessionStorage + in-memory cache.
 */
export function setApiKey(key: string): void {
  _cachedKey = key;
  if (typeof window !== 'undefined') {
    sessionStorage.setItem(STORAGE_KEY, key);
    // Ensure localStorage copy is removed (migration cleanup)
    localStorage.removeItem(STORAGE_KEY);
  }
}

/**
 * Clear the API key from all storage.
 */
export function clearApiKey(): void {
  _cachedKey = null;
  if (typeof window !== 'undefined') {
    sessionStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY);
  }
}

"""ARGUS-X TUI — Single source of truth for all colors.
Every widget imports from here. No hardcoded hex outside this file."""

# Core surfaces
VOID         = "#0a0a0b"
SURFACE      = "#18181b"
ELEVATED     = "#27272a"
BORDER       = "#27272a"

# Foreground
FG_PRIMARY   = "#f2f2f2"
FG_SECONDARY = "#64748b"
FG_DIM       = "#3f3f46"

# Single accent
ACCENT       = "#3b82f6"

# Threat — reserved, appears only on real danger
THREAT       = "#ef4444"

# Status semantics
STATUS_BLOCKED  = "#ef4444"
STATUS_CLEAN    = "#22c55e"
STATUS_WARNING  = "#f59e0b"
STATUS_MUTATION = "#a78bfa"
STATUS_CRITICAL = "#dc2626"

# Threat type → (abbreviation, color)
THREAT_TYPE_STYLES: dict[str, tuple[str, str]] = {
    "PROMPT_INJECTION":   ("PI", STATUS_BLOCKED),
    "JAILBREAK":          ("JB", STATUS_BLOCKED),
    "DATA_EXFILTRATION":  ("DE", STATUS_MUTATION),
    "ROLE_HIJACKING":     ("RH", STATUS_WARNING),
    "INDIRECT_INJECTION": ("II", STATUS_BLOCKED),
    "MULTI_TURN":         ("MT", STATUS_WARNING),
    "CLEAN":              ("CL", STATUS_CLEAN),
}

# Recommended action → color
ACTION_COLORS: dict[str, str] = {
    "ALLOW":                 FG_DIM,
    "ALLOW_WITH_MONITORING": STATUS_CLEAN,
    "BLOCK":                 STATUS_WARNING,
    "BLOCK_AND_MONITOR":     STATUS_WARNING,
    "ESCALATE_MONITORING":   STATUS_MUTATION,
    "TERMINATE_SESSION":     STATUS_CRITICAL,
}

# Sophistication label → color
LEVEL_COLORS: dict[str, str] = {
    "NAIVE":        FG_SECONDARY,
    "ELEMENTARY":   ACCENT,
    "INTERMEDIATE": STATUS_WARNING,
    "ADVANCED":     STATUS_BLOCKED,
    "APEX":         STATUS_MUTATION,
}

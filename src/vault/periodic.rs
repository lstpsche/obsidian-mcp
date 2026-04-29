//! Periodic notes: date format conversion, daily/weekly/monthly note resolution.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use chrono::{Datelike, Local, NaiveDate};
use regex::Regex;
use serde::Deserialize;
use walkdir::WalkDir;

use crate::error::{VaultError, VaultResult};
use crate::models::{NotePeriod, PeriodicNoteConfig};

/// Placeholder injected for the Moment.js `Q` (quarter) token,
/// which has no chrono equivalent. Resolved by `format_date`.
const QUARTER_PLACEHOLDER: &str = "\x01Q\x01";

// ── Internal deserialization types ─────────────────────────────────────

/// Core daily notes config: `.obsidian/daily-notes.json`
#[derive(Deserialize)]
struct CoreDailyNotesConfig {
    format: Option<String>,
    folder: Option<String>,
    template: Option<String>,
}

/// A single period entry in the Periodic Notes plugin config.
#[derive(Deserialize)]
struct PluginPeriodEntry {
    enabled: Option<bool>,
    format: Option<String>,
    folder: Option<String>,
    #[serde(alias = "templatePath")]
    template: Option<String>,
}

/// Periodic Notes community plugin config.
/// Supports the newer CalendarSet format and the legacy flat format.
#[derive(Deserialize)]
#[serde(untagged)]
enum PluginPeriodicConfig {
    CalendarSets {
        #[serde(rename = "calendarSets")]
        calendar_sets: Vec<CalendarSet>,
    },
    Legacy(Box<LegacyPluginConfig>),
}

#[derive(Deserialize)]
struct CalendarSet {
    day: Option<PluginPeriodEntry>,
    week: Option<PluginPeriodEntry>,
    month: Option<PluginPeriodEntry>,
    quarter: Option<PluginPeriodEntry>,
    year: Option<PluginPeriodEntry>,
}

#[derive(Deserialize)]
struct LegacyPluginConfig {
    #[serde(alias = "day")]
    daily: Option<PluginPeriodEntry>,
    #[serde(alias = "week")]
    weekly: Option<PluginPeriodEntry>,
    #[serde(alias = "month")]
    monthly: Option<PluginPeriodEntry>,
    #[serde(alias = "quarter")]
    quarterly: Option<PluginPeriodEntry>,
    #[serde(alias = "year")]
    yearly: Option<PluginPeriodEntry>,
}

// ── Default formats ────────────────────────────────────────────────────

fn default_format(period: &NotePeriod) -> &'static str {
    match period {
        NotePeriod::Daily => "YYYY-MM-DD",
        NotePeriod::Weekly => "gggg-[W]ww",
        NotePeriod::Monthly => "YYYY-MM",
        NotePeriod::Quarterly => "YYYY-[Q]Q",
        NotePeriod::Yearly => "YYYY",
    }
}

fn default_config(period: &NotePeriod) -> PeriodicNoteConfig {
    PeriodicNoteConfig {
        format: default_format(period).to_string(),
        folder: None,
        template: None,
    }
}

// ── Moment.js → chrono format conversion ───────────────────────────────

/// Token mapping table. **Must** be sorted by token length descending so the
/// scanner picks the longest match first (e.g. `MMMM` before `MM`).
const TOKEN_MAP: &[(&str, &str)] = &[
    // 4-char
    ("GGGG", "%G"),
    ("MMMM", "%B"),
    ("YYYY", "%Y"),
    ("dddd", "%A"),
    ("gggg", "%G"),
    // 3-char
    ("MMM", "%b"),
    ("ddd", "%a"),
    // 2-char
    ("DD", "%d"),
    ("Do", "%-d"),
    ("GG", "%G"),
    ("HH", "%H"),
    ("MM", "%m"),
    ("WW", "%V"),
    ("YY", "%y"),
    ("ZZ", "%z"),
    ("dd", "%a"),
    ("gg", "%G"),
    ("hh", "%I"),
    ("mm", "%M"),
    ("ss", "%S"),
    ("ww", "%V"),
    // 1-char
    ("A", "%p"),
    ("D", "%-d"),
    ("E", "%u"),
    ("H", "%-H"),
    ("M", "%-m"),
    ("Q", QUARTER_PLACEHOLDER),
    ("S", ""),
    ("W", "%V"),
    ("X", "%s"),
    ("Z", "%:z"),
    ("a", "%p"),
    ("d", "%w"),
    ("e", "%w"),
    ("h", "%-I"),
    ("m", "%-M"),
    ("s", "%-S"),
    ("w", "%V"),
];

/// Convert a Moment.js date format string to a chrono format string.
///
/// Handles `[literal]` escaping and longest-token-first replacement.
/// The quarter token `Q` becomes [`QUARTER_PLACEHOLDER`], resolved later
/// by [`format_date`].
fn momentjs_to_chrono(moment_format: &str) -> String {
    let mut literals: Vec<String> = Vec::new();
    let mut preprocessed = String::with_capacity(moment_format.len());
    let mut chars = moment_format.chars();

    while let Some(ch) = chars.next() {
        if ch == '[' {
            let mut literal = String::new();
            for inner in chars.by_ref() {
                if inner == ']' {
                    break;
                }
                literal.push(inner);
            }
            let idx = literals.len();
            literals.push(literal);
            preprocessed.push('\x00');
            preprocessed.push_str(&idx.to_string());
            preprocessed.push('\x00');
        } else {
            preprocessed.push(ch);
        }
    }

    let mut result = String::with_capacity(preprocessed.len() * 2);
    let mut pos = 0;

    while pos < preprocessed.len() {
        let slice = &preprocessed[pos..];
        let mut matched = false;

        for &(token, replacement) in TOKEN_MAP {
            if slice.starts_with(token) {
                result.push_str(replacement);
                pos += token.len();
                matched = true;
                break;
            }
        }

        if !matched {
            let ch = slice.chars().next().expect("non-empty slice");
            result.push(ch);
            pos += ch.len_utf8();
        }
    }

    for (idx, literal) in literals.iter().enumerate() {
        let placeholder = format!("\x00{idx}\x00");
        result = result.replace(&placeholder, literal);
    }

    result
}

/// Format a `NaiveDate` using a Moment.js format string.
/// Converts to chrono internally and resolves the quarter placeholder.
fn format_date(date: &NaiveDate, moment_format: &str) -> String {
    let chrono_fmt = momentjs_to_chrono(moment_format);
    let formatted = date.format(&chrono_fmt).to_string();

    if formatted.contains(QUARTER_PLACEHOLDER) {
        let quarter = (date.month() - 1) / 3 + 1;
        formatted.replace(QUARTER_PLACEHOLDER, &quarter.to_string())
    } else {
        formatted
    }
}

// ── Config reading ─────────────────────────────────────────────────────

/// Read periodic notes config from Obsidian's config files.
///
/// Resolution order:
/// 1. Periodic Notes community plugin (`.obsidian/plugins/periodic-notes/data.json`)
/// 2. Core daily-notes config (`.obsidian/daily-notes.json`) — only for `Daily`
/// 3. Built-in defaults
pub fn read_periodic_config(
    vault_root: &Path,
    period: &NotePeriod,
) -> VaultResult<PeriodicNoteConfig> {
    let plugin_path = vault_root
        .join(".obsidian")
        .join("plugins")
        .join("periodic-notes")
        .join("data.json");

    if let Some(config) = try_read_plugin_config(&plugin_path, period)? {
        return Ok(config);
    }

    if matches!(period, NotePeriod::Daily) {
        let core_path = vault_root.join(".obsidian").join("daily-notes.json");
        if let Some(config) = try_read_core_daily_config(&core_path)? {
            return Ok(config);
        }
    }

    Ok(default_config(period))
}

/// Read and deserialize a JSON config file, returning `None` if missing.
fn read_json_config<T: serde::de::DeserializeOwned>(path: &Path) -> VaultResult<Option<T>> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(VaultError::Io(e)),
    };
    let parsed: T = serde_json::from_str(&content)
        .map_err(|e| VaultError::Other(format!("Failed to parse {}: {e}", path.display())))?;
    Ok(Some(parsed))
}

fn try_read_plugin_config(
    path: &Path,
    period: &NotePeriod,
) -> VaultResult<Option<PeriodicNoteConfig>> {
    let Some(parsed): Option<PluginPeriodicConfig> = read_json_config(path)? else {
        return Ok(None);
    };

    let entry = match parsed {
        PluginPeriodicConfig::CalendarSets { calendar_sets } => calendar_sets
            .into_iter()
            .next()
            .and_then(|set| extract_from_calendar_set(set, period)),
        PluginPeriodicConfig::Legacy(legacy) => extract_from_legacy(*legacy, period),
    };

    match entry {
        Some(e) if e.enabled.unwrap_or(false) => Ok(Some(entry_to_config(e, period))),
        _ => Ok(None),
    }
}

fn extract_from_calendar_set(set: CalendarSet, period: &NotePeriod) -> Option<PluginPeriodEntry> {
    match period {
        NotePeriod::Daily => set.day,
        NotePeriod::Weekly => set.week,
        NotePeriod::Monthly => set.month,
        NotePeriod::Quarterly => set.quarter,
        NotePeriod::Yearly => set.year,
    }
}

fn extract_from_legacy(
    legacy: LegacyPluginConfig,
    period: &NotePeriod,
) -> Option<PluginPeriodEntry> {
    match period {
        NotePeriod::Daily => legacy.daily,
        NotePeriod::Weekly => legacy.weekly,
        NotePeriod::Monthly => legacy.monthly,
        NotePeriod::Quarterly => legacy.quarterly,
        NotePeriod::Yearly => legacy.yearly,
    }
}

fn entry_to_config(entry: PluginPeriodEntry, period: &NotePeriod) -> PeriodicNoteConfig {
    PeriodicNoteConfig {
        format: entry
            .format
            .filter(|f| !f.is_empty())
            .unwrap_or_else(|| default_format(period).to_string()),
        folder: entry.folder.filter(|f| !f.is_empty()),
        template: entry.template.filter(|t| !t.is_empty()),
    }
}

fn try_read_core_daily_config(path: &Path) -> VaultResult<Option<PeriodicNoteConfig>> {
    let Some(parsed): Option<CoreDailyNotesConfig> = read_json_config(path)? else {
        return Ok(None);
    };

    Ok(Some(PeriodicNoteConfig {
        format: parsed
            .format
            .filter(|f| !f.is_empty())
            .unwrap_or_else(|| "YYYY-MM-DD".to_string()),
        folder: parsed.folder.filter(|f| !f.is_empty()),
        template: parsed.template.filter(|t| !t.is_empty()),
    }))
}

// ── Path derivation ────────────────────────────────────────────────────

/// Derive the file path for a periodic note given a date.
/// Returns path relative to vault root.
pub fn periodic_note_path(config: &PeriodicNoteConfig, date: &NaiveDate) -> PathBuf {
    let filename = format!("{}.md", format_date(date, &config.format));
    match &config.folder {
        Some(folder) if !folder.is_empty() => PathBuf::from(folder).join(filename),
        _ => PathBuf::from(filename),
    }
}

/// Get the file path for the current period's periodic note.
pub fn current_periodic_note_path(config: &PeriodicNoteConfig, _period: &NotePeriod) -> PathBuf {
    let today = Local::now().date_naive();
    periodic_note_path(config, &today)
}

// ── List recent periodic notes ─────────────────────────────────────────

/// List recent periodic notes by scanning the configured folder
/// and parsing filenames back to dates. Returns sorted newest-first.
pub fn list_recent_periodic_notes(
    vault_root: &Path,
    config: &PeriodicNoteConfig,
    limit: usize,
) -> VaultResult<Vec<PathBuf>> {
    let scan_dir = match &config.folder {
        Some(folder) if !folder.is_empty() => vault_root.join(folder),
        _ => vault_root.to_path_buf(),
    };

    if !scan_dir.is_dir() {
        return Ok(Vec::new());
    }

    let chrono_fmt = momentjs_to_chrono(&config.format);
    let has_quarter = chrono_fmt.contains(QUARTER_PLACEHOLDER);

    let mut dated_paths: Vec<(NaiveDate, PathBuf)> = Vec::new();

    for entry in WalkDir::new(&scan_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }

        let rel = match path.strip_prefix(&scan_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let stem = rel.with_extension("");
        let stem_str = stem.to_string_lossy().replace('\\', "/");

        if let Some(date) = try_parse_date(&stem_str, &chrono_fmt, has_quarter) {
            let vault_rel = match &config.folder {
                Some(folder) if !folder.is_empty() => PathBuf::from(folder).join(rel),
                _ => rel.to_path_buf(),
            };
            dated_paths.push((date, vault_rel));
        }
    }

    dated_paths.sort_by_key(|x| std::cmp::Reverse(x.0));
    dated_paths.truncate(limit);

    Ok(dated_paths.into_iter().map(|(_, p)| p).collect())
}

/// Try to parse a filename stem into a `NaiveDate`.
///
/// Attempts several strategies to handle daily (full date), weekly (ISO week),
/// monthly (year-month), and yearly (year only) formats.
fn try_parse_date(stem: &str, chrono_fmt: &str, has_quarter: bool) -> Option<NaiveDate> {
    if has_quarter {
        return try_parse_quarter_date(stem, chrono_fmt);
    }

    if let Ok(date) = NaiveDate::parse_from_str(stem, chrono_fmt) {
        return Some(date);
    }

    // ISO week without day-of-week: append Monday
    if chrono_fmt.contains("%G") || chrono_fmt.contains("%V") {
        let with_dow = format!("{stem}-1");
        let fmt_with_dow = format!("{chrono_fmt}-%u");
        if let Ok(date) = NaiveDate::parse_from_str(&with_dow, &fmt_with_dow) {
            return Some(date);
        }
    }

    // Monthly: append day-01
    {
        let with_day = format!("{stem}-01");
        let fmt_with_day = format!("{chrono_fmt}-%d");
        if let Ok(date) = NaiveDate::parse_from_str(&with_day, &fmt_with_day) {
            return Some(date);
        }
    }

    // Yearly: append month-01 and day-01
    {
        let with_md = format!("{stem}-01-01");
        let fmt_with_md = format!("{chrono_fmt}-%m-%d");
        if let Ok(date) = NaiveDate::parse_from_str(&with_md, &fmt_with_md) {
            return Some(date);
        }
    }

    None
}

/// Parse dates from filenames that contain the quarter token.
/// Extracts year and quarter via regex, then reconstructs a `NaiveDate`
/// using the first day of the quarter.
fn try_parse_quarter_date(stem: &str, chrono_fmt: &str) -> Option<NaiveDate> {
    let with_markers = chrono_fmt
        .replace(QUARTER_PLACEHOLDER, "<<Q>>")
        .replace("%Y", "<<Y4>>")
        .replace("%y", "<<Y2>>");

    let escaped = regex::escape(&with_markers)
        .replace("<<Q>>", r"(\d)")
        .replace("<<Y4>>", r"(\d{4})")
        .replace("<<Y2>>", r"(\d{2})");

    let re = Regex::new(&format!("^{escaped}$")).ok()?;
    let caps = re.captures(stem)?;

    let mut capture_info: Vec<(usize, char)> = Vec::new();
    for (marker, kind) in [("<<Y4>>", '4'), ("<<Y2>>", '2'), ("<<Q>>", 'Q')] {
        if let Some(pos) = with_markers.find(marker) {
            capture_info.push((pos, kind));
        }
    }
    capture_info.sort_by_key(|&(pos, _)| pos);

    let mut year: Option<i32> = None;
    let mut quarter: Option<u32> = None;

    for (i, &(_, kind)) in capture_info.iter().enumerate() {
        let val = caps.get(i + 1)?.as_str();
        match kind {
            '4' => year = val.parse().ok(),
            '2' => year = val.parse::<i32>().ok().map(|y| 2000 + y),
            'Q' => quarter = val.parse().ok(),
            _ => {}
        }
    }

    let y = year?;
    let q = quarter.filter(|&q| (1..=4).contains(&q))?;
    let month = (q - 1) * 3 + 1;
    NaiveDate::from_ymd_opt(y, month, 1)
}

// ── Template expansion ─────────────────────────────────────────────────

/// Read a template file and expand template variables:
/// `{{date}}`, `{{date:FORMAT}}`, `{{title}}`, `{{time}}`.
///
/// If `template_path` has no file extension, `.md` is appended automatically.
pub fn expand_template(
    vault_root: &Path,
    template_path: &Path,
    date: &NaiveDate,
    title: &str,
) -> VaultResult<String> {
    let full_path = vault_root.join(template_path);
    let full_path = if full_path.extension().is_none() {
        full_path.with_extension("md")
    } else {
        full_path
    };

    let content = std::fs::read_to_string(&full_path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => VaultError::NoteNotFound(template_path.to_path_buf()),
        _ => VaultError::Io(e),
    })?;

    Ok(expand_template_string(&content, date, title))
}

static DATE_FORMAT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{date:([^}]+)\}\}").unwrap());

fn expand_template_string(content: &str, date: &NaiveDate, title: &str) -> String {
    let result = DATE_FORMAT_RE
        .replace_all(content, |caps: &regex::Captures| {
            format_date(date, &caps[1])
        })
        .into_owned();

    let result = result.replace("{{title}}", title);
    let result = result.replace("{{date}}", &format_date(date, "YYYY-MM-DD"));
    result.replace("{{time}}", &Local::now().format("%H:%M").to_string())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── momentjs_to_chrono ─────────────────────────────────────────────

    #[test]
    fn momentjs_daily_format() {
        assert_eq!(momentjs_to_chrono("YYYY-MM-DD"), "%Y-%m-%d");
    }

    #[test]
    fn momentjs_full_month_and_weekday() {
        assert_eq!(momentjs_to_chrono("dddd, MMMM D, YYYY"), "%A, %B %-d, %Y");
    }

    #[test]
    fn momentjs_time_tokens() {
        assert_eq!(momentjs_to_chrono("HH:mm:ss"), "%H:%M:%S");
    }

    #[test]
    fn momentjs_bracket_escaping() {
        assert_eq!(momentjs_to_chrono("gggg-[W]ww"), "%G-W%V");
    }

    #[test]
    fn momentjs_quarter_format() {
        let result = momentjs_to_chrono("YYYY-[Q]Q");
        assert!(result.starts_with("%Y-Q"));
        assert!(result.contains(QUARTER_PLACEHOLDER));
    }

    #[test]
    fn momentjs_two_digit_year() {
        assert_eq!(momentjs_to_chrono("YY"), "%y");
    }

    #[test]
    fn momentjs_short_month() {
        assert_eq!(momentjs_to_chrono("MMM"), "%b");
    }

    #[test]
    fn momentjs_12hour_format() {
        assert_eq!(momentjs_to_chrono("hh:mm A"), "%I:%M %p");
    }

    #[test]
    fn momentjs_multiple_brackets() {
        assert_eq!(
            momentjs_to_chrono("[Today is] dddd [the] Do [of] MMMM"),
            "Today is %A the %-d of %B"
        );
    }

    #[test]
    fn momentjs_monthly() {
        assert_eq!(momentjs_to_chrono("YYYY-MM"), "%Y-%m");
    }

    #[test]
    fn momentjs_yearly() {
        assert_eq!(momentjs_to_chrono("YYYY"), "%Y");
    }

    // ── format_date ────────────────────────────────────────────────────

    #[test]
    fn format_date_daily() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY-MM-DD"), "2026-03-19");
    }

    #[test]
    fn format_date_quarter_q1() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q"), "2026-Q1");
    }

    #[test]
    fn format_date_quarter_q2() {
        let d = NaiveDate::from_ymd_opt(2026, 6, 15).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q"), "2026-Q2");
    }

    #[test]
    fn format_date_quarter_q3() {
        let d = NaiveDate::from_ymd_opt(2026, 9, 1).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q"), "2026-Q3");
    }

    #[test]
    fn format_date_quarter_q4() {
        let d = NaiveDate::from_ymd_opt(2026, 12, 31).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q"), "2026-Q4");
    }

    #[test]
    fn format_date_monthly() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY-MM"), "2026-03");
    }

    #[test]
    fn format_date_yearly() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY"), "2026");
    }

    #[test]
    fn format_date_full_text() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(
            format_date(&d, "dddd, MMMM D, YYYY"),
            "Thursday, March 19, 2026"
        );
    }

    // ── periodic_note_path ─────────────────────────────────────────────

    #[test]
    fn path_with_folder() {
        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: Some("Daily".to_string()),
            template: None,
        };
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(
            periodic_note_path(&config, &d),
            PathBuf::from("Daily/2026-03-19.md")
        );
    }

    #[test]
    fn path_without_folder() {
        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: None,
            template: None,
        };
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(
            periodic_note_path(&config, &d),
            PathBuf::from("2026-03-19.md")
        );
    }

    #[test]
    fn path_quarterly() {
        let config = PeriodicNoteConfig {
            format: "YYYY-[Q]Q".to_string(),
            folder: Some("Quarterly".to_string()),
            template: None,
        };
        let d = NaiveDate::from_ymd_opt(2026, 7, 1).unwrap();
        assert_eq!(
            periodic_note_path(&config, &d),
            PathBuf::from("Quarterly/2026-Q3.md")
        );
    }

    #[test]
    fn path_empty_folder_treated_as_none() {
        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: Some(String::new()),
            template: None,
        };
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        assert_eq!(
            periodic_note_path(&config, &d),
            PathBuf::from("2026-01-01.md")
        );
    }

    // ── read_periodic_config ───────────────────────────────────────────

    #[test]
    fn config_defaults_when_no_files() {
        let tmp = TempDir::new().unwrap();
        let config = read_periodic_config(tmp.path(), &NotePeriod::Daily).unwrap();
        assert_eq!(config.format, "YYYY-MM-DD");
        assert!(config.folder.is_none());
        assert!(config.template.is_none());
    }

    #[test]
    fn config_defaults_weekly() {
        let tmp = TempDir::new().unwrap();
        let config = read_periodic_config(tmp.path(), &NotePeriod::Weekly).unwrap();
        assert_eq!(config.format, "gggg-[W]ww");
    }

    #[test]
    fn config_defaults_monthly() {
        let tmp = TempDir::new().unwrap();
        let config = read_periodic_config(tmp.path(), &NotePeriod::Monthly).unwrap();
        assert_eq!(config.format, "YYYY-MM");
    }

    #[test]
    fn config_defaults_quarterly() {
        let tmp = TempDir::new().unwrap();
        let config = read_periodic_config(tmp.path(), &NotePeriod::Quarterly).unwrap();
        assert_eq!(config.format, "YYYY-[Q]Q");
    }

    #[test]
    fn config_defaults_yearly() {
        let tmp = TempDir::new().unwrap();
        let config = read_periodic_config(tmp.path(), &NotePeriod::Yearly).unwrap();
        assert_eq!(config.format, "YYYY");
    }

    #[test]
    fn config_core_daily() {
        let tmp = TempDir::new().unwrap();
        let obsidian = tmp.path().join(".obsidian");
        fs::create_dir_all(&obsidian).unwrap();
        fs::write(
            obsidian.join("daily-notes.json"),
            r#"{"format":"DD-MM-YYYY","folder":"Journal","template":"Templates/Day"}"#,
        )
        .unwrap();

        let config = read_periodic_config(tmp.path(), &NotePeriod::Daily).unwrap();
        assert_eq!(config.format, "DD-MM-YYYY");
        assert_eq!(config.folder.as_deref(), Some("Journal"));
        assert_eq!(config.template.as_deref(), Some("Templates/Day"));
    }

    #[test]
    fn config_core_daily_empty_format_uses_default() {
        let tmp = TempDir::new().unwrap();
        let obsidian = tmp.path().join(".obsidian");
        fs::create_dir_all(&obsidian).unwrap();
        fs::write(
            obsidian.join("daily-notes.json"),
            r#"{"format":"","folder":"Journal"}"#,
        )
        .unwrap();

        let config = read_periodic_config(tmp.path(), &NotePeriod::Daily).unwrap();
        assert_eq!(config.format, "YYYY-MM-DD");
        assert_eq!(config.folder.as_deref(), Some("Journal"));
    }

    #[test]
    fn config_plugin_legacy_overrides_core() {
        let tmp = TempDir::new().unwrap();
        let obsidian = tmp.path().join(".obsidian");
        fs::create_dir_all(&obsidian).unwrap();
        fs::write(
            obsidian.join("daily-notes.json"),
            r#"{"format":"DD-MM-YYYY","folder":"Journal"}"#,
        )
        .unwrap();

        let plugin_dir = obsidian.join("plugins").join("periodic-notes");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("data.json"),
            r#"{"daily":{"enabled":true,"format":"YYYY/MM/DD","folder":"Daily","templatePath":""}}"#,
        )
        .unwrap();

        let config = read_periodic_config(tmp.path(), &NotePeriod::Daily).unwrap();
        assert_eq!(config.format, "YYYY/MM/DD");
        assert_eq!(config.folder.as_deref(), Some("Daily"));
        assert!(config.template.is_none()); // empty templatePath → None
    }

    #[test]
    fn config_plugin_calendar_sets() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join(".obsidian/plugins/periodic-notes");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("data.json"),
            r#"{"calendarSets":[{"id":"default","ctime":"2026-01-01","week":{"enabled":true,"format":"GGGG-[W]WW","folder":"Weekly","openAtStartup":false}}]}"#,
        )
        .unwrap();

        let config = read_periodic_config(tmp.path(), &NotePeriod::Weekly).unwrap();
        assert_eq!(config.format, "GGGG-[W]WW");
        assert_eq!(config.folder.as_deref(), Some("Weekly"));
    }

    #[test]
    fn config_plugin_disabled_falls_to_default() {
        let tmp = TempDir::new().unwrap();
        let plugin_dir = tmp.path().join(".obsidian/plugins/periodic-notes");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("data.json"),
            r#"{"weekly":{"enabled":false,"format":"gggg-[W]ww","folder":"W"}}"#,
        )
        .unwrap();

        let config = read_periodic_config(tmp.path(), &NotePeriod::Weekly).unwrap();
        assert_eq!(config.format, "gggg-[W]ww");
        assert!(config.folder.is_none());
    }

    // ── list_recent_periodic_notes ─────────────────────────────────────

    #[test]
    fn list_daily_sorted_and_limited() {
        let tmp = TempDir::new().unwrap();
        let daily = tmp.path().join("Daily");
        fs::create_dir_all(&daily).unwrap();
        fs::write(daily.join("2026-03-19.md"), "").unwrap();
        fs::write(daily.join("2026-03-18.md"), "").unwrap();
        fs::write(daily.join("2026-03-17.md"), "").unwrap();
        fs::write(daily.join("not-a-date.md"), "").unwrap();
        fs::write(daily.join("readme.txt"), "").unwrap();

        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: Some("Daily".to_string()),
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], PathBuf::from("Daily/2026-03-19.md"));
        assert_eq!(result[1], PathBuf::from("Daily/2026-03-18.md"));
    }

    #[test]
    fn list_monthly() {
        let tmp = TempDir::new().unwrap();
        let monthly = tmp.path().join("Monthly");
        fs::create_dir_all(&monthly).unwrap();
        fs::write(monthly.join("2026-03.md"), "").unwrap();
        fs::write(monthly.join("2026-02.md"), "").unwrap();
        fs::write(monthly.join("2026-01.md"), "").unwrap();

        let config = PeriodicNoteConfig {
            format: "YYYY-MM".to_string(),
            folder: Some("Monthly".to_string()),
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 10).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], PathBuf::from("Monthly/2026-03.md"));
        assert_eq!(result[1], PathBuf::from("Monthly/2026-02.md"));
        assert_eq!(result[2], PathBuf::from("Monthly/2026-01.md"));
    }

    #[test]
    fn list_quarterly() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("Q");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("2026-Q1.md"), "").unwrap();
        fs::write(dir.join("2025-Q4.md"), "").unwrap();

        let config = PeriodicNoteConfig {
            format: "YYYY-[Q]Q".to_string(),
            folder: Some("Q".to_string()),
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 10).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], PathBuf::from("Q/2026-Q1.md"));
        assert_eq!(result[1], PathBuf::from("Q/2025-Q4.md"));
    }

    #[test]
    fn list_yearly() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("Yearly");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("2026.md"), "").unwrap();
        fs::write(dir.join("2025.md"), "").unwrap();
        fs::write(dir.join("2024.md"), "").unwrap();

        let config = PeriodicNoteConfig {
            format: "YYYY".to_string(),
            folder: Some("Yearly".to_string()),
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], PathBuf::from("Yearly/2026.md"));
        assert_eq!(result[1], PathBuf::from("Yearly/2025.md"));
    }

    #[test]
    fn list_nonexistent_folder_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: Some("Nonexistent".to_string()),
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 10).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn list_no_folder_scans_vault_root() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("2026-03-19.md"), "").unwrap();
        fs::write(tmp.path().join("2026-03-18.md"), "").unwrap();

        let config = PeriodicNoteConfig {
            format: "YYYY-MM-DD".to_string(),
            folder: None,
            template: None,
        };
        let result = list_recent_periodic_notes(tmp.path(), &config, 10).unwrap();
        assert_eq!(result.len(), 2);
    }

    // ── expand_template ────────────────────────────────────────────────

    #[test]
    fn template_title_replacement() {
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let result = expand_template_string("Hello {{title}}!", &d, "World");
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn template_date_replacement() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        let result = expand_template_string("Date: {{date}}", &d, "");
        assert_eq!(result, "Date: 2026-03-19");
    }

    #[test]
    fn template_date_with_custom_format() {
        let d = NaiveDate::from_ymd_opt(2026, 12, 25).unwrap();
        let result = expand_template_string("{{date:MMMM DD, YYYY}}", &d, "");
        assert_eq!(result, "December 25, 2026");
    }

    #[test]
    fn template_time_has_hh_mm_pattern() {
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let result = expand_template_string("Time: {{time}}", &d, "");
        let time_part = result.strip_prefix("Time: ").unwrap();
        assert_eq!(time_part.len(), 5);
        assert_eq!(&time_part[2..3], ":");
    }

    #[test]
    fn template_combined_variables() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        let result = expand_template_string(
            "# {{title}}\nDate: {{date}}\n{{date:dddd, MMMM D}}",
            &d,
            "2026-03-19",
        );
        assert!(result.contains("# 2026-03-19"));
        assert!(result.contains("Date: 2026-03-19"));
        assert!(result.contains("Thursday, March 19"));
    }

    #[test]
    fn template_file_read() {
        let tmp = TempDir::new().unwrap();
        let tpl_dir = tmp.path().join("Templates");
        fs::create_dir_all(&tpl_dir).unwrap();
        fs::write(tpl_dir.join("Daily.md"), "# {{title}}\n{{date}}").unwrap();

        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        let result =
            expand_template(tmp.path(), Path::new("Templates/Daily"), &d, "My Note").unwrap();
        assert!(result.contains("# My Note"));
        assert!(result.contains("2026-03-19"));
    }

    #[test]
    fn template_missing_file_returns_error() {
        let tmp = TempDir::new().unwrap();
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let result = expand_template(tmp.path(), Path::new("nope"), &d, "");
        assert!(result.is_err());
    }

    // ── try_parse_quarter_date ─────────────────────────────────────────

    #[test]
    fn parse_quarter_date_standard() {
        let chrono_fmt = momentjs_to_chrono("YYYY-[Q]Q");
        assert_eq!(
            try_parse_quarter_date("2026-Q1", &chrono_fmt),
            NaiveDate::from_ymd_opt(2026, 1, 1)
        );
        assert_eq!(
            try_parse_quarter_date("2025-Q4", &chrono_fmt),
            NaiveDate::from_ymd_opt(2025, 10, 1)
        );
    }

    #[test]
    fn parse_quarter_date_invalid() {
        let chrono_fmt = momentjs_to_chrono("YYYY-[Q]Q");
        assert!(try_parse_quarter_date("2026-Q5", &chrono_fmt).is_none());
        assert!(try_parse_quarter_date("2026-Q0", &chrono_fmt).is_none());
        assert!(try_parse_quarter_date("not-a-date", &chrono_fmt).is_none());
    }
}

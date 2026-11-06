//! Periodic notes: date format conversion, daily/weekly/monthly note resolution.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use super::fs::VaultRoot;
use super::path::{VaultPath, logical_join};
use chrono::{Datelike, Local, NaiveDate};
use regex::Regex;
use serde::Deserialize;

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
            if ch == '%' {
                result.push_str("%%");
            } else {
                result.push(ch);
            }
            pos += ch.len_utf8();
        }
    }

    for (idx, literal) in literals.iter().enumerate() {
        let placeholder = format!("\x00{idx}\x00");
        result = result.replace(&placeholder, &literal.replace('%', "%%"));
    }

    result
}

/// Format a `NaiveDate` using a Moment.js format string.
/// Converts to chrono internally and resolves the quarter placeholder.
fn format_date(date: &NaiveDate, moment_format: &str) -> VaultResult<String> {
    let chrono_fmt = momentjs_to_chrono(moment_format);
    let mut formatted = String::new();
    std::fmt::write(&mut formatted, format_args!("{}", date.format(&chrono_fmt))).map_err(
        |source| VaultError::InvalidDateFormat {
            format: moment_format.into(),
            source,
        },
    )?;

    if formatted.contains(QUARTER_PLACEHOLDER) {
        let quarter = (date.month() - 1) / 3 + 1;
        Ok(formatted.replace(QUARTER_PLACEHOLDER, &quarter.to_string()))
    } else {
        Ok(formatted)
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
    read_config_rooted(&VaultRoot::open(vault_root)?, period)
}

const PLUGIN_CONFIG_PATH: &str = ".obsidian/plugins/periodic-notes/data.json";
const DAILY_CONFIG_PATH: &str = ".obsidian/daily-notes.json";

pub(super) fn read_config_rooted(
    root: &VaultRoot,
    period: &NotePeriod,
) -> VaultResult<PeriodicNoteConfig> {
    resolve_config(
        read_json_config(root, Path::new(PLUGIN_CONFIG_PATH))?,
        || read_json_config(root, Path::new(DAILY_CONFIG_PATH)),
        period,
    )
}

/// Read and deserialize a JSON config file, returning `None` only if missing.
fn read_json_config<T: serde::de::DeserializeOwned>(
    root: &VaultRoot,
    path: &Path,
) -> VaultResult<Option<T>> {
    match root.read_file(path) {
        Ok(content) => parse_json_config(&content, path).map(Some),
        Err(error) if error.is_not_found() => Ok(None),
        Err(error) => Err(error),
    }
}

fn parse_json_config<T: serde::de::DeserializeOwned>(content: &str, path: &Path) -> VaultResult<T> {
    serde_json::from_str(content)
        .map_err(|error| VaultError::Other(format!("Failed to parse {}: {error}", path.display())))
}

/// Parse already-confined source bytes without assuming a native filesystem.
pub(super) fn read_config_strings(
    plugin: Option<&str>,
    daily: Option<&str>,
    period: &NotePeriod,
) -> VaultResult<PeriodicNoteConfig> {
    resolve_config(
        plugin
            .map(|content| parse_json_config(content, Path::new(PLUGIN_CONFIG_PATH)))
            .transpose()?,
        || {
            daily
                .map(|content| parse_json_config(content, Path::new(DAILY_CONFIG_PATH)))
                .transpose()
        },
        period,
    )
}

pub(super) fn plugin_config(
    raw: Option<&str>,
    period: &NotePeriod,
) -> VaultResult<Option<PeriodicNoteConfig>> {
    Ok(raw
        .map(|content| parse_json_config(content, Path::new(PLUGIN_CONFIG_PATH)))
        .transpose()?
        .and_then(|config| config_from_plugin(config, period)))
}

/// Resolve both source forms through the same precedence and conversion rules.
/// Core daily configuration is observed only when it can affect this period.
fn resolve_config(
    plugin: Option<PluginPeriodicConfig>,
    daily: impl FnOnce() -> VaultResult<Option<CoreDailyNotesConfig>>,
    period: &NotePeriod,
) -> VaultResult<PeriodicNoteConfig> {
    if let Some(config) = plugin.and_then(|config| config_from_plugin(config, period)) {
        return Ok(config);
    }
    if matches!(period, NotePeriod::Daily)
        && let Some(config) = daily()?
    {
        return Ok(PeriodicNoteConfig {
            format: config
                .format
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| default_format(period).into()),
            folder: config.folder.filter(|value| !value.is_empty()),
            template: config.template.filter(|value| !value.is_empty()),
        });
    }
    Ok(default_config(period))
}

fn config_from_plugin(
    plugin: PluginPeriodicConfig,
    period: &NotePeriod,
) -> Option<PeriodicNoteConfig> {
    let entry = match plugin {
        PluginPeriodicConfig::CalendarSets { calendar_sets } => calendar_sets
            .into_iter()
            .next()
            .and_then(|set| extract_from_calendar_set(set, period)),
        PluginPeriodicConfig::Legacy(legacy) => extract_from_legacy(*legacy, period),
    };
    entry
        .filter(|entry| entry.enabled.unwrap_or(false))
        .map(|entry| entry_to_config(entry, period))
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

pub(super) fn filter_recent_paths(
    config: &PeriodicNoteConfig,
    paths: Vec<PathBuf>,
    limit: usize,
) -> VaultResult<Vec<PathBuf>> {
    let folder = VaultPath::parse(config.folder.as_deref().unwrap_or(""))?;
    let format = momentjs_to_chrono(&config.format);
    let quarter = format.contains(QUARTER_PLACEHOLDER);
    let mut dated = Vec::new();
    for path in paths {
        let text = path
            .to_str()
            .ok_or_else(|| VaultError::InvalidPath("vault paths must be UTF-8".into()))?;
        let relative = if folder.as_str().is_empty() {
            text
        } else {
            text.strip_prefix(folder.as_str())
                .and_then(|suffix| suffix.strip_prefix('/'))
                .ok_or_else(|| VaultError::OutsideVault(path.clone()))?
        };
        let Some(stem) = relative.strip_suffix(".md") else {
            continue;
        };
        if let Some(date) = try_parse_date(stem, &format, quarter) {
            dated.push((date, path))
        }
    }
    dated.sort_by_key(|(date, _)| std::cmp::Reverse(*date));
    dated.truncate(limit);
    Ok(dated.into_iter().map(|(_, path)| path).collect())
}

// ── Path derivation ────────────────────────────────────────────────────

/// Derive the file path for a periodic note given a date.
/// Returns path relative to vault root.
pub fn periodic_note_path(config: &PeriodicNoteConfig, date: &NaiveDate) -> VaultResult<PathBuf> {
    let filename = format!("{}.md", format_date(date, &config.format)?);
    logical_join(
        Path::new(config.folder.as_deref().unwrap_or("")),
        Path::new(&filename),
    )
}

/// Get the file path for the current period's periodic note.
pub fn current_periodic_note_path(
    config: &PeriodicNoteConfig,
    _period: &NotePeriod,
) -> VaultResult<PathBuf> {
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
    list_rooted(&VaultRoot::open(vault_root)?, config, limit)
}

pub(super) fn list_rooted(
    root: &VaultRoot,
    config: &PeriodicNoteConfig,
    limit: usize,
) -> VaultResult<Vec<PathBuf>> {
    let folder = Path::new(config.folder.as_deref().unwrap_or(""));
    let directory = match root.resolve_existing(folder) {
        Ok(resolved) => resolved.relative,
        Err(error) if error.is_not_found() => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let chrono_fmt = momentjs_to_chrono(&config.format);
    let has_quarter = chrono_fmt.contains(QUARTER_PLACEHOLDER);
    let mut dated_paths = Vec::new();
    root.visit_entries(
        &directory,
        true,
        |_| true, // Explicit date formats can include dot-prefixed directories.
        || Ok(()),
        |path, is_file| {
            if is_file && path.extension().is_some_and(|ext| ext == "md") {
                let relative = path
                    .strip_prefix(&directory)
                    .map_err(|_| VaultError::OutsideVault(path.to_owned()))?;
                let stem = relative.with_extension("");
                let text = stem
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/");
                if let Some(date) = try_parse_date(&text, &chrono_fmt, has_quarter) {
                    dated_paths.push((date, folder.join(relative)));
                }
            }
            Ok(())
        },
    )?;
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
    expand_rooted(&VaultRoot::open(vault_root)?, template_path, date, title)
}

pub(super) fn expand_rooted(
    root: &VaultRoot,
    template_path: &Path,
    date: &NaiveDate,
    title: &str,
) -> VaultResult<String> {
    let relative_path = if template_path.extension().is_none() {
        template_path.with_extension("md")
    } else {
        template_path.to_path_buf()
    };
    let content = root.read_file(&relative_path)?;
    expand_template_string(&content, date, title)
}

static DATE_FORMAT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{date:([^}]+)\}\}").unwrap());

pub(super) fn expand_template_string(
    content: &str,
    date: &NaiveDate,
    title: &str,
) -> VaultResult<String> {
    let mut result = String::with_capacity(content.len());
    let mut cursor = 0;
    for captures in DATE_FORMAT_RE.captures_iter(content) {
        let matched = captures
            .get(0)
            .expect("regex capture always has a full match");
        result.push_str(&content[cursor..matched.start()]);
        result.push_str(&format_date(date, &captures[1])?);
        cursor = matched.end();
    }
    result.push_str(&content[cursor..]);
    let result = result.replace("{{title}}", title);
    let result = result.replace("{{date}}", &format_date(date, "YYYY-MM-DD")?);
    Ok(result.replace("{{time}}", &Local::now().format("%H:%M").to_string()))
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
        assert_eq!(format_date(&d, "YYYY-MM-DD").unwrap(), "2026-03-19");
    }

    #[test]
    fn format_date_quarter_q1() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q").unwrap(), "2026-Q1");
    }

    #[test]
    fn format_date_quarter_q2() {
        let d = NaiveDate::from_ymd_opt(2026, 6, 15).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q").unwrap(), "2026-Q2");
    }

    #[test]
    fn format_date_quarter_q3() {
        let d = NaiveDate::from_ymd_opt(2026, 9, 1).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q").unwrap(), "2026-Q3");
    }

    #[test]
    fn format_date_quarter_q4() {
        let d = NaiveDate::from_ymd_opt(2026, 12, 31).unwrap();
        assert_eq!(format_date(&d, "YYYY-[Q]Q").unwrap(), "2026-Q4");
    }

    #[test]
    fn format_date_monthly() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY-MM").unwrap(), "2026-03");
    }

    #[test]
    fn format_date_yearly() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(format_date(&d, "YYYY").unwrap(), "2026");
    }

    #[test]
    fn format_date_full_text() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(
            format_date(&d, "dddd, MMMM D, YYYY").unwrap(),
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
            periodic_note_path(&config, &d).unwrap(),
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
            periodic_note_path(&config, &d).unwrap(),
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
            periodic_note_path(&config, &d).unwrap(),
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
            periodic_note_path(&config, &d).unwrap(),
            PathBuf::from("2026-01-01.md")
        );
    }

    #[test]
    fn periodic_paths_preserve_source_names_and_slash_separators() {
        let config = PeriodicNoteConfig {
            format: "[a:]YYYY/MM/DD".into(),
            folder: Some("CON/Journal".into()),
            template: None,
        };
        let date = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        assert_eq!(
            periodic_note_path(&config, &date).unwrap().to_str(),
            Some("CON/Journal/a:2026/03/19.md")
        );
        assert_eq!(
            filter_recent_paths(
                &config,
                vec![
                    PathBuf::from("CON/Journal/a:2026/03/18.md"),
                    PathBuf::from("CON/Journal/a:2026/03/19.md"),
                    PathBuf::from("CON/Journal/a:2026/03/20.txt"),
                ],
                1,
            )
            .unwrap(),
            vec![PathBuf::from("CON/Journal/a:2026/03/19.md")]
        );
    }

    #[test]
    fn recent_source_paths_require_an_exact_folder_boundary() {
        let mut config = default_config(&NotePeriod::Daily);
        config.folder = Some("a:b".into());
        assert!(matches!(
            filter_recent_paths(&config, vec![PathBuf::from("a:bc/2026-03-19.md")], 10),
            Err(VaultError::OutsideVault(_))
        ));
        assert_eq!(
            filter_recent_paths(&config, vec![PathBuf::from("a:b/2026-03-19.md")], 10).unwrap(),
            vec![PathBuf::from("a:b/2026-03-19.md")]
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

    #[cfg(unix)]
    #[test]
    fn configuration_symlinks_cannot_import_external_settings() {
        use std::os::unix::fs::symlink;
        let vault = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let source = outside.path().join("settings.json");
        std::fs::write(&source, r#"{"enabled":true,"folder":"PRIVATE_CONTENT"}"#).unwrap();
        for name in [
            ".obsidian/plugins/periodic-notes/data.json",
            ".obsidian/daily-notes.json",
        ] {
            let path = vault.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            symlink(&source, &path).unwrap();
            let error = read_periodic_config(vault.path(), &NotePeriod::Daily).unwrap_err();
            assert!(!error.to_string().contains("PRIVATE_CONTENT"));
            std::fs::remove_file(path).unwrap();
        }
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

    #[test]
    fn confined_files_and_source_strings_share_configuration_resolution() {
        let cases = [
            (None, None),
            (None, Some(r#"{"format":"","folder":"Core","template":""}"#)),
            (
                Some(
                    r#"{"daily":{"enabled":true,"format":"","folder":"Plugin","templatePath":"Templates/Daily"},"weekly":{"enabled":true,"folder":"Weekly"},"monthly":{"enabled":true},"quarterly":{"enabled":true},"yearly":{"enabled":true}}"#,
                ),
                Some("invalid unused daily JSON"),
            ),
            (
                Some(
                    r#"{"daily":{"enabled":false,"folder":"Disabled"},"weekly":{"format":"YYYY","folder":"NotEnabled"}}"#,
                ),
                Some(r#"{"format":"DD-MM-YYYY","folder":"Core","template":"Templates/Core"}"#),
            ),
            (
                Some(
                    r#"{"calendarSets":[{"day":{"enabled":false},"week":{"enabled":true,"format":"GGGG-[W]WW","folder":"First"}},{"day":{"enabled":true,"folder":"Ignored"},"month":{"enabled":true,"folder":"Ignored"}}]}"#,
                ),
                Some(r#"{"format":"YYYY/MM/DD","folder":"Core"}"#),
            ),
        ];
        for (plugin, daily) in cases {
            let vault = TempDir::new().unwrap();
            for (path, content) in [(PLUGIN_CONFIG_PATH, plugin), (DAILY_CONFIG_PATH, daily)] {
                if let Some(content) = content {
                    let path = vault.path().join(path);
                    fs::create_dir_all(path.parent().unwrap()).unwrap();
                    fs::write(path, content).unwrap();
                }
            }
            let root = VaultRoot::open(vault.path()).unwrap();
            for period in [
                NotePeriod::Daily,
                NotePeriod::Weekly,
                NotePeriod::Monthly,
                NotePeriod::Quarterly,
                NotePeriod::Yearly,
            ] {
                let native = read_config_rooted(&root, &period).unwrap();
                let source = read_config_strings(plugin, daily, &period).unwrap();
                assert_eq!(
                    (native.format, native.folder, native.template),
                    (source.format, source.folder, source.template),
                );
            }
        }
    }

    #[test]
    fn source_and_native_configuration_errors_name_the_same_path() {
        for (plugin, daily, expected_path) in [
            (Some("{"), None, PLUGIN_CONFIG_PATH),
            (None, Some("{"), DAILY_CONFIG_PATH),
        ] {
            let vault = TempDir::new().unwrap();
            let path = vault.path().join(expected_path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "{").unwrap();
            let root = VaultRoot::open(vault.path()).unwrap();
            let native = read_config_rooted(&root, &NotePeriod::Daily).unwrap_err();
            let source = read_config_strings(plugin, daily, &NotePeriod::Daily).unwrap_err();
            assert_eq!(native.to_string(), source.to_string());
            assert!(source.to_string().contains(expected_path));
        }
    }

    #[test]
    fn plugin_precedence_avoids_observing_unused_core_configuration() {
        let vault = TempDir::new().unwrap();
        let plugin = r#"{"daily":{"enabled":true,"folder":"Plugin"}}"#;
        let path = vault.path().join(PLUGIN_CONFIG_PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, plugin).unwrap();
        // Reading a directory as a settings file must fail if it is consulted.
        fs::create_dir(vault.path().join(DAILY_CONFIG_PATH)).unwrap();
        let root = VaultRoot::open(vault.path()).unwrap();
        assert_eq!(
            read_config_rooted(&root, &NotePeriod::Daily)
                .unwrap()
                .folder
                .as_deref(),
            Some("Plugin")
        );
        assert_eq!(
            read_config_strings(Some(plugin), Some("{"), &NotePeriod::Daily)
                .unwrap()
                .folder
                .as_deref(),
            Some("Plugin")
        );
        assert_eq!(
            plugin_config(Some(plugin), &NotePeriod::Daily)
                .unwrap()
                .unwrap()
                .folder
                .as_deref(),
            Some("Plugin")
        );
        assert!(plugin_config(None, &NotePeriod::Daily).unwrap().is_none());
        assert!(plugin_config(Some("{"), &NotePeriod::Daily).is_err());
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

    #[test]
    fn list_rejects_folders_outside_vault() {
        let tmp = TempDir::new().unwrap();
        let vault = tmp.path().join("vault");
        let outside = tmp.path().join("outside/Daily");
        fs::create_dir_all(vault.join("Daily")).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("2026-03-19.md"), "outside note").unwrap();

        for folder in [
            "../outside/Daily",
            "Daily/../../outside/Daily",
            "../missing",
        ] {
            let mut config = default_config(&NotePeriod::Daily);
            config.folder = Some(folder.into());
            for limit in [0, 10] {
                assert!(matches!(
                    list_recent_periodic_notes(&vault, &config, limit),
                    Err(VaultError::OutsideVault(_))
                ));
            }
        }

        let mut config = default_config(&NotePeriod::Daily);
        config.folder = Some(outside.to_string_lossy().into_owned());
        assert!(matches!(
            list_recent_periodic_notes(&vault, &config, 10),
            Err(VaultError::InvalidPath(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn list_rejects_outside_folder_symlinks() {
        let tmp = TempDir::new().unwrap();
        let vault = tmp.path().join("vault");
        let outside = tmp.path().join("vault-other/Daily");
        fs::create_dir_all(&vault).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("2026-03-19.md"), "outside note").unwrap();
        std::os::unix::fs::symlink(&outside, vault.join("Daily")).unwrap();
        std::os::unix::fs::symlink(outside.parent().unwrap(), vault.join("Alias")).unwrap();

        for folder in ["Daily", "Alias/Daily", "Alias/Missing"] {
            let mut config = default_config(&NotePeriod::Daily);
            config.folder = Some(folder.into());
            assert!(matches!(
                list_recent_periodic_notes(&vault, &config, 10),
                Err(VaultError::OutsideVault(_))
            ));
        }

        if vault.join("daily").is_dir() {
            let mut config = default_config(&NotePeriod::Daily);
            config.folder = Some("daily".into());
            assert!(matches!(
                list_recent_periodic_notes(&vault, &config, 10),
                Err(VaultError::OutsideVault(_))
            ));
        }
    }

    #[cfg(unix)]
    #[test]
    fn list_allows_in_vault_folder_symlinks() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir(tmp.path().join("Journal")).unwrap();
        fs::write(tmp.path().join("Journal/2026-03-19.md"), "note").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("Journal"), tmp.path().join("Daily")).unwrap();
        let mut config = default_config(&NotePeriod::Daily);
        config.folder = Some("Daily".into());

        assert_eq!(
            list_recent_periodic_notes(tmp.path(), &config, 10).unwrap(),
            vec![PathBuf::from("Daily/2026-03-19.md")]
        );
    }

    #[test]
    fn list_preserves_native_folder_case_matching() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir(tmp.path().join("Daily")).unwrap();
        fs::write(tmp.path().join("Daily/2026-03-19.md"), "note").unwrap();
        let mut config = default_config(&NotePeriod::Daily);
        config.folder = Some("daily".into());

        let result = list_recent_periodic_notes(tmp.path(), &config, 10).unwrap();
        if tmp.path().join("daily").is_dir() {
            assert_eq!(result, vec![PathBuf::from("daily/2026-03-19.md")]);
        } else {
            assert!(result.is_empty());
        }
    }

    #[test]
    fn list_resolves_unicode_equivalent_folder_names() {
        let tmp = TempDir::new().unwrap();
        let decomposed = "Cafe\u{301}";
        fs::create_dir(tmp.path().join(decomposed)).unwrap();
        fs::write(tmp.path().join(decomposed).join("2026-03-19.md"), "note").unwrap();
        let mut config = default_config(&NotePeriod::Daily);
        config.folder = Some("Caf\u{e9}".into());

        assert_eq!(
            list_recent_periodic_notes(tmp.path(), &config, 10).unwrap(),
            vec![PathBuf::from("Caf\u{e9}/2026-03-19.md")]
        );
    }

    // ── expand_template ────────────────────────────────────────────────

    #[test]
    fn template_title_replacement() {
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let result = expand_template_string("Hello {{title}}!", &d, "World").unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn template_date_replacement() {
        let d = NaiveDate::from_ymd_opt(2026, 3, 19).unwrap();
        let result = expand_template_string("Date: {{date}}", &d, "").unwrap();
        assert_eq!(result, "Date: 2026-03-19");
    }

    #[test]
    fn template_date_with_custom_format() {
        let d = NaiveDate::from_ymd_opt(2026, 12, 25).unwrap();
        let result = expand_template_string("{{date:MMMM DD, YYYY}}", &d, "").unwrap();
        assert_eq!(result, "December 25, 2026");
    }

    #[test]
    fn template_time_has_hh_mm_pattern() {
        let d = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let result = expand_template_string("Time: {{time}}", &d, "").unwrap();
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
        )
        .unwrap();
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

    #[test]
    fn percent_literals_survive_periodic_and_template_formats() {
        let date = NaiveDate::from_ymd_opt(2026, 9, 8).unwrap();
        assert_eq!(
            format_date(&date, "YYYY-MM-DD [100%]").unwrap(),
            "2026-09-08 100%"
        );
        assert_eq!(
            expand_template_string("{{date:[100%] YYYY}}", &date, "").unwrap(),
            "100% 2026"
        );
    }

    #[test]
    fn date_only_formats_return_errors_for_time_directives() {
        let date = NaiveDate::from_ymd_opt(2026, 9, 8).unwrap();
        assert!(matches!(
            format_date(&date, "YYYY-MM-DD HH:mm"),
            Err(VaultError::InvalidDateFormat { .. })
        ));
        assert!(expand_template_string("{{date:HH:mm}}", &date, "").is_err());
    }
}

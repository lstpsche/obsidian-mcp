//! Wikilink, tag, and block-reference extraction and resolution.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::path::{canonical_unicode_key, logical_file_name};

/// Resolves wikilink targets to file paths using Obsidian's
/// shortest-unique-path algorithm.
///
/// Two lookup tables enable O(1) resolution:
/// - `by_stem`: lowercased filename stem -> all matching vault-relative paths
/// - `by_path`: lowercased vault-relative path (sans `.md`) -> canonical path
#[derive(Debug, Clone)]
pub struct LinkResolver {
    by_stem: HashMap<String, Vec<PathBuf>>,
    by_path: HashMap<String, PathBuf>,
}

/// Build a [`LinkResolver`] from a slice of vault-relative note paths.
pub fn build_link_resolver(note_paths: &[PathBuf]) -> LinkResolver {
    let mut resolver = LinkResolver {
        by_stem: HashMap::with_capacity(note_paths.len()),
        by_path: HashMap::with_capacity(note_paths.len()),
    };
    for path in note_paths {
        resolver.insert(path.clone());
    }
    resolver
}

impl LinkResolver {
    /// Resolve a wikilink target to a file path.
    ///
    /// Implements Obsidian's resolution rules:
    /// 1. If target contains `/`, try as relative path (`.md` extension optional)
    /// 2. If target has no `/`, search by filename stem across the entire vault —
    ///    unique match resolves, ambiguous (0 or 2+) returns `None`
    /// 3. All matching is case-insensitive
    pub fn resolve(&self, target: &str) -> Option<PathBuf> {
        let normalized = Self::normalize_target(target);
        if normalized.is_empty() {
            return None;
        }

        if normalized.contains('/') {
            self.by_path.get(&normalized).cloned()
        } else {
            match self.by_stem.get(&normalized)?.as_slice() {
                [single] => Some(single.clone()),
                _ => None,
            }
        }
    }

    /// Return all candidate paths for a target (useful when ambiguous).
    pub fn resolve_candidates(&self, target: &str) -> Vec<PathBuf> {
        let normalized = Self::normalize_target(target);
        if normalized.is_empty() {
            return Vec::new();
        }

        if normalized.contains('/') {
            self.by_path
                .get(&normalized)
                .map(|p| vec![p.clone()])
                .unwrap_or_default()
        } else {
            self.by_stem.get(&normalized).cloned().unwrap_or_default()
        }
    }

    /// Check if a wikilink target resolves to any existing note.
    pub fn is_resolved(&self, target: &str) -> bool {
        self.resolve(target).is_some()
    }

    /// Register a new file path in the resolver.
    pub fn add_path(&mut self, path: PathBuf) {
        self.insert(path);
    }

    /// Remove a file path from the resolver.
    pub fn remove_path(&mut self, path: &Path) {
        let stem = Self::stem_key(path);
        let pkey = Self::path_key(path);

        self.by_path.remove(&pkey);

        if let Some(paths) = self.by_stem.get_mut(&stem) {
            paths.retain(|p| p.as_os_str() != path.as_os_str());
            if paths.is_empty() {
                self.by_stem.remove(&stem);
            }
        }
    }

    /// Atomically move a path from `old` to `new` in both maps.
    pub fn rename_path(&mut self, old: &Path, new: PathBuf) {
        self.remove_path(old);
        self.insert(new);
    }

    fn insert(&mut self, path: PathBuf) {
        let stem = Self::stem_key(&path);
        let pkey = Self::path_key(&path);

        self.by_path.insert(pkey, path.clone());
        self.by_stem.entry(stem).or_default().push(path);
    }

    /// Normalize a wikilink target for lookup:
    /// Unicode-normalize, strip trailing `.md`, normalize path separators, lowercase.
    fn normalize_target(target: &str) -> String {
        lookup_key_without_md(&target.replace('\\', "/"))
    }

    /// Compute the path-based lookup key for a vault-relative path.
    /// `.md` files: path without `.md`, lowercased.
    /// Other files: full path, lowercased.
    fn path_key(path: &Path) -> String {
        lookup_key_without_md(&path.to_string_lossy().replace('\\', "/"))
    }

    /// Compute the stem-based lookup key for a vault-relative path.
    /// `.md` files: filename without `.md`, lowercased.
    /// Other files: full filename with extension, lowercased.
    fn stem_key(path: &Path) -> String {
        lookup_key_without_md(logical_file_name(path).unwrap_or_default())
    }
}

/// Normalize to NFC, lowercase, then strip `.md` suffix (if present).
/// Lowercasing first ensures case-insensitive `.md`/`.MD`/`.Md` stripping.
fn lookup_key_without_md(s: &str) -> String {
    let normalized = canonical_unicode_key(s);
    let lower = normalized.to_lowercase();
    lower.strip_suffix(".md").map(String::from).unwrap_or(lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn paths(strs: &[&str]) -> Vec<PathBuf> {
        strs.iter().map(|s| p(s)).collect()
    }

    // -- stem resolution --

    #[test]
    fn unique_stem_resolves() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve("hello"), Some(p("notes/hello.md")));
    }

    #[test]
    fn ambiguous_stem_returns_none() {
        let r = build_link_resolver(&paths(&["notes/hello.md", "archive/hello.md"]));
        assert_eq!(r.resolve("hello"), None);
    }

    #[test]
    fn ambiguous_stem_lists_all_candidates() {
        let r = build_link_resolver(&paths(&["notes/hello.md", "archive/hello.md"]));
        let mut c = r.resolve_candidates("hello");
        c.sort();
        assert_eq!(c, paths(&["archive/hello.md", "notes/hello.md"]));
    }

    #[test]
    fn nonexistent_stem_returns_none() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve("missing"), None);
        assert!(r.resolve_candidates("missing").is_empty());
    }

    // -- path resolution --

    #[test]
    fn path_resolves_without_md() {
        let r = build_link_resolver(&paths(&["notes/hello.md", "archive/hello.md"]));
        assert_eq!(r.resolve("notes/hello"), Some(p("notes/hello.md")));
        assert_eq!(r.resolve("archive/hello"), Some(p("archive/hello.md")));
    }

    #[test]
    fn path_resolves_with_md() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve("notes/hello.md"), Some(p("notes/hello.md")));
    }

    #[test]
    fn path_miss_returns_none() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve("wrong/hello"), None);
    }

    #[test]
    fn stem_resolves_across_unicode_normalization_forms() {
        let composed = "знаний.md";
        let decomposed: String = composed.nfd().collect();
        let decomposed_path = PathBuf::from(format!("notes/{decomposed}"));
        let r = build_link_resolver(std::slice::from_ref(&decomposed_path));

        assert_eq!(r.resolve("знаний"), Some(decomposed_path));
    }

    #[test]
    fn stem_resolves_from_decomposed_target_to_composed_path() {
        let composed_path = p("notes/знаний.md");
        let decomposed_target: String = "знаний".nfd().collect();
        let r = build_link_resolver(std::slice::from_ref(&composed_path));

        assert_eq!(r.resolve(&decomposed_target), Some(composed_path));
    }

    #[test]
    fn normalized_stem_ambiguity_returns_none() {
        let composed_path = p("notes/café.md");
        let decomposed: String = "café.md".nfd().collect();
        let decomposed_path = PathBuf::from(format!("archive/{decomposed}"));
        let r = build_link_resolver(&[composed_path, decomposed_path]);

        assert_eq!(r.resolve("café"), None);
    }

    #[test]
    fn path_resolves_across_unicode_normalization_forms() {
        let composed = "02_База-знаний/Сущности/lic1c.md";
        let decomposed: String = composed.nfd().collect();
        let decomposed_path = PathBuf::from(&decomposed);
        let r = build_link_resolver(std::slice::from_ref(&decomposed_path));

        assert_eq!(
            r.resolve("02_База-знаний/Сущности/lic1c"),
            Some(decomposed_path)
        );
    }

    // -- case insensitivity --

    #[test]
    fn stem_case_insensitive() {
        let r = build_link_resolver(&paths(&["Notes/Hello World.md"]));
        assert_eq!(r.resolve("hello world"), Some(p("Notes/Hello World.md")));
        assert_eq!(r.resolve("HELLO WORLD"), Some(p("Notes/Hello World.md")));
        assert_eq!(r.resolve("Hello World"), Some(p("Notes/Hello World.md")));
    }

    #[test]
    fn path_case_insensitive() {
        let r = build_link_resolver(&paths(&["Notes/Hello.md"]));
        assert_eq!(r.resolve("notes/hello"), Some(p("Notes/Hello.md")));
        assert_eq!(r.resolve("NOTES/HELLO"), Some(p("Notes/Hello.md")));
    }

    // -- .md extension handling --

    #[test]
    fn md_extension_stripped_from_target() {
        let r = build_link_resolver(&paths(&["hello.md"]));
        assert_eq!(r.resolve("hello"), Some(p("hello.md")));
        assert_eq!(r.resolve("hello.md"), Some(p("hello.md")));
    }

    // -- non-markdown files --

    #[test]
    fn non_md_requires_extension_in_stem_lookup() {
        let r = build_link_resolver(&paths(&["assets/image.png"]));
        assert_eq!(r.resolve("image.png"), Some(p("assets/image.png")));
        assert_eq!(r.resolve("image"), None);
    }

    #[test]
    fn non_md_path_lookup() {
        let r = build_link_resolver(&paths(&["assets/image.png"]));
        assert_eq!(r.resolve("assets/image.png"), Some(p("assets/image.png")));
    }

    // -- edge cases --

    #[test]
    fn empty_target_returns_none() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve(""), None);
        assert!(r.resolve_candidates("").is_empty());
    }

    #[test]
    fn empty_resolver() {
        let r = build_link_resolver(&[]);
        assert_eq!(r.resolve("anything"), None);
        assert!(!r.is_resolved("anything"));
    }

    #[test]
    fn root_level_note() {
        let r = build_link_resolver(&paths(&["note.md"]));
        assert_eq!(r.resolve("note"), Some(p("note.md")));
    }

    #[test]
    fn server_names_resolve_without_native_drive_interpretation() {
        let r = build_link_resolver(&paths(&["a:b.md", "notes/CON.md"]));
        assert_eq!(r.resolve("a:b"), Some(p("a:b.md")));
        assert_eq!(r.resolve("CON"), Some(p("notes/CON.md")));
        assert_eq!(r.resolve("notes/CON"), Some(p("notes/CON.md")));
    }

    #[test]
    fn deeply_nested_note() {
        let r = build_link_resolver(&paths(&["a/b/c/d/note.md"]));
        assert_eq!(r.resolve("note"), Some(p("a/b/c/d/note.md")));
        assert_eq!(r.resolve("a/b/c/d/note"), Some(p("a/b/c/d/note.md")));
    }

    // -- is_resolved --

    #[test]
    fn is_resolved_delegates_to_resolve() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert!(r.is_resolved("hello"));
        assert!(!r.is_resolved("missing"));
    }

    // -- mutations --

    #[test]
    fn add_path_makes_resolvable() {
        let mut r = build_link_resolver(&[]);
        assert!(!r.is_resolved("hello"));

        r.add_path(p("notes/hello.md"));
        assert_eq!(r.resolve("hello"), Some(p("notes/hello.md")));
        assert_eq!(r.resolve("notes/hello"), Some(p("notes/hello.md")));
    }

    #[test]
    fn remove_path_clears_both_maps() {
        let mut r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert!(r.is_resolved("hello"));

        r.remove_path(&p("notes/hello.md"));
        assert!(!r.is_resolved("hello"));
        assert_eq!(r.resolve("notes/hello"), None);
    }

    #[test]
    fn remove_from_ambiguous_restores_unique() {
        let mut r = build_link_resolver(&paths(&["notes/hello.md", "archive/hello.md"]));
        assert_eq!(r.resolve("hello"), None);

        r.remove_path(&p("archive/hello.md"));
        assert_eq!(r.resolve("hello"), Some(p("notes/hello.md")));
    }

    #[test]
    fn rename_updates_both_lookups() {
        let mut r = build_link_resolver(&paths(&["old/note.md"]));
        assert_eq!(r.resolve("old/note"), Some(p("old/note.md")));

        r.rename_path(&p("old/note.md"), p("new/note.md"));

        assert_eq!(r.resolve("old/note"), None);
        assert_eq!(r.resolve("new/note"), Some(p("new/note.md")));
        assert_eq!(r.resolve("note"), Some(p("new/note.md")));
    }

    #[test]
    fn rename_into_ambiguous() {
        let mut r = build_link_resolver(&paths(&["a/note.md"]));
        r.add_path(p("b/note.md"));
        assert_eq!(r.resolve("note"), None);

        r.rename_path(&p("b/note.md"), p("b/other.md"));
        assert_eq!(r.resolve("note"), Some(p("a/note.md")));
        assert_eq!(r.resolve("other"), Some(p("b/other.md")));
    }

    #[test]
    fn remove_nonexistent_is_harmless() {
        let mut r = build_link_resolver(&paths(&["notes/hello.md"]));
        r.remove_path(&p("does/not/exist.md"));
        assert!(r.is_resolved("hello"));
    }

    #[test]
    fn backslash_normalized_in_target() {
        let r = build_link_resolver(&paths(&["notes/hello.md"]));
        assert_eq!(r.resolve("notes\\hello"), Some(p("notes/hello.md")));
    }
}
